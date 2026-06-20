import peft
import copy
import torch
import types
import opacus
import torchmetrics

def set_clf(self, num_classes, label2idx=None, verbose=False):
	target_model = self

	if isinstance(self, peft.PeftModel):
		target_model = self.base_model.model

	if hasattr(target_model, "classifier"):

		if isinstance(target_model.classifier, torch.nn.Sequential):
			in_features = target_model.classifier[-1].in_features
			target_model.classifier[-1] = torch.nn.Linear(in_features, num_classes).to(target_model.device)
		elif isinstance(target_model.classifier, torch.nn.Linear):
			in_features = target_model.classifier.in_features
			target_model.classifier = torch.nn.Linear(in_features, num_classes).to(target_model.device)
		else:
			raise ValueError("Model classifier type is not supported.")

		if verbose: print(f"Classifier updated: {in_features} -> {num_classes} classes")
	else:
		raise ValueError("Model does not have a classifier attribute.")
	self.label2class = label2idx if label2idx is not None else {i:i for i in range(num_classes)}
	return self

def prune_clf(self, erase_classes, verbose=False):
	target_model = self
	if isinstance(self, peft.PeftModel):
		target_model = self.base_model.model

	if not hasattr(target_model, "classifier"):
		raise ValueError("Model does not have a classifier attribute.")

	# Extract existing classifier details
	old_num_classes = target_model.classifier.out_features
	in_features = target_model.classifier.in_features
	erase_classes = sorted(set(erase_classes))

	if any(c >= old_num_classes for c in erase_classes):
		raise ValueError(f"Invalid class indices in erase_classes: {erase_classes}")

	# Indices of classes to keep
	keep_indices = [i for i in range(old_num_classes) if i not in erase_classes]
	new_num_classes = len(keep_indices)

	# Extract old classifier parameters
	old_weights = target_model.classifier.weight.data
	old_bias = target_model.classifier.bias.data if target_model.classifier.bias is not None else None

	# Create new classifier with pruned classes
	new_classifier = torch.nn.Linear(in_features, new_num_classes).to(target_model.device)
	new_classifier.weight = torch.nn.Parameter(old_weights[keep_indices, :])

	if old_bias is not None:
		new_classifier.bias = torch.nn.Parameter(old_bias[keep_indices])

	# Update classifier
	target_model.classifier = new_classifier
	self.label2class = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

	if verbose:
		print(f"Pruned classifier: {old_num_classes} -> {new_num_classes} classes (Removed {len(erase_classes)})")
	return len(self.label2class)

def enable_dp(self, verbose=False):
	if not opacus.validators.ModuleValidator.is_valid(self):
		if verbose: print("Patching model to be DP compatible.")
		self = opacus.validators.ModuleValidator.fix(self)
	return self

def set_weights(self, state_dict):
	for k, v in state_dict.items():
		if v.device != self.device:
			state_dict[k] = v.to(self.device, non_blocking=True)
	self.load_state_dict(state_dict, strict=True)
	return self

def get_weights(self):
	return {k: v.detach().clone().to("cpu", non_blocking=True) for k, v in copy.deepcopy(self.state_dict()).items()}

def evaluate(self, data, loss_fn=torch.nn.CrossEntropyLoss()):
	self.eval()
	acc, loss = torchmetrics.Accuracy(task="multiclass", num_classes=len(self.label2class)).to(self.device), torchmetrics.MeanMetric().to(self.device)
	with torch.no_grad():
		for (x, y) in data:
			x, y = x.to(self.device, non_blocking=True), torch.tensor([self.label2class[l.item()] for l in y], dtype=torch.long, device=self.device)
			logits = self(x)
			_loss = loss_fn(logits, y)
			loss.update(_loss.item()); acc.update(logits, y)
	return {"loss": round(loss.compute().item(),4), "accuracy": round(acc.compute().item(),2)}

def evaluate_classes(self, data, **kwargs):
	erased = not (len(self.label2class) == len(set(data.dataset.labels)))
	return self._evaluate_classes_after_erased(data=data, **kwargs) if erased else self._evaluate_classes_before_erased(data=data, **kwargs)

def _evaluate_classes_before_erased(self, data, loss_fn=torch.nn.CrossEntropyLoss(reduction='none'), erased_classes=[]):
	self.eval()
	num_classes = len(self.label2class)
	correct, total, loss_per_class = torch.zeros(num_classes, device=self.device), torch.zeros(num_classes, device=self.device), torch.zeros(num_classes, device=self.device)
	with torch.no_grad():
		for x, y in data:
			x, y = x.to(self.device, non_blocking=True), torch.tensor([self.label2class[l.item()] for l in y], dtype=torch.long, device=self.device)
			logits = self(x)
			_loss = loss_fn(logits, y).detach()
			preds = logits.argmax(dim=1)
			for i in range(num_classes):
				mask = y == i
				if mask.any():
					correct[i] += (preds[mask] == i).sum().item()
					total[i] += mask.sum().item()
					loss_per_class[i] += _loss[mask].sum().item()
	c_loss = (loss_per_class / total.clamp(min=1)).cpu().tolist()
	loss = loss_per_class.sum().item() / total.sum().item()
	c_acc = (correct / total.clamp(min=1)).cpu().tolist()
	acc = (correct.sum().item() / total.sum().item() * 100)
	erase_acc = sum(c_acc[i] for i in erased_classes if i < num_classes) / (len(erased_classes) or 1) if erased_classes else 0
	n_erase_acc = sum(c_acc[i] for i in range(num_classes) if i not in erased_classes) / (num_classes - len(erased_classes) or 1) if erased_classes else 0

	return {'accuracy': acc, 'loss': loss, 'erase_acc':erase_acc, 'n_erase_acc':n_erase_acc,
		"c_loss": {f"c_{i}":c_loss[i] for i in range(num_classes)}, "c_acc": {f"c_{i}":c_acc[i] for i in range(num_classes)}
	}

def _evaluate_classes_after_erased(self, data, loss_fn=torch.nn.CrossEntropyLoss(reduction='none'), erased_classes=[]):
	self.eval()
	num_classes = len(self.label2class)
	correct, total, loss_per_class = (torch.zeros(num_classes, device=self.device), torch.zeros(num_classes, device=self.device), torch.zeros(num_classes, device=self.device),)
	erased_classes = set(erased_classes)  # Convert to set for fast lookup
	available_classes = [i for i in range(num_classes) if i not in erased_classes]

	if not available_classes:
		raise ValueError("All classes are erased! At least one class must remain.")

	with torch.no_grad():
		for x, y in data:
			x, y = x.to(self.device), y.to(self.device)
			# Fix device mismatch
			erased_mask = torch.tensor([l.item() in erased_classes for l in y], device=y.device)
			non_erased_mask = (~erased_mask).to(y.device)
			# Process non-erased samples normally
			x_non_erased, y_non_erased = x[non_erased_mask], y[non_erased_mask]
			mapped_y_non_erased = torch.tensor([self.label2class[l.item()] for l in y_non_erased], dtype=torch.long, device=self.device)
			# Process erased samples (map them to random available classes)
			x_erased, y_erased = x[erased_mask], y[erased_mask]
			y_erased_mapped = (torch.tensor([random.choice(available_classes) for _ in range(len(y_erased))], dtype=torch.long, device=self.device) if len(x_erased) > 0 else torch.tensor([], dtype=torch.long, device=self.device))
			# Get logits for both non-erased and erased samples
			logits_non_erased = self(x_non_erased).logits if len(x_non_erased) > 0 else None
			logits_erased = self(x_erased).logits if len(x_erased) > 0 else None
			# Compute loss
			loss_non_erased = loss_fn(logits_non_erased, mapped_y_non_erased).detach() if logits_non_erased is not None else None
			loss_erased = loss_fn(logits_erased, y_erased_mapped).detach() if logits_erased is not None else None
			# Compute predictions
			preds_non_erased = logits_non_erased.argmax(dim=1) if logits_non_erased is not None else None
			preds_erased = logits_erased.argmax(dim=1) if logits_erased is not None else None
			# Update metrics for non-erased samples
			if logits_non_erased is not None:
				for i in range(num_classes):
					mask_class = mapped_y_non_erased == i
					if mask_class.any():
						correct[i] += (preds_non_erased[mask_class] == i).sum().item()
						total[i] += mask_class.sum().item()
						loss_per_class[i] += loss_non_erased[mask_class].sum().item()
			# Update metrics for erased samples
			erased_correct = (preds_erased == y_erased_mapped).sum().item() if logits_erased is not None else 0
			erased_total = len(y_erased_mapped)

	# Compute loss and accuracy
	c_loss = (loss_per_class / total.clamp(min=1)).cpu().tolist()
	loss = loss_per_class.sum().item() / total.sum().item() if total.sum().item() > 0 else 0
	c_acc = (correct / total.clamp(min=1)).cpu().tolist()
	acc = (correct.sum().item() / total.sum().item() * 100) if total.sum().item() > 0 else 0
	# Compute accuracy for erased samples (mapped to random classes)
	erase_acc = erased_correct / erased_total if erased_total > 0 else 0
	non_erased_classes = [i for i in range(num_classes) if i not in erased_classes]
	n_erase_acc = (sum(c_acc[i] for i in non_erased_classes) / (len(non_erased_classes) or 1)) if erased_classes else 0

	return {'accuracy': acc, 'loss': loss, 'erase_acc': erase_acc, 'n_erase_acc': n_erase_acc,
		"c_loss": {f"c_{i}": c_loss[i] for i in range(num_classes)}, "c_acc": {f"c_{i}": c_acc[i] for i in range(num_classes)}}

def _print_trainable_parameters(self):
	trainable_params, all_params = 0, 0
	for _, param in self.named_parameters():
		all_params += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(f"Trainable params: {trainable_params} || Total params: {all_params} || Trainable%: {100 * trainable_params / all_params:.2f}%")

def inject_fn_to_network(model, name):

	inject_fn_dict = {
		"set_clf": set_clf, "prune_clf": prune_clf,
		"enable_dp": enable_dp,
		"set_weights": set_weights, "get_weights": get_weights,
		"evaluate": evaluate, "evaluate_classes": evaluate_classes, "_evaluate_classes_before_erased": _evaluate_classes_before_erased, "_evaluate_classes_after_erased": _evaluate_classes_after_erased,
		"_print_trainable_parameters": _print_trainable_parameters,
	}

	for name, func in inject_fn_dict.items():
		setattr(model, name, types.MethodType(func, model))

	return model

def get_lora_target_modules(model_name, model):
	if model_name == "facebook/convnextv2-atto-1k-224":
		return ["pwconv1", "pwconv2"]
	elif model_name == "microsoft/resnet-18":
		return [n for n,m in model.named_modules() if isinstance(m, torch.nn.Conv2d) and "encoder" in n]
	else:
		raise ValueError(f"Unsupported model: {model_name}")




