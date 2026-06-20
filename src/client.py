import copy
import torch
import random
import opacus
import torchmetrics
import transformers
from utils.apgd import apgd
from utils.mas import get_weights_importance

def local_train(model, data, lr=3e-4, num_epochs=1, loss_fn=torch.nn.CrossEntropyLoss()):
	model.train()
	num_classes = len(model.label2class)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
	acc, loss = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(model.device), torchmetrics.MeanMetric().to(model.device)
	for _ in range(num_epochs):
		acc.reset(); loss.reset()
		for (x, y) in data:
			x, y = x.to(model.device, non_blocking=True), torch.tensor([model.label2class[l.item()] for l in y], dtype=torch.long, device=model.device)
			optimizer.zero_grad()
			logits = model(x)
			_loss = loss_fn(logits, y).float()
			_loss.backward(); optimizer.step()
			loss.update(_loss.detach().item()); acc.update(logits, y)
	return model, len(data.dataset), {"loss": loss.compute().item(), "accuracy": acc.compute().item()}

def local_unlearn(model, data, num_epochs=1, lr=1e-3, lambda_imp=1.0, eps=0.4):
	model.train()
	num_classes = len(model.label2class)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
	acc, loss = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(model.device), torchmetrics.MeanMetric().to(model.device)
	# Copy the model for adversarial sample generation
	adv_model = copy.deepcopy(model).to(model.device)
	adv_model.eval()
	# Store original trainable weights (θ̃)
	reference_weights = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
	# Compute weight importance Ω
	importance = get_weights_importance(model, data)
	for _ in range(num_epochs):
		acc.reset(); loss.reset()
		for x, y in data:
			x, y = x.to(model.device, non_blocking=True), torch.tensor([model.label2class[l.item()] for l in y], dtype=torch.long, device=model.device)
			optimizer.zero_grad()
			logits = model(x)
			loss_unlearn = -torch.nn.functional.cross_entropy(logits, y)
			loss.update(loss_unlearn.detach().item()); acc.update(logits, y)
			adv_y = torch.tensor([random.choice([cls for cls in range(data.dataset.num_classes) if cls != label.item()]) for label in y], device=model.device)
			adv_x = apgd(model=adv_model, inputs=x, labels=adv_y, eps=eps, norm=2, targeted=True, n_iter=100, n_restarts=1, loss_function="ce")
			loss_adv = torch.nn.functional.cross_entropy(model(adv_x), adv_y)
			reg_loss = sum((importance[n] * (p - reference_weights[n]).pow(2)).sum() for n, p in model.named_parameters() if n in importance)
			(loss_unlearn + loss_adv + lambda_imp*reg_loss).backward()
			optimizer.step()
	return model, len(data.dataset), {"loss": loss.compute().item(), "accuracy": acc.compute().item()}

def local_train_dp(model, data, lr=3e-4, num_epochs=10, loss_fn=torch.nn.CrossEntropyLoss(), delta=1e-5, max_grad=1.5, max_noise=1.0, verbose=False):
	model.train()
	# Training variables
	device = model.device
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
	scheduler = transformers.get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=10, num_training_steps=len(data) * num_epochs)
	acc, loss, epsilon = torchmetrics.Accuracy(task="multiclass", num_classes=data.dataset.num_classes).to(device), \
		torchmetrics.MeanMetric().to(device), torchmetrics.MaxMetric().to(device)
	# Enable DP
	dp = opacus.PrivacyEngine(secure_mode=False)
	model, optimizer, data = dp.make_private(module=model, optimizer=optimizer,data_loader=data, max_grad_norm=max_grad, noise_multiplier=max_noise)
	# Train loop
	for e in range(num_epochs):
		acc.reset(); loss.reset(); # epsilon.reset() # reset metrics
		for _, (x,y) in enumerate(data):
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			optimizer.zero_grad()
			logits = model(x)
			_loss = loss_fn(logits, y)
			_loss.backward(); optimizer.step(); scheduler.step() # Update model
			loss.update(_loss.detach().item()); acc.update(logits, y); epsilon.update(dp.get_epsilon(delta)) # Update metrics
		if verbose: print(f"Epoch {e+1}/{num_epochs} - Loss: {loss.compute().item():.4f}, Accuracy: {acc.compute().item():.2%}, (ε:{epsilon.compute().item():.2f}, δ:{delta})")
	return model._module, {'loss':loss.compute().item(), 'accuracy': acc.compute().item(), 'epsilon': epsilon.compute().item(), 'delta': delta}
