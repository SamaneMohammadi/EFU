import peft
import types
import transformers
from .utils import inject_fn_to_network, get_lora_target_modules

def load_resnet18(model_name="microsoft/resnet-18", lora_rank=None, random=False, num_classes=None, verbose=False, **kwargs):

	model = transformers.AutoModelForImageClassification._from_config(transformers.AutoConfig.from_pretrained(model_name)) if random \
		else transformers.AutoModelForImageClassification.from_pretrained(model_name)

	# Adapt forward method
	model.backbone_name = model_name.split('/')[1].split('-')[0]
	def forward_wrapper(self, x, return_z=False):
		backbone_out = getattr(self, self.backbone_name)(x)
		embeddings = backbone_out.pooler_output
		embeddings.retain_grad()
		logits = self.classifier(embeddings)
		if return_z:
			return logits, embeddings
		return logits
	model.forward = types.MethodType(forward_wrapper, model)

	# Inject utility functions to model
	model = inject_fn_to_network(model, name=model_name.split('/')[1].split('-')[0])

	if verbose: model._print_trainable_parameters()

	if lora_rank is not None:
		model = peft.get_peft_model(model, peft.LoraConfig(r=lora_rank, lora_alpha=32, lora_dropout=0.05, target_modules=get_lora_target_modules(model_name, model)))
		if verbose: model._print_trainable_parameters()

	if num_classes is not None:
		model.set_clf(num_classes=num_classes, verbose=verbose)

	# Add parameters for attack
	model.cpa_attack_layer_index = -3
	model.model_type = "conv"
	model.attack_index = -3
	return model

def load_resnet18_tf(model_name="microsoft/resnet-18", **kwargs):
	return transformers.AutoImageProcessor.from_pretrained(model_name, use_fast=False)