import torch
import torch.nn as nn
import torchvision
from .utils import inject_fn_to_network  # Make sure this is available in your package

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, grayscale=False, **kwargs):
        super(LeNet5, self).__init__()
        self.grayscale = grayscale
        self.num_classes = num_classes
        in_channels = 1 if self.grayscale else 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16 * in_channels, 120 * in_channels, kernel_size=5),
            nn.ReLU()
        )

        # Correct the classifier input dimension.
        # For grayscale: 120*1 = 120; for RGB: 120*3 = 360.
        self.classifier = nn.Sequential(
            nn.Linear(120 * in_channels, 120 * in_channels),
            nn.ReLU(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.ReLU(),
            nn.Linear(84 * in_channels, num_classes)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, return_z=False):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        if return_z:
            return logits, x
        return logits

def load_lenet_tf(model_name='lenet5', input_dim=32, train=True, **kwargs):
	if train:
		transform = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(input_dim, padding=4),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
		])
	else:
		transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
		])
	return transform

def load_lenet(model_name="lenet5", num_classes=10, verbose=False, grayscale=False, **kwargs):
	model = LeNet5(num_classes=num_classes, grayscale=grayscale)
	model = inject_fn_to_network(model, model_name)
	if num_classes is not None: model.set_clf(num_classes=num_classes, verbose=verbose)
	if verbose: model._print_trainable_parameters()
	# Add parameters for potential attacks
	model.cpa_attack_layer_index = 0
	model.model_type = "conv"
	model.attack_index = 0
	return model

# if __name__ == "__main__":
# 	model = load_lenet(num_classes=10, verbose=True, grayscale=False)
# 	print("Model device:", model.device)
# 	# Create a dummy input (batch of 16 RGB images of size 32x32)
# 	dummy_input = torch.randn(16, 3, 32, 32)
# 	logits, probas = model(dummy_input)
# 	print("Logits shape:", logits.shape)
# 	print("Probabilities shape:", probas.shape)