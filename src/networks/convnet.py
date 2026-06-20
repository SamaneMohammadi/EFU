import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .utils import inject_fn_to_network  # Make sure this is available in your package

class ConvNet(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        super(ConvNet, self).__init__()
        # Feature extractor block
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
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


def load_convnet_tf(model_name='convnet', input_dim=32, train=True, **kwargs):
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

def load_convnet(model_name="convnet", num_classes=10, verbose=False, grayscale=False, **kwargs):
	model = ConvNet(num_classes=num_classes, grayscale=grayscale)
	model = inject_fn_to_network(model, model_name)
	if num_classes is not None: model.set_clf(num_classes=num_classes, verbose=verbose)
	if verbose: model._print_trainable_parameters()
	# Add parameters for potential attacks
	model.cpa_attack_layer_index = 0
	model.model_type = "conv"
	model.attack_index = 0
	return model


if __name__ == '__main__':
    model = ConvNet(num_classes=10)
    dummy_input = torch.randn(8, 3, 32, 32)  # Batch size 8, RGB images of size 32x32
    logits, loss = model(dummy_input, targets=torch.randint(0, 10, (8,)))
    print("Logits shape:", logits.shape)
    if loss is not None:
        print("Loss:", loss.item())