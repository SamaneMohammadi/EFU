
import torch
import torchvision
from .utils import inject_fn_to_network

class FC2(torch.nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, num_classes=10):
        super(FC2, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1)
        )
        self.classifier = torch.nn.Linear(hidden_dim // 2, num_classes)

    @property
    def device(self):
        # Returns the device of the model's parameters.
        return next(self.parameters()).device

    def forward(self, x, return_z=False):
        # Flatten the input (batch, 3, 32, 32) to (batch, 3072)
        x = torch.flatten(x, start_dim=1)
        features = self.fc(x)
        logits = self.classifier(features)
        if return_z:
            return logits, features
        return logits

def load_fc_tf(model_name='fc', input_dim=32, train=True, **kwargs):
    if train:
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(input_dim, padding=4), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
    return transform

def load_fc(model_name="fc", num_classes=None, verbose=False, input_dim=3072, hidden_dim=512, **kwargs):
	model = FC2(input_dim=input_dim, hidden_dim=hidden_dim)
	model = inject_fn_to_network(model, model_name)
	if num_classes is not None: model.set_clf(num_classes=num_classes, verbose=verbose)
	if verbose: model._print_trainable_parameters()
	# Add parameters for attack
	model.cpa_attack_layer_index = 0
	model.model_type = "fc"
	model.attack_index = 0
	return model

if __name__ == "__main__":
	model = load_fc(num_classes=30, verbose=True)
