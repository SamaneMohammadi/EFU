import torch
import torch.nn as nn
import torchvision
import types
from .utils import inject_fn_to_network  # Ensure this is available in your package

def load_vgg16_tf(model_name='vgg16', input_dim=224, train=True, **kwargs):
    if train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_dim), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_dim), torchvision.transforms.CenterCrop(input_dim), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def load_vgg16(model_name="vgg16", num_classes=10, verbose=False, **kwargs):
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = inject_fn_to_network(model, model_name)

    def forward_wrapper(self, x, return_z=False):
        x = self.features(x)
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        logits = self.classifier(z)
        if return_z:
            return logits, z
        return logits
    model.forward = types.MethodType(forward_wrapper, model)

    # Inject a device property into the model's class.
    @property
    def device(self):
        return next(self.parameters()).device
    setattr(model.__class__, 'device', device)

    if num_classes is not None and hasattr(model, "set_clf"):
        model.set_clf(num_classes=num_classes, verbose=verbose)
    if verbose and hasattr(model, "_print_trainable_parameters"):
        model._print_trainable_parameters()

    # Add attributes for potential attack parameters.
    model.cpa_attack_layer_index = 26
    model.model_type = "conv"
    model.attack_index = 26
    return model

if __name__ == '__main__':
    model = load_vgg16(num_classes=10, verbose=True)
    print(model)
    dummy_input = torch.randn(8, 3, 224, 224)  # Batch size 8
    output = model(dummy_input)
    print("Output shape:", output.shape)