
from .lfw import LFW, load_lfw
from .cifar10 import _CIFAR10, load_cifar10

datasets = {
    'cifar10': _CIFAR10,
    'lfw':		LFW,
}

dataloaders = {
    'cifar10': load_cifar10,
    'lfw':		load_lfw,
}
