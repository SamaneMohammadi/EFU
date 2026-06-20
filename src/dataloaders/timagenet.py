import os
import torch
import torchvision

class TinyImageNet200(torchvision.datasets.ImageFolder):

	def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs,):
		subfolder = "train" if train else "val"
		root_sub = os.path.join(root, subfolder)
		if not os.path.exists(root):
			raise ValueError("Dataset not found at {}. Please download it from {}.".format(root, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"))
		super().__init__(root=root_sub, transform=transform, target_transform=target_transform,)

if __name__ == "__main__":
	root = '/home/isma/GitHub/encrypt_fu/datasets/tiny_imagenet/tiny-imagenet-200'
	ds = TinyImageNet200(root=root)
