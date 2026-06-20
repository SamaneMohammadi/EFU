import os
import utils # Custom module
import torchvision

os.environ['DATASETS'] = '../datasets'

##########################################################################################

class CIFAR10(utils.ImageDataset):

	custom_collate_fn = None
	train_set_size = 50_000

	def __init__(self, root_dir='../datasets/cifar10', train=True, few_shot=False, num_shots=10, seed=42, download=False, **kwargs):

		self.root_dir = os.path.abspath(root_dir)

		if train:
			kwargs['transform'] = torchvision.transforms.Compose([
				torchvision.transforms.RandomCrop(32,padding=4),
				torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
				torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
				torchvision.transforms.RandomHorizontalFlip(p=0.5),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])
		else:
			kwargs['transform'] = torchvision.transforms.Compose([
				torchvision.transforms.CenterCrop(32),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])

		temp = torchvision.datasets.CIFAR10(root=self.root_dir, train=train, download=download)
		super().__init__(data=temp.data, targets=temp.targets, train=train, few_shot=few_shot, num_shots=num_shots, seed=seed, **kwargs)
		del temp

##########################################################################################

def available_datasets():
	return {'cifar10': (CIFAR10, 'cifar10'),}

def load_data(name='cifar100', num_partitions=10, num_shots=None, min_num_samples=10, split='iid', seed=42, transform=None):

	assert name in list(available_datasets().keys()), 'Dataset `{}` is not available. Available datasets are `{}`'.format(name, available_datasets())

	if split == 'iid':
		alpha = 100000.0
	elif split == 'noniid':
		alpha = 0.5
	elif isinstance(split, int):
		alpha = split
	else:
		ValueError('`split` can be either `iid`, `noniid`, `int` or `float`. Passed {} of type {}'.format(split, type(split)))

	ds, folder_name = available_datasets()[name]

	return utils.FederatedDataset(dataloader=ds,
			root=os.path.join(os.environ['DATASETS'], folder_name), # NOTE: Define `TORCH_DATA_DIR` in ENV parameters before running.
			transform=transform, # NOTE: You can override the default transformation used.
			few_shot=True if num_shots is not None else False, allow_few_shot_test=False, num_shots=num_shots,
			alpha=alpha, # NOTE: Determine data distribution among partitions
			num_partitions=num_partitions, min_partition_size=min_num_samples,
			self_balancing=True, shuffle=False, seed=seed)

if __name__ == "__main__":
	ds = load_data('cifar10')