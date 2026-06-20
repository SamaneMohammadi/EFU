import torch
import torchvision
import numpy as np
from PIL import Image
try:
    from .utils import compute_partitions
except ImportError:
    from utils import compute_partitions

class _CIFAR10(torch.utils.data.Dataset):

	def __init__(self, root, train=True, transform=None, retrieve_ids=None, seed=2025, exclude_labels=None):
		self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=None)
		self.samples, self.labels = self.dataset.data, np.array(self.dataset.targets)
		self.transform = transform

		# Retrieve specific class IDs
		if retrieve_ids is not None:
			# Assume retrieve_ids is a list or array of sample indices.
			self.samples = self.samples[retrieve_ids]
			self.labels = self.labels[retrieve_ids]
		# Exclude specific labels
		if exclude_labels:
			exclude_labels = {exclude_labels} if isinstance(exclude_labels, int) else set(exclude_labels)
			mask = ~np.isin(self.labels, list(exclude_labels))
			self.samples, self.labels = self.samples[mask], self.labels[mask]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		x, y = self.samples[idx], self.labels[idx]
		x = Image.fromarray(x)
		if self.transform:
			x = self.transform(x)
		return x, y

	@property
	def num_classes(self):
		return len(set(self.labels))

	@property
	def name(self):
		return "cifar10"

	@property
	def class_names(self):
		return {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def load_cifar10(processor_fn, root='../../datasets/cifar10',  num_clients=None, batch_size=16, seed=2025, centralized=False, device='cuda', exclude_cids=None, partitions=None, **kwargs):
	# Partition dataset for clients
	partitions = compute_partitions(num_classes=10, exclude_labels=None, num_clients=num_clients, seed=seed)  if partitions is None else partitions
	exclude_labels = [i for s in [partitions[c] for c in exclude_cids] for i in s] if (exclude_cids and partitions) else None
	exclude_cids = [] if exclude_cids is None else exclude_cids
	assert len(partitions) == num_clients-len(exclude_cids)
	# Load test dataset
	test_ds = _CIFAR10(root=root, train=False, transform=processor_fn(train=False), exclude_labels=exclude_labels)
	num_classes = test_ds.num_classes
	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, pin_memory_device=device)

	if centralized:
		train_ds = _CIFAR10(root=root, train=True, transform=processor_fn(train=True), exclude_labels=exclude_labels)
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, pin_memory_device=device)
		return train_loader, test_loader, num_classes

	assert num_clients is not None
	assert num_clients <= num_classes, "Number of clients cannot exceed number of classes."

	labels = _CIFAR10(root=root, train=True, transform=None, exclude_labels=None).dataset.targets
	class_indices = [np.where(np.array(labels) == i)[0] for i in range(num_classes)]
	client_partitions = [list(np.concatenate([class_indices[i] for i in partitions[cid]])) for cid in range(num_clients)]

	train_loaders = []
	for cid in range(num_clients):
		client_ds = _CIFAR10(root=root, train=True, transform=processor_fn(train=True), retrieve_ids=client_partitions[cid])
		train_loaders.append(torch.utils.data.DataLoader(client_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, pin_memory_device=device) if len(client_ds) > 0 else None)

	return train_loaders, test_loader, num_classes


if __name__ == "__main__":
	from torchvision.utils import make_grid, save_image
	def load_fc_tf(**kwargs):
		return 	torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
	])
	train_ds, test_ds, num_classes = load_cifar10(processor_fn=load_fc_tf, root='../../../datasets/cifar10', num_clients=5, batch_size=128)

	#images, labels = next(iter(train_ds[0]))
	#grid = make_grid(images, nrow=4)
	#save_image(grid, "samples.png")

	from utils import store_ds_to_pickle
	store_ds_to_pickle(train_ds[0], '../../../reconstruct/assets/data/erased_cifar10_cid_0.pkl')