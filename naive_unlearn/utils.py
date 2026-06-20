import torch
import typing
import random
import numpy as np
import collections
import numpy.typing as npt
from PIL import Image

class Subset(torch.utils.data.Subset):

	@property
	def num_classes(self):
		return self.dataset.num_classes

class DataLoader(torch.utils.data.DataLoader):

	@property
	def num_classes(self):
		return self.dataset.num_classes

class DirichletDataPartitioner:

	def __init__(self, dataset: torch.utils.data.Dataset, num_partitions: int,
		alpha: typing.Union[int, float, typing.List[float], np.ndarray], min_partition_size: int = 10,
		self_balancing: bool = False, shuffle: bool = True, seed: typing.Optional[int] = 42, custom_partitioner_function=None) -> None:

		self.dataset = dataset
		self._num_partitions = num_partitions
		self._check_num_partitions_greater_than_zero()
		self._alpha = self._initialize_alpha(alpha)
		self._min_partition_size: int = min_partition_size
		self._self_balancing = self_balancing
		self._shuffle = shuffle
		self._seed = seed
		self._rng = np.random.default_rng(seed=self._seed)
		
		# Utility attributes
		self._avg_num_of_samples_per_partition: typing.Optional[float] = None
		self._unique_classes: typing.Optional[typing.Union[typing.List[int], typing.List[str]]] = None
		self._partition_id_to_indices: typing.Dict[int, typing.List[int]] = {}
		self._partition_id_to_indices_determined = False

		# Additional custom code. Some datasets require separate code for partitioning, that we should override the defaults with instead.
		self.custom_partitioner_function = custom_partitioner_function

	@property
	def num_partitions(self) -> int:
		self._check_num_partitions_correctness_if_needed()
		self._determine_partition_id_to_indices_if_needed()
		return self._num_partitions

	def _check_num_partitions_greater_than_zero(self) -> None:
		if not self._num_partitions > 0:
			raise ValueError("The number of partitions needs to be greater than zero.")

	def _check_num_partitions_correctness_if_needed(self) -> None:
		if not self._partition_id_to_indices_determined:
			if self._num_partitions > self.dataset.num_rows:
				raise ValueError("The number of partitions needs to be smaller than the number of samples in the dataset.")
				
	def _initialize_alpha(self, alpha: typing.Union[int, float, typing.List[float], npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
		if isinstance(alpha, int):
			alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
		elif isinstance(alpha, float):
			alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
		elif isinstance(alpha, typing.List):
			if len(alpha) != self._num_partitions:
				raise ValueError("If passing alpha as a List, it needs to be of length of equal to num_partitions.")
			alpha = np.asarray(alpha)
		elif isinstance(alpha, np.ndarray):
			if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
				raise ValueError("If passing alpha as an NDArray, its length needs to be of length equal to num_partitions.")
			elif alpha.ndim == 2:
				alpha = alpha.flatten()
				if alpha.shape[0] != self._num_partitions:
					raise ValueError("If passing alpha as an NDArray, its size needs to be of length equal to num_partitions.")
		else:
			raise ValueError("The given alpha format is not supported.")
		if not (alpha > 0).all():
			raise ValueError(f"Alpha values should be strictly greater than zero. Instead it'd be converted to {alpha}")
		return alpha

	def _determine_partition_id_to_indices_if_needed(self) -> None:
		if self._partition_id_to_indices_determined:
			return

		if self.custom_partitioner_function is not None:
			self._partition_id_to_indices = self.custom_partitioner_function(self._num_partitions, len(self.dataset), shuffle_indices=True)
		else:
			# Generate information needed for Dirichlet partitioning
			targets = np.asarray(self.dataset.targets) #np.array([self.dataset[i][1] for i in range(len(self.dataset))])
			self._unique_classes = np.unique(targets).tolist()
			assert self._unique_classes is not None
			self._avg_num_of_samples_per_partition = len(self.dataset) / self._num_partitions

			# Repeat the sampling procedure based on the Dirichlet distribution until the min_partition_size is reached.
			sampling_try = 0
			while True:
				# Prepare data structure to store indices assigned to partition ids
				partition_id_to_indices: typing.Dict[int, typing.List[int]] = {nid: [] for nid in range(self._num_partitions)}
				# Iterate over all unique labels
				for k in self._unique_classes:
					# Access all the indices associated with class k
					indices_representing_class_k = np.where(targets == k)[0]
					# Determine division (the fractions) of the data representing class k among the partitions
					class_k_division_proportions = self._rng.dirichlet(self._alpha)
					nid_to_proportion_of_k_samples = {nid: class_k_division_proportions[nid] for nid in range(self._num_partitions)}
					# Balancing
					if self._self_balancing:
						assert self._avg_num_of_samples_per_partition is not None
						for nid in nid_to_proportion_of_k_samples.copy():
							if len(partition_id_to_indices[nid]) > self._avg_num_of_samples_per_partition:
								nid_to_proportion_of_k_samples[nid] = 0
						sum_proportions = sum(nid_to_proportion_of_k_samples.values())
						for nid in nid_to_proportion_of_k_samples:
							nid_to_proportion_of_k_samples[nid] /= sum_proportions
					# Determine the split indices
					cumsum_division_fractions = np.cumsum(list(nid_to_proportion_of_k_samples.values()))
					cumsum_division_numbers = cumsum_division_fractions * len(indices_representing_class_k)
					indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]
					split_indices = np.split(indices_representing_class_k, indices_on_which_split)
					# Append new indices (coming from class k) to the existing indices
					for nid in range(self._num_partitions):
						partition_id_to_indices[nid].extend(split_indices[nid].tolist())
				# Check if the indices assignment meets the min_partition_size
				min_sample_size_on_client = min(len(indices) for indices in partition_id_to_indices.values())
				if min_sample_size_on_client >= self._min_partition_size:
					break
				sampling_try += 1
				if sampling_try == 10:
					raise ValueError("The max number of attempts (10) was reached. Please update the values of alpha and try again.")

			# Shuffle the indices if shuffle is True
			if self._shuffle:
				for indices in partition_id_to_indices.values():
					self._rng.shuffle(indices)
			self._partition_id_to_indices = partition_id_to_indices

		self._partition_id_to_indices_determined = True

	def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
		self._determine_partition_id_to_indices_if_needed()
		indices = self._partition_id_to_indices[partition_id]
		return Subset(self.dataset, indices)

	def max_partition_size(self):
		self._determine_partition_id_to_indices_if_needed()
		return max(len(self._partition_id_to_indices[partition_id]) for partition_id in range(self._num_partitions))

	@property
	def num_partitions(self) -> int:
		self._determine_partition_id_to_indices_if_needed()
		return self._num_partitions

class FewShotDataset(torch.utils.data.Dataset):

	def __init__(self, subset: Subset, num_shots: int, seed: int = 42):

		self.subset = subset
		self.num_shots = num_shots
		self.seed = seed
		self._targets = np.asarray(subset.dataset.targets)[subset.indices]
		self.indices = self._select_few_shot_indices()

	@property
	def num_classes(self):
		return self.subset.dataset.num_classes

	def _select_few_shot_indices(self):
		random.seed(self.seed)
		class_indices = collections.defaultdict(list)
		
		for idx, target in zip(self.subset.indices, self._targets):
			class_indices[target].append(idx)

		few_shot_indices = []
		for _, indices in class_indices.items():
			if len(indices) <= self.num_shots:
				few_shot_indices.extend(indices)
			else:
				few_shot_indices.extend(random.sample(indices, self.num_shots))
		return few_shot_indices

	def __getitem__(self, idx):
		original_idx = self.indices[idx]
		return self.subset.dataset[original_idx]

	def __len__(self):
		return len(self.indices)

class FederatedDataset(torch.utils.data.Dataset):

	def __init__(self, dataloader, root, transform=None, few_shot=False, allow_few_shot_test=False, num_shots=10, # dataset params
		alpha=0.5, num_partitions=10, min_partition_size=10, self_balancing=True, # federated params
		shuffle=False, seed=42, # random-ness params
		**kwargs,
		):

		self._dataloader = dataloader
		self._dataloader_args = {'root':root, 'transform':transform, 'seed':seed,}

		# Add all dataset-specific params
		self._dataloader_args.update(kwargs)

		# Create train dataloader
		self.train_ds = self._dataloader(train=True,
			few_shot=False, allow_few_shot_test=False, # Only apply few-shot after partitioning data.
			**self._dataloader_args)

		custom_partitioner_function = self.train_ds.custom_partitioner_function if hasattr(self.train_ds, 'custom_partitioner_function') else None

		# Create data partitioner
		self.partitioner = DirichletDataPartitioner(dataset=self.train_ds,
			num_partitions=num_partitions, alpha=alpha, min_partition_size=min_partition_size,
			self_balancing=self_balancing, shuffle=shuffle, seed=seed, custom_partitioner_function=custom_partitioner_function)

		# Few-shot params
		self._few_shots = few_shot
		self._num_shots = num_shots
		self._allow_few_shot_test = allow_few_shot_test
		self._seed = seed

	@property
	def num_classes(self):
		return self.train_ds.num_classes

	@property
	def classes(self):
		return list(self.train_ds._classes.keys())

	@property
	def max_client_batch(self):
		return self.partitioner.max_partition_size()

	def __getitem__(self, idx):
		return self.train_ds[idx]

	def __len__(self):
		return len(self.train_ds)

	def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
		partition = self.partitioner.load_partition(partition_id)
		if self._few_shots and isinstance(self._num_shots,int):
			partition = FewShotDataset(partition, num_shots=self._num_shots, seed=self._seed)
		return partition

	def load_test_set(self) -> torch.utils.data.Dataset:
		test_ds = self._dataloader(train=False,
			few_shot=self._few_shots, allow_few_shot_test=self._allow_few_shot_test, num_shots=self._num_shots,
			**self._dataloader_args)
		return test_ds

	def load_validation_set(self) -> torch.utils.data.Dataset:
		"""
		Note: this function should only be used when a dataset explicitly has support for a validation set.
		"""
		valid_ds = self._dataloader(train=False, validation=True, few_shot=self._few_shots, allow_few_shot_test=self._allow_few_shot_test,
			num_shots=self._num_shots, **self._dataloader_args)
		return valid_ds

class ImageDataset(torch.utils.data.Dataset):

	def __init__(self, data, targets, train=True, transform=None, few_shot=False, num_shots=10, seed=42, **kwargs):

		self.data, self.targets = data, targets
		if few_shot and isinstance(num_shots, int) and train:
			self.data, self.targets = self.few_shot_data(num_shots=num_shots, seed=seed)
		self._classes = sorted(set(self.targets))
		self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}
		self.transform = transform

	@property
	def num_classes(self):
		return len(self._classes)

	def __getitem__(self, idx):
		if self.transform is not None or self.data_processor is not None:
			with Image.fromarray(self.data[idx]) as x:
				if self.transform is not None:
					x = self.transform(x)
				return {'image':x, 'label':self.targets[idx]}

	def __len__(self):
		return len(self.targets)

	def few_shot_data(self, num_shots=10, seed=42):
		random.seed(seed)
		unique_labels = list(set(self.targets))
		new_fps = []
		new_labels = []
		for label in unique_labels:
			indices = [i for i, x in enumerate(self.targets) if x == label]
			selected_indices = indices if num_shots>len(indices) else random.sample(indices, num_shots)
			for i in selected_indices:
				new_fps.append(self.data[i])
				new_labels.append(self.targets[i])
		return new_fps, new_labels