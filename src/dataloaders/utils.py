import numpy as np
import pickle
from PIL import Image
import torch
import torchvision

def store_ds_to_pickle(dataloader, file_path):
    data_pairs = []
    to_pil = torchvision.transforms.ToPILImage()
    for i in range(len(dataloader.dataset)):
        x, y = dataloader.dataset[i]
        # Ensure that x is a PIL image.
        if not isinstance(x, Image.Image):
            if isinstance(x, torch.Tensor):
                x = to_pil(x)
            elif isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            else:
                raise TypeError(f"Unsupported type for image at index {i}: {type(x)}")
        data_pairs.append((x, y))
    with open(file_path, 'wb') as f:
        pickle.dump(data_pairs, f)
    print(f"Saved {len(data_pairs)} (image, label) pairs to {file_path}")

def compute_partitions(num_classes, exclude_labels=None, num_clients=None, seed=None):
    # Create the full set of classes (0 to num_classes-1)
    all_classes = set(range(num_classes))

    # Exclude labels if provided
    if exclude_labels is not None:
        if isinstance(exclude_labels, int):
            exclude_labels = {exclude_labels}
        else:
            exclude_labels = set(exclude_labels)
        available_classes = sorted(all_classes - exclude_labels)
    else:
        available_classes = sorted(all_classes)

    # Shuffle the available classes reproducibly if a seed is provided.
    if seed is not None:
        rng = np.random.RandomState(seed)
        rng.shuffle(available_classes)

    # If no specific number of clients is provided, just return the available classes.
    if num_clients is None:
        return available_classes

    # Make sure we have enough available classes to partition among clients.
    if num_clients > len(available_classes):
        raise ValueError("Number of clients cannot exceed the number of available classes.")

    # Partition available classes among clients.
    available_classes = np.array(available_classes)
    splitted = np.array_split(available_classes, num_clients)
    partitions = [arr.tolist() for arr in splitted]

    return partitions
