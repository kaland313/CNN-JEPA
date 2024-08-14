# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import h5py
import torch
import time
import resource
import tracemalloc
import numpy as np
import io
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Function to save ImageNet dataset to HDF5 based on https://blade6570.github.io/soumyatripathy/hdf5_blog.html
def save_imagenet_to_hdf5(dataset_root, output_file, num_workers=4):
    def binary_loader(img_path):
        with open(img_path, 'rb') as img_f:
            binary_data = img_f.read()      # read the image as python binary
            binary_data_np = np.asarray(binary_data)
            return binary_data_np
    imagenet_data = datasets.ImageFolder(root=dataset_root, loader=binary_loader)

    # Initialize HDF5 file
    if os.path.isfile(output_file):
        os.remove(output_file)
    with h5py.File(output_file, 'w') as hf:
        # Create datasets in HDF5 file
        images_dset = hf.create_group('images')
        labels_dset = hf.create_group('labels')

        # Use DataLoader to load images in parallel, but they are in binary format, and can't stack them 
        dataloader = DataLoader(imagenet_data, batch_size=32, num_workers=num_workers, collate_fn=lambda x: x)

        # Save data to HDF5 file
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Saving to {output_file}"):
            for j, (image, label) in enumerate(batch):
                images_dset.create_dataset(f"{i*32+j}", data=image)
                labels_dset.create_dataset(f"{i*32+j}", data=label)

# Define custom dataset class to load data from HDF5 file
class HDF5ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        """A custom PyTorch dataset class for loading images and labels from an HDF5 file.

        Args:
        root (str): Path to the HDF5 file containing images and labels.
        transform (callable, optional): A function/transform that takes in an image
            and returns a transformed version. Default: None.
        """
        self.transform = transform
        self.hdf5_file = h5py.File(root, 'r', driver='sec2')

        self.images = self.hdf5_file['images']
        self.labels = self.hdf5_file['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = str(idx)
        # Why using [()]: https://github.com/h5py/h5py/issues/1779#issuecomment-743447638
        # Why using .copy(): https://github.com/h5py/h5py/issues/2010 
        # Without .copy(), online eval fails with omm error 
        bytes = self.images[idx][()].copy()
        label = self.labels[idx][()].copy()
        image = Image.open(io.BytesIO(np.array(bytes)))
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

  
if __name__ == "__main__":
    data_root = '/data'
    dataset = 'imagenet-100'
    # for split in ['val','train']:
    #     dataset_root = os.path.join(data_root, dataset, split)
    #     output_file = os.path.join(data_root, f'{dataset}-{split}.h5')
    #     save_imagenet_to_hdf5(dataset_root, output_file, num_workers=4)

    # HDF5 file generation times
    # | Dataset      | Split | Time     |
    # |--------------|-------|----------|
    # | imagenet-100 | val   | 25 sec   |
    # | imagenet-100 | train | 9.5 min  |
    # | imagenet-1k  | val   | 4 min    |
    # | imagenet-1k  | train | 1h 35 min|

    # Evaluate performance
    tracemalloc.start()
    output_file = os.path.join(data_root, f'{dataset}-train.h5')
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.2, 1.0), interpolation=3
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    custom_dataset = HDF5ImageFolder(output_file, transform=transform)
    dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True, num_workers=12)

    start = time.time()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass
    print(f"Time to iterate over the dataloader: {time.time()-start:.2f} seconds")
    print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak memory usage: {peak/10**6:.2f} MB")

    # Repeat with a standard ImageFolder dataset
    # standard_dataset = datasets.ImageFolder(root=os.path.join(data_root, dataset, 'train'), transform=transform)
    # standard_dataloader = DataLoader(standard_dataset, batch_size=32, shuffle=True, num_workers=12)
    # start = time.time()
    # for i, batch in tqdm(enumerate(standard_dataloader), total=len(standard_dataloader)):
    #     pass
    # print(f"Time to iterate over the standard dataloader: {time.time()-start:.2f} seconds")


    # Performance comparison on imagenet-100 train dataset
    # | Dataset type                                               | Time      | Memory usage |
    # |------------------------------------------------------------|-----------|--------------|
    # | Custom HDF5                                                |   3.7 min |              |
    # | Standard ImageFolder, ext3 image mounted using fuse2fs     | 22.83 min |              |
    # | Custom HDF5 in container                                   |   1.8 min |  399.41 MB   |
    # | Custom HDF5 in container 'sec2' driver                     |     2 min |  398.05 MB   |
    # | Standard ImageFolder, ext3 image mounted using singularity |  12.2 min |              |
    # Note: the in-container experiments might be improved by caching

    # Performance comparison on imagenet train dataset
    # | Dataset type                                               | Time      | Memory usage |
    # |------------------------------------------------------------|-----------| -------------|
    # | Custom HDF5 in container                                   | 53 min    |  491.46 MB   |
