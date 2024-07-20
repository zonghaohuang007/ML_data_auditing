"""Data class, holding information about dataloaders and mark ids."""

import torch
import numpy as np

import pickle

import datetime
import os
import warnings
import random
import PIL
import shutil

from .datasets import construct_datasets, Subset
from .cached_dataset import CachedDataset

from .diff_data_augmentation import RandomTransform

from ..consts import PIN_MEMORY, BENCHMARK, DISTRIBUTED_BACKEND, SHARING_STRATEGY, MAX_THREADING
from ..utils import set_random_seed
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class Kettle():
    """Brew mark with given arguments.

    Data class.
    Attributes:
    - trainloader
    - validloader
    - markloader
    - mark_ids
    - trainset/markset/targetset

    Most notably .mark_lookup is a dictionary that maps image ids to their slice in the mark_delta tensor. (idx, num)

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_mark
    - export_mark

    """

    def __init__(self, args, batch_size, augmentations, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.trainset, self.validset = self.prepare_data(normalize=True)
        num_workers = self.get_num_workers()

        if self.args.lmdb_path is not None:
            from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb
            self.trainset = LMDBDataset(self.trainset, self.args.lmdb_path, 'train')
            self.validset = LMDBDataset(self.validset, self.args.lmdb_path, 'val')

        if self.args.cache_dataset:
            self.trainset = CachedDataset(self.trainset, num_workers=num_workers)
            self.validset = CachedDataset(self.validset, num_workers=num_workers)
            num_workers = 0

        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=min(self.batch_size, len(self.validset)),
                                                       shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

        # self.print_status()


    """ STATUS METHODS """

    def print_status(self):
        print(
            f'budget of {self.args.budget * 100}% images:')
        print(f'--Mark images drawn from all classes.')


    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 4 * num_gpus
        else:
            max_num_workers = 4
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        print(f'Data is loaded with {worker_count} workers.')
        return worker_count

    """ CONSTRUCTION METHODS """

    def prepare_data(self, normalize=True):
        trainset, validset = construct_datasets(self.args.dataset, self.args.data_path, normalize)

        # Prepare data mean and std for later:
        self.dm = torch.tensor(trainset.data_mean)[None, :, None, None].to(**self.setup)
        self.ds = torch.tensor(trainset.data_std)[None, :, None, None].to(**self.setup)


        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.augmentations is not None or self.args.paugment:
            if 'CIFAR' in self.args.dataset:
                params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
            elif 'MNIST' in self.args.dataset:
                params = dict(source_size=28, target_size=28, shift=4, fliplr=True)
            elif 'TinyImageNet' in self.args.dataset:
                params = dict(source_size=64, target_size=64, shift=64 // 4, fliplr=True)
            elif 'ImageNet' in self.args.dataset:
                params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)

            if self.augmentations == 'default':
                self.augment = RandomTransform(**params, mode='bilinear')
            elif not self.augmentations:
                print('Data augmentations are disabled.')
                self.augment = RandomTransform(**params, mode='bilinear')
            else:
                raise ValueError(f'Invalid diff. transformation given: {self.augmentations}.')

        return trainset, validset
    

    """ EXPORT METHODS """

    def export_data(self, path=None, mode='automl'):
        """Export marks in either packed mode (just ids and raw data) or in full export mode, exporting all images.

        In full export mode, export data into folder structure that can be read by a torchvision.datasets.ImageFolder

        In automl export mode, export data into a single folder and produce a csv file that can be uploaded to
        google storage.
        """
        if path is None:
            path = self.args.mark_path

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location."""
            filename = os.path.join(location, str(idx) + '.png')

            _torch_to_PIL(input).save(filename)

        # Save either into packed mode, ImageDataSet Mode or google storage mode
        if mode == 'full':
            # Save training set
            names = self.trainset.classes
            for name in names:
                if os.path.isdir(os.path.join(path, 'train', name)):
                    shutil.rmtree(os.path.join(path, 'train', name))
                if os.path.isdir(os.path.join(path, 'test', name)):
                    shutil.rmtree(os.path.join(path, 'test', name))
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)

        elif mode == 'numpy':
            _, h, w = self.trainset[0][0].shape
            training_data = np.zeros([len(self.trainset), h, w, 3])
            labels = np.zeros(len(self.trainset))
            for input, label, idx in self.trainset:
                training_data[idx] = np.asarray(_torch_to_PIL(input))
                labels[idx] = label

            np.save(os.path.join(path, 'marked_training_data.npy'), training_data)
            np.save(os.path.join(path, 'marked_training_labels.npy'), labels)

        else:
            raise NotImplementedError()

        print('Dataset fully exported.')
