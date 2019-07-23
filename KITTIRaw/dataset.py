import os
import torch
import numpy as np
from PIL import Image
from Dataloader import Kittiloader
from Transformer import Transformer
from torch.utils.data import Dataset, DataLoader


class KittiDataset(Dataset):
    def __init__(self,
                 kittiDir,
                 mode,
                 transform=None):
        self.mode = mode
        self.kitti_root = kittiDir
        self.transform = transform

        # use left image by default
        self.kittiloader = Kittiloader(kittiDir, mode, cam=2)

    def __getitem__(self, idx):
        # load an item according to the given index
        data_item = self.kittiloader.load_item(idx)

        data_transed = self.transform(data_item)

        return data_transed

    def __len__(self):
        return self.kittiloader.data_length()


class DataGenerator(object):
    def __init__(self,
                 KittiDir,
                 phase,
                 high_gpu=True):
        self.phase = phase
        self.high_gpu = high_gpu

        if not self.phase in ['train', 'test', 'val']:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass

        transformer = Transformer(self.phase)
        self.dataset = KittiDataset(KittiDir,
                                    phase,
                                    transformer.get_transform())

    def create_data(self, batch_size, nthreads=0):
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=(self.phase=='train'),
                          num_workers=nthreads,
                          pin_memory=self.high_gpu)
