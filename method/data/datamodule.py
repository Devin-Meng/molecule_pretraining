import pytorch_lightning as pl
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

from method.data.dataset import MolDataset
from method.data.collator import collator

class ZincDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_path,
        train_prop=0.8,
        valid_prop=0.1,
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        max_node=128,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.input_path = input_path
        self.split_prop = {'train': train_prop,
                            'valid': valid_prop}
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def setup(self, stage, str = None):
        self.data = np.load(self.input_path, 
                            allow_pickle=True)
        length = len(self.data)
        train_split = int(length * self.split_prop['train'])
        valid_split = int(length * (self.split_prop['train'] +
                                    self.split_prop['valid']))
        self.dataset_train = MolDataset(self.data[:train_split])
        self.dataset_val = MolDataset(self.data[train_split:valid_split])
        self.dataset_test = MolDataset(self.data[valid_split:])

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator,
                               max_node=self.max_node,
                               multi_hop_max_dist=self.multi_hop_max_dist,
                               spatial_pos_max=self.spatial_pos_max)
        )
        #print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator,
                               max_node=self.max_node,
                               multi_hop_max_dist=self.multi_hop_max_dist,
                               spatial_pos_max=self.spatial_pos_max)
        )
        #print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator,
                               max_node=self.max_node,
                               multi_hop_max_dist=self.multi_hop_max_dist,
                               spatial_pos_max=self.spatial_pos_max)
        )
        #print('len(test_dataloader)', len(loader))
        return loader
