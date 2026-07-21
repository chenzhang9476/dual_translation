import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from .SynthRAD_HN import SynthRAD_HN_Dataset

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dataset = SynthRAD_HN_Dataset(
            root=opt.dataroot,
            index_json="slice_index_HN.json",
            target_size=(opt.crop_size, opt.crop_size),
            min_fg_ratio=0.10
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        return {
            'A': item['image'],
            'B': item['ct'],
            'A_paths': f"{item['patient_id']}_{item['slice_idx']}",
            'B_paths': f"{item['patient_id']}_{item['slice_idx']}_ct"
        }

    def __len__(self):
        return len(self.dataset)