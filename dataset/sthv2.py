from functools import partial, wraps
from .data_utils import cast_num_frames, list_to_tensor, identity
import torch.nn.functional as F
from torchvision import transforms as T
import os.path as osp
import json
from torch.utils.data import Dataset
from torch.utils import data
from pathlib import Path
class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        val_batch_size,
        channels = 3,
        num_frames = 80,
        split = 'train',
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['jpg'],
        normalize = True,
    ):
        super().__init__()
        self.folder = folder
        self.annotations_dir = osp.join(folder,'annotations')
        self.raw_frames_dir = osp.join(folder,'rawframes')
        self.image_size = image_size
        self.channels = channels
        self.exts = exts
        self.split = split
        self.val_batch_size = val_batch_size
        if split == 'train':
            f = open(osp.join(self.annotations_dir,'train.json'))
        elif split == 'val':
            f = open(osp.join(self.annotations_dir,'validation.json'))
        elif split == 'test':
            f = open(osp.join(self.annotations_dir,'test.json'))
        self.text_dict = json.load(f)
        self.num_frames = num_frames
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity
        if normalize:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
                T.CenterCrop(image_size),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.text_dict)

    def __getitem__(self, index):
        label_id = self.text_dict[index]['id']
        text_prompts = self.text_dict[index]['label']
        path = [p for ext in self.exts for p in Path(f'{osp.join(self.raw_frames_dir,label_id)}').glob(f'**/*.{ext}')]
        tensor = list_to_tensor(path, self.channels, transform = self.transform)
        tensor = 2.*tensor - 1.
        return self.cast_num_frames_fn(tensor), text_prompts