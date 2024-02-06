from functools import partial, wraps
from .data_utils import cast_num_frames, list_to_tensor, identity
import torch.nn.functional as F
from torchvision import transforms as T
import os.path as osp
import csv
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
        self.annotations_dir = './epic-kitchens-100-annotations'
        self.raw_frames_dir = osp.join(folder,'EPIC-KITCHENS')
        self.image_size = image_size
        self.channels = channels
        self.exts = exts
        self.split = split
        self.val_batch_size = val_batch_size
        if split == 'train':
            f = osp.join(self.annotations_dir,'EPIC_100_train.csv')
        elif split == 'val':
            f = osp.join(self.annotations_dir,'EPIC_100_validation.csv')
        self.text_dict = []
        with open(f, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in  :
                text_prompt,start_frame,end_frame = row["narration"].strip(),int(row["start_frame"].strip()),int(row["stop_frame"].strip())
                self.text_dict.append({"dir_id":row["participant_id"],"video_id":row["video_id"],"text_prompt":text_prompt,"start":start_frame,"end":end_frame})
        
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
        label_id = self.text_dict[index]['dir_id']
        video_id = self.text_dict[index]['video_id']
        start_frame = self.text_dict[index]['start']
        end_frame = self.text_dict[index]['end']
        text_prompt = self.text_dict[index]['text_prompt']
        root_dir = osp.join(self.raw_frames_dir,label_id+'/rgb_frames/'+video_id)
        if (end_frame-start_frame)>=100:
            path = [osp.join(root_dir,'frame_'+str(p).zfill(10)+'.'+self.exts[0]) for p in range(start_frame,end_frame+1,(end_frame-start_frame)//100)]
        else:
            path = [osp.join(root_dir,'frame_'+str(p).zfill(10)+'.'+self.exts[0]) for p in range(start_frame,end_frame+1)]
        tensor = list_to_tensor(path, self.channels, transform = self.transform)
        tensor = 2.*tensor - 1.
        return self.cast_num_frames_fn(tensor), text_prompt