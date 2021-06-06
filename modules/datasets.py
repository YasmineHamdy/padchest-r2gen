import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchtext.legacy import data
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.dataset_name = args.dataset_name

        if self.dataset_name == 'padchest':
            ImageID = data.Field()
            Report = data.LabelField()

            fields = [(None, None), ('image_path', ImageID), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),('report', Report)]
            whole_data = data.TabularDataset(
                path = self.ann_path,
                format = 'csv',
                fields = fields,
                skip_header = True
            )
            dataset = whole_data[:3000]
            splitedData = np.split(dataset, [2000, 2400, 3000])

            self.ann = {"train": splitedData[0], "val": splitedData[1], "test": splitedData[2]  }
            temp_examples = []
        else:
            self.ann = json.loads(open(self.ann_path, 'r').read())
            self.examples = self.ann[self.split]
            

        split = self.ann[self.split]

        for i in range(len(split)):
            if self.dataset_name == 'padchest':
                ids = tokenizer(vars(split[i])['report'])[:self.max_seq_length]
                mask = [1] * len(ids)
                temp_examples.append(dict(((k, vars(split[i])[k]) for k in vars(split[i])), ids=ids, mask = mask))
                
                self.examples = temp_examples
            else:
                self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
                self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

class PadChestSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_path = example['image_path'][0]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
       
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_path, image, report_ids, report_masks, seq_length)
        return sample
