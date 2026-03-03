import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(IuxrayMultiImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = self._load_data(args.label_path)

    def _load_data(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].to_list()

            label_dict[idx] = list(map(lambda x: 1 if x == 1.0 else 0, label))

        return label_dict

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

        pid = image_id.split('_')[0][3:]
        try:
            labels = torch.tensor(self.label[int(pid)], dtype=torch.float32)
        except:
            # print('Except id ', pid)
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(MimiccxrSingleImageDataset, self).__init__(args, tokenizer, split, transform)
        self.label = pd.read_csv(args.label_path)


    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB') #MIMICCXR数据集需要这行代码
        # image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        pid = image_id.split('_')[0][3:]
        try:
            labels = torch.tensor(self.label[int(pid)], dtype=torch.float32)
        except:
            # print('Except id ', pid)
            labels = torch.tensor([0 for _ in range(14)], dtype=torch.float32)

        # d = self.label[self.label['dicom_id'] == image_id]
        # labels = torch.tensor(d.values.tolist()[0][8:], dtype=torch.float32)
        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample



class LGKNoLabelDataset(BaseDataset):
    def __init__(self, args, tokenizer, split, transform=None):
        super(LGKNoLabelDataset, self).__init__(args, tokenizer, split, transform)
        # 没有label文件，不做label加载

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path'][0]  # 只读一张图
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # 没有label，用全0填充，占位用（假设14类）
        labels = torch.zeros(14, dtype=torch.float32)

        sample = (image_id, image, report_ids, report_masks, seq_length, labels)
        return sample
