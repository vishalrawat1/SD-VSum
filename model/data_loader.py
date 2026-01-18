# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import os


class VideoData(Dataset):
    def __init__(self, mode, dataset='S_NewsVSum', split_num=0):
        """
        Dataset loader for SD-VSum (Kaggle compatible)
        mode: train | val | test
        dataset: S_VideoXum | S_NewsVSum
        """

        self.mode = mode.lower()

        # -------- Dataset paths --------
        if dataset == 'S_VideoXum':
            self.filename = './dataset/script_videoxum.h5'
            self.dataset_split = './dataset/script_videoxum_split.json'

        elif dataset == 'S_NewsVSum':
            self.filename = './dataset/S_NewsVSum.h5'
            self.dataset_split = './dataset/S_NewsVSum_split.json'

        else:
            raise ValueError("Dataset must be S_VideoXum or S_NewsVSum")

        # -------- Load split file --------
        with open(self.dataset_split, 'r') as f:
            splits = json.load(f)

        # S_NewsVSum has multiple splits
        if dataset == 'S_NewsVSum':
            splits = splits[split_num]

        self.split_keys = splits[self.mode]

        # -------- Load HDF5 --------
        hdf = h5py.File(self.filename, 'r')

        self.video_features = []
        self.text_features = []
        self.gt_scores = []
        self.video_names = []

        for video_name in hdf.keys():
            if video_name not in self.split_keys:
                continue

            grp = hdf[video_name]

            # ---- embeddings ----
            video_emb = torch.tensor(grp['video_embeddings'][()], dtype=torch.float32)
            text_emb = torch.tensor(grp['text_embeddings'][()], dtype=torch.float32)

            # ---- ground truth (robust) ----
            if 'gtscores' in grp:
                gt = grp['gtscores'][()]
            elif 'scores' in grp:
                gt = grp['scores'][()]
            elif 'labels' in grp:
                gt = grp['labels'][()]
            else:
                raise KeyError(f"No GT scores found for video: {video_name}")

            gt = torch.tensor(gt, dtype=torch.float32)

            self.video_features.append(video_emb)
            self.text_features.append(text_emb)
            self.gt_scores.append(gt)
            self.video_names.append(video_name)

        hdf.close()

        print(f"[INFO] Loaded {len(self.video_names)} samples for {mode}")

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        return (
            self.video_features[idx],
            self.text_features[idx],
            self.gt_scores[idx]
        )


def get_loader(mode, dataset='S_NewsVSum', split_num=0):
    """
    Returns PyTorch DataLoader
    """
    if mode.lower() == 'train':
        dataset_obj = VideoData(mode, dataset=dataset, split_num=split_num)
        return DataLoader(dataset_obj, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, dataset=dataset, split_num=split_num)


if __name__ == "__main__":
    pass
