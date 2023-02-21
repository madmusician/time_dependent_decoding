from typing import Union

import numpy as np
from os import listdir

import torch
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from pathlib import Path


def cointoss(p):
    return random.random() < p


class DBreader_Vimeo90k(Dataset):
    def __init__(self, db_dir, tri_or_sep, frame_seqids: list[Union[int, str]], split='', random_crop=None, resize=None, augment_s=True, augment_t=True):
        db_dir += '/sequences'
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t
        if split == 'test':
            for pred in [random_crop, resize, augment_t, augment_s]:
                assert pred is None or pred is False  # Ensure no augmentation is used during testing

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        assert tri_or_sep in ['tri', 'sep']

        if split == 'train':  # only data in train.txt
            with open(db_dir + f'/../{tri_or_sep}_trainlist.txt', 'r') as sep_list:
                self.triplet_list = [db_dir + '/' + line.strip() for line in sep_list]
        elif split == 'test':  # only data in train.txt
            with open(db_dir + f'/../{tri_or_sep}_testlist.txt', 'r') as sep_list:
                self.triplet_list = [db_dir + '/' + line.strip() for line in sep_list]
        else:  # use whole vimeo90k train + test
            self.folder_list = [(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))]
            self.triplet_list = []
            for folder in self.folder_list:
                self.triplet_list += [(folder + '/' + f) for f in listdir(folder) if isdir(join(folder, f))]

        self.triplet_list = np.array(self.triplet_list)
        self.file_len = len(self.triplet_list)

        self.frame_seqids = frame_seqids

    def __getitem__(self, index):
        from typing import List, Callable
        # rawFrame0 = Image.open(self.triplet_list[index] + "/im1.png")
        # rawFrame1 = Image.open(self.triplet_list[index] + "/im2.png")
        # rawFrame2 = Image.open(self.triplet_list[index] + "/im3.png")
        rawFrames = [Image.open("%s/im%d.png" % (self.triplet_list[index], i))
                     for i in self.frame_seqids]

        trans = []  # type: List[Callable[[Image], Image]]

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrames[0], output_size=self.random_crop)
            # rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            # rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            # rawFrame2 = TF.crop(rawFrame2, i, j, h, w)
            trans.append(lambda frame: TF.crop(frame, i, j, h, w))

        if self.augment_s:
            if cointoss(0.5):
                # rawFrame0 = TF.hflip(rawFrame0)
                # rawFrame1 = TF.hflip(rawFrame1)
                # rawFrame2 = TF.hflip(rawFrame2)
                trans.append(lambda frame: TF.hflip(frame))
            if cointoss(0.5):
                # rawFrame0 = TF.vflip(rawFrame0)
                # rawFrame1 = TF.vflip(rawFrame1)
                # rawFrame2 = TF.vflip(rawFrame2)
                trans.append(lambda frame: TF.vflip(frame))

        # frame0 = self.transform(rawFrame0)
        # frame1 = self.transform(rawFrame1)
        # frame2 = self.transform(rawFrame2)
        trans.extend(self.transform.transforms)

        if self.augment_t:
            if cointoss(0.5):
                # return frame2, frame1, frame0
                rawFrames.reverse()
            else:
                # return frame0, frame1, frame2
                pass
        else:
            # return frame0, frame1, frame2
            pass
        return tuple(transforms.Compose(trans)(frame) for frame in rawFrames)

    def __len__(self):
        return self.file_len


class DBreader_UCF101QVI(Dataset):
    def __init__(self, db_dir, num_frames, random_crop=None, resize=None, augment_s=True, augment_t=True):
        assert num_frames == 4
        db_dir = Path(db_dir)
        assert db_dir.name == 'ucf101_extracted'

        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        self.triplet_list = [folder for folder in db_dir.iterdir() if folder.is_dir()]

        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        from typing import List, Callable
        # rawFrame0 = Image.open(self.triplet_list[index] + "/im1.png")
        # rawFrame1 = Image.open(self.triplet_list[index] + "/im2.png")
        # rawFrame2 = Image.open(self.triplet_list[index] + "/im3.png")
        rawFrames = [Image.open(self.triplet_list[index] / ("im%s.png" % (i,))
                                for i in [0, 1, 't', 2, 3])]

        trans = []  # type: List[Callable[[Image], Image]]

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrames[0], output_size=self.random_crop)
            # rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            # rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            # rawFrame2 = TF.crop(rawFrame2, i, j, h, w)
            trans.append(lambda frame: TF.crop(frame, i, j, h, w))

        if self.augment_s:
            if cointoss(0.5):
                # rawFrame0 = TF.hflip(rawFrame0)
                # rawFrame1 = TF.hflip(rawFrame1)
                # rawFrame2 = TF.hflip(rawFrame2)
                trans.append(lambda frame: TF.hflip(frame))
            if cointoss(0.5):
                # rawFrame0 = TF.vflip(rawFrame0)
                # rawFrame1 = TF.vflip(rawFrame1)
                # rawFrame2 = TF.vflip(rawFrame2)
                trans.append(lambda frame: TF.vflip(frame))

        # frame0 = self.transform(rawFrame0)
        # frame1 = self.transform(rawFrame1)
        # frame2 = self.transform(rawFrame2)
        trans.extend(self.transform.transforms)

        if self.augment_t:
            if cointoss(0.5):
                # return frame2, frame1, frame0
                rawFrames.reverse()
            else:
                # return frame0, frame1, frame2
                pass
        else:
            # return frame0, frame1, frame2
            pass
        return tuple(transforms.Compose(trans)(frame) for frame in rawFrames)

    def __len__(self):
        return self.file_len


class Vimeo90kSmall(DBreader_Vimeo90k):
    """
    A small vimeo90k dataset, for quickly finish training iteration to test evaluation functionality.
    """
    def __init__(self, db_dir, num_frames, *args, **kwargs):
        super(Vimeo90kSmall, self).__init__(db_dir, num_frames, *args, **kwargs)
        self.triplet_list = self.triplet_list[:16]
        self.file_len = len(self.triplet_list)


# GoPro reader from FLAVR
class GoPro(Dataset):
    def __init__(self, data_root, mode="train", interFrames=3, n_inputs=4, use_augment=False):
        super().__init__()

        import os

        self.interFrames = interFrames
        self.n_inputs = n_inputs
        self.setLength = (n_inputs - 1) * (
                    interFrames + 1) + 1  ## We require these many frames in total for interpolating `interFrames` number of
        ## intermediate frames with `n_input` input frames.
        self.data_root = os.path.join(data_root, mode)

        video_list = os.listdir(self.data_root)
        self.frames_list = []

        self.file_list = []
        for video in video_list:
            frames = sorted(os.listdir(os.path.join(self.data_root, video)))
            n_sets = (len(frames) - self.setLength) // (interFrames + 1) + 1
            videoInputs = [frames[(interFrames + 1) * i:(interFrames + 1) * i + self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(video, f) for f in group] for group in videoInputs]
            self.file_list.extend(videoInputs)

        assert not (mode == 'test' and use_augment is True)  # forbid augmentation during test
        self.use_augment = use_augment

        self.simple_transforms = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        import os
        imgpaths = [os.path.join(self.data_root, fp) for fp in self.file_list[idx]]

        pick_idxs = list(range(0, self.setLength, self.interFrames + 1))
        rem = self.interFrames % 2
        gt_idx = list(
            range(self.setLength // 2 - self.interFrames // 2, self.setLength // 2 + self.interFrames // 2 + rem))

        input_paths = [imgpaths[idx] for idx in pick_idxs]
        gt_paths = [imgpaths[idx] for idx in gt_idx]

        images = [Image.open(pth_) for pth_ in input_paths]
        # images = [self.simple_transforms(img_) for img_ in images]

        gt_images = [Image.open(pth_) for pth_ in gt_paths]
        # gt_images = [self.simple_transforms(img_) for img_ in gt_images]

        images, gt_images = self.transform(images, gt_images)
        return images, gt_images

    def __len__(self):
        return len(self.file_list)

    def transform(self, images: list[Image], gt_images: list[Image]) -> (list[torch.Tensor], list[torch.Tensor]):
        if self.use_augment:
            trans = []

            # random cropping
            i, j, h, w = transforms.RandomCrop.get_params(images[0], output_size=(512, 512))
            trans.append(lambda frame: TF.crop(frame, i, j, h, w))

            # random flipping
            if cointoss(0.5):
                trans.append(lambda frame: TF.hflip(frame))
            if cointoss(0.5):
                trans.append(lambda frame: TF.vflip(frame))

            trans.append(transforms.ToTensor())
            trans = transforms.Compose(trans)

            # random temporal reversal
            if cointoss(0.5):
                images.reverse()
                gt_images.reverse()

            return [trans(img_) for img_ in images], \
                   [trans(img_) for img_ in gt_images]
        else:
            return [self.simple_transforms(img_) for img_ in images], \
                   [self.simple_transforms(img_) for img_ in gt_images]


preset_dataset_factory = {
    'vimeo_triplet': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_triplet', 'tri', [1, 2, 3], *args, **kwargs),
    'vimeo_triplet_train': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_triplet', 'tri', [1, 2, 3], split='train', *args, **kwargs),
    'vimeo_triplet_test': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_triplet', 'tri', [1, 2, 3], split='test', augment_s=False, augment_t=False, *args, **kwargs),
    'vimeo_septuplet': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_septuplet', 'sep', [1, 3, 4, 5, 7], *args, **kwargs),
    'vimeo_septuplet_train': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_septuplet', 'sep', [1, 3, 4, 5, 7], split='train', *args, **kwargs),
    'vimeo_septuplet_test': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_septuplet', 'sep', [1, 3, 4, 5, 7], split='test', augment_s=False, augment_t=False, *args, **kwargs),
    'vimeo_septuplet_train_4as2': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_septuplet', 'sep', [3, 4, 5], split='train', *args, **kwargs),
    'vimeo_septuplet_test_4as2': lambda *args, **kwargs: DBreader_Vimeo90k('./db/vimeo_septuplet', 'sep', [3, 4, 5], split='test', augment_s=False, augment_t=False, *args, **kwargs),
    'vimeo_septuplet_small': lambda *args, **kwargs: Vimeo90kSmall('./db/vimeo_septuplet', 'sep', [1, 3, 4, 5, 7], *args, **kwargs),
    'gopro_flavr_train': lambda *args, **kwargs: GoPro('./db/gopro', 'train', interFrames=7, n_inputs=4, *args, **kwargs),
    'gopro_flavr_train_4to1': lambda *args, **kwargs: GoPro('./db/gopro', 'train', interFrames=7, n_inputs=4, *args, **kwargs),
    'gopro_flavr_train_2to1': lambda *args, **kwargs: GoPro('./db/gopro', 'train', interFrames=7, n_inputs=2, *args, **kwargs),
    'gopro_flavr_train_group7_4to1': lambda *args, **kwargs: GoPro('./db/gopro', 'train', interFrames=1, n_inputs=4, *args, **kwargs),
    'gopro_flavr_train_group7_2to1': lambda *args, **kwargs: GoPro('./db/gopro', 'train', interFrames=1, n_inputs=2, *args, **kwargs),
    'gopro_flavr_test': lambda *args, **kwargs: GoPro('./db/gopro', 'test', interFrames=7, n_inputs=4, *args, **kwargs),
    'gopro_flavr_test_group7': lambda *args, **kwargs: GoPro('./db/gopro', 'test', interFrames=1, n_inputs=4, *args, **kwargs),
}
