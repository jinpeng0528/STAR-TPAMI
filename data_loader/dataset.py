import os
import pathlib
import json
import random
# import tqdm
import torch
import numpy as np
import torchvision as tv

from PIL import Image
from torch import distributed
from data_loader import DATASETS_IMG_DIRS
from data_loader import custom_transforms as tr
from base.base_dataset import BaseDataset, lbl_contains_any, lbl_contains_all


CITY_MAPPING = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 0,
    10: 0,
    11: 3,
    12: 4,
    13: 5,
    14: 0,
    15: 0,
    16: 0,
    17: 6,
    18: 0,
    19: 7,
    20: 8,
    21: 9,
    22: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
    29: 0,
    30: 0,
    31: 17,
    32: 18,
    33: 19,
    -1: 0
}


class VOCSegmentationIncremental(BaseDataset):
    """
    PascalVoc dataset
    """
    def __init__(
        self,
        test=False, val=False, setting='overlap', classes_idx_new=[], classes_idx_old=[],
        transform=True, transform_args={}, masking_value=0, idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['voc']),
            transform=transform,
        )
        self.setting = setting
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)

        self.masking_value = masking_value

        if self.train:
            self.split = 'train_aug'
        else:
            self.split = 'val'

        if 'aug' not in self.split:
            self._image_dir = self._base_dir / "JPEGImages"
            self._cat_dir = self._base_dir / "SegmentationClass"
        else:
            self._image_dir = self._base_dir
            self._cat_dir = self._base_dir
        _splits_dir = self._base_dir / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.categories = []

        if (idxs_path is not None) and (os.path.exists(idxs_path)):
            self.im_ids = np.load(idxs_path).tolist()
            for x in self.im_ids:
                if 'aug' not in self.split:
                    _image = self._image_dir / f"{x}.jpg"
                    _cat = self._cat_dir / f"{x}.png"
                else:
                    _image = self._image_dir / x.split()[0][1:]
                    _cat = self._cat_dir / x.split()[1][1:]

                assert _image.is_file(), _image
                assert _image.is_file(), _cat
                
                self.images.append(_image)
                self.categories.append(_cat)
        else:
            if distributed.get_rank() == 0:
                print("Filtering images....")

            lines = (_splits_dir / f"{self.split}.txt").read_text().splitlines()
            for ii, line in enumerate(lines):
                if (ii % 1000 == 0) and (distributed.get_rank() == 0):
                    print(f"[{ii} / {len(lines)}]")
                    
                if 'aug' not in self.split:
                    _image = self._image_dir / f"{line}.jpg"
                    _cat = self._cat_dir / f"{line}.png"
                else:
                    _image = self._image_dir / line.split()[0][1:]
                    _cat = self._cat_dir / line.split()[1][1:]
                assert _image.is_file(), _image
                assert _cat.is_file(), _cat

                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)

                if (self.train or self.val):
                    # Remove the sample if the g.t mask does not contain new class
                    if not lbl_contains_any(cat, self.classes_idx_new):
                        continue
                    # Unique set
                    # : Remove the sample if the g.t mask contains any other labels that not appeared yet.
                    if (self.train) and (self.setting == 'disjoint' or self.setting == 'sequential'):
                        if not lbl_contains_all(cat, list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))):
                            continue
                else:  # Test
                    if not lbl_contains_any(cat, list(set(self.classes_idx_old + self.classes_idx_new))):
                        continue

                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

            if (idxs_path is not None) and (distributed.get_rank() == 0):
                np.save(idxs_path, np.array(self.im_ids))

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["trainval_aug", "trainval", "train_aug", "train"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["val_aug", "val"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample['label'] = self.transform_target_masking(sample['label'])
        # sample["image_name"] = str(self.images[index])
        sample["image_name"] = str(self.im_ids[index])
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # composed_transforms = tr.Compose(
        #     [
        #         tr.Resize(size=self.transform_args['crop_size']),
        #         tr.ToTensor(),
        #         tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)

    def __str__(self):
        return f"VOC2012(split={self.split})"

    def __len__(self):
        return len(self.images)


class ADESegmentationIncremental(BaseDataset):
    """
    ADE20k dataset
    """
    def __init__(
        self,
        test=False,
        val=False,
        setting='overlap',
        classes_idx_new=[],
        classes_idx_old=[],
        transform=True,
        transform_args={},
        masking_value=0,
        idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['ade']),
            transform=transform,
        )
        self.setting = setting
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)

        self.masking_value = masking_value

        if self.train:
            self.split = 'training'
        else:
            self.split = 'validation'

        self._image_dir = self._base_dir / "images" / self.split
        self._cat_dir = self._base_dir / "annotations" / self.split
        
        fnames = sorted(os.listdir(self._image_dir))

        self.im_ids = []
        self.categories = []

        if idxs_path is not None and os.path.exists(idxs_path):
            self.im_ids = np.load(idxs_path).tolist()
            for x in self.im_ids:
                _image = self._image_dir / f"{x}.jpg"  # .jpg
                _cat = self._cat_dir / f"{x}.png"  # .png
                assert _image.is_file(), _image
                assert _image.is_file(), _cat
                
                self.images.append(_image)
                self.categories.append(_cat)
        else:
            if distributed.get_rank() == 0:
                print("Filtering images....")

            for ii, x in enumerate(fnames):
                _image = self._image_dir / x  # .jpg
                _cat = self._cat_dir / f"{x[:-4]}.png"  # .png
                assert _image.is_file(), _image
                assert _image.is_file(), _cat

                cat = np.array(Image.open(_cat), dtype=np.uint8)
                if (self.train or self.val):
                    # Remove the sample if the g.t mask does not contain new class
                    if not lbl_contains_any(cat, self.classes_idx_new):
                        continue
                    # Unique set
                    # : Remove the sample if the g.t mask contains any other labels that not appeared yet.
                    if (self.train) and (self.setting == 'disjoint' or self.setting == 'sequential'):
                        if not lbl_contains_all(cat, list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))):
                            continue
                else:  # Test
                    if not lbl_contains_any(cat, list(set(self.classes_idx_old + self.classes_idx_new))):
                        continue
                
                if (ii % 1000 == 0) and (distributed.get_rank() == 0):
                    print(f"[{ii} / {len(fnames)}]")

                self.im_ids.append(x[:-4])
                self.images.append(_image)
                self.categories.append(_cat)

            if (idxs_path is not None) and (distributed.get_rank() == 0):
                np.save(idxs_path, np.array(self.im_ids))

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["training"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["validation"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample['label'] = self.transform_target_masking(sample['label'])
        # sample["image_name"] = str(self.images[index])
        sample["image_name"] = str(self.im_ids[index])
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)

    def __str__(self):
        return f"ADE20k(split={self.split})"

    def __len__(self):
        return len(self.images)


class CitySegmentationIncremental(BaseDataset):
    """
    PascalVoc dataset
    """

    def __init__(
            self,
            test=False, val=False, setting='overlap', classes_idx_new=[], classes_idx_old=[],
            transform=True, transform_args={}, masking_value=0, idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['city']),
            transform=transform,
        )
        self.setting = setting
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)

        self.masking_value = masking_value

        if self.train:
            self.split = 'fine_train.txt'
        else:
            self.split = 'fine_val.txt'

        self._image_dir = self._base_dir
        self._cat_dir = self._base_dir

        # load image list
        image_and_cat_list_path = self._base_dir / self.split
        with open(image_and_cat_list_path, "r") as f:
            fnames = f.read().splitlines()
        # fnames = sorted(os.listdir(self._image_dir))
        fnames = [tuple(x.split()) for x in fnames]

        self.im_ids = []
        self.categories = []

        if idxs_path is not None and os.path.exists(idxs_path):
            self.im_ids = np.load(idxs_path).tolist()
            for x in self.im_ids:
                _image = self._image_dir / x[0]  # .jpg
                _cat = self._cat_dir / x[1]  # .png
                assert _image.is_file(), _image
                assert _image.is_file(), _cat

                self.images.append(_image)
                self.categories.append(_cat)
        else:
            if distributed.get_rank() == 0:
                print("Filtering images....")

            for ii, x in enumerate(fnames):
                _image = self._image_dir / x[0]  # .jpg
                _cat = self._cat_dir / x[1] # .png
                assert _image.is_file(), _image
                assert _cat.is_file(), _cat

                cat = np.array(Image.open(_cat), dtype=np.uint8)
                # map to 20 classes
                cat = np.vectorize(CITY_MAPPING.get)(cat)
                if (self.train or self.val):
                    # Remove the sample if the g.t mask does not contain new class
                    if not lbl_contains_any(cat, self.classes_idx_new):
                        continue
                    # Unique set
                    # : Remove the sample if the g.t mask contains any other labels that not appeared yet.
                    if (self.train) and (self.setting == 'disjoint' or self.setting == 'sequential'):
                        if not lbl_contains_all(cat, list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))):
                            continue
                else:  # Test
                    if not lbl_contains_any(cat, list(set(self.classes_idx_old + self.classes_idx_new))):
                        continue

                if (ii % 100 == 0) and (distributed.get_rank() == 0):
                    print(f"[{ii} / {len(fnames)}]")

                self.im_ids.append((x[0], x[1]))
                self.images.append(_image)
                self.categories.append(_cat)

            if (idxs_path is not None) and (distributed.get_rank() == 0):
                np.save(idxs_path, np.array(self.im_ids))

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}
        # _target = np.vectorize(CITY_MAPPING.get)(_target)

        if self.transform:
            if self.split in ["fine_train.txt"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["fine_val.txt"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample['label'] = self.transform_target_masking(sample['label'])
        # sample["image_name"] = str(self.images[index])
        sample["image_name"] = str(self.im_ids[index])
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        # # map to 20 classes then to Image
        # _target = np.vectorize(CITY_MAPPING.get)(_target)
        # _target = Image.fromarray(_target)
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                # tr.RandomResizedCrop(
                #     (self.transform_args['crop_size'], self.transform_args['crop_size'] * 2),
                #     (0.5, 2.0),
                #     (3. / 2., 8. / 3.)
                # ),
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                # tr.Resize(size=(self.transform_args['crop_size'], 2 * self.transform_args['crop_size'])),
                # tr.CenterCrop((self.transform_args['crop_size'], 2 * self.transform_args['crop_size'])),
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        # map to 20 classes by Torch
        target = target.apply_(lambda x: CITY_MAPPING.get(x, 0))
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)

    def __str__(self):
        return f"CityScapes(split={self.split})"

    def __len__(self):
        return len(self.images)


class VOCSegmentationIncrementalMemory(BaseDataset):
    def __init__(
        self,
        test=False, val=False, setting='overlap', step=0, classes_idx_new=[], classes_idx_old=[], transform=True,
        transform_args={}, masking_value=0, idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['voc']),
            transform=transform,
        )
        self.setting = setting
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)

        self.masking_value = masking_value

        if self.train:
            self.split = 'train_aug'
        else:
            self.split = 'val'

        if 'aug' not in self.split:
            self._image_dir = self._base_dir / "JPEGImages"
            self._cat_dir = self._base_dir / "SegmentationClass"
        else:
            self._image_dir = self._base_dir
            self._cat_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(idxs_path, "r") as json_file:
            memory_list = json.load(json_file)

        file_names = memory_list[f"step_{step}"]["memory_list"]
        for x in file_names:
            if 'aug' not in self.split:
                _image = self._image_dir / f"{x}.jpg"
                _cat = self._cat_dir / f"{x}.png"
            else:
                _image = self._image_dir / x.split()[0][1:]
                _cat = self._cat_dir / x.split()[1][1:]

            assert _image.is_file(), _image
            assert _image.is_file(), _cat
            
            self.im_ids.append(x)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["trainval_aug", "trainval", "train_aug", "train"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["val_aug", "val"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample['label'] = self.transform_target_masking(sample['label'])
        sample["image_name"] = str(self.im_ids[index])
        # sample["memory"] = True

        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)

    def __len__(self):
        return len(self.images)


class ADESegmentationIncrementalMemory(BaseDataset):
    def __init__(
        self,
        test=False, val=False, setting='overlap', step=0, classes_idx_new=[], classes_idx_old=[],
        transform=True, transform_args={}, masking_value=0, idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['ade']),
            transform=transform,
        )
        self.setting = setting
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)

        self.masking_value = masking_value

        if self.train:
            self.split = 'training'
        else:
            self.split = 'validation'

        self._image_dir = self._base_dir / "images" / self.split
        self._cat_dir = self._base_dir / "annotations" / self.split

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(idxs_path, "r") as json_file:
            memory_list = json.load(json_file)

        file_names = memory_list[f"step_{step}"]["memory_list"]
        for x in file_names:
            _image = self._image_dir / f"{x}.jpg"  # .jpg
            _cat = self._cat_dir / f"{x}.png"  # .png
            assert _image.is_file(), _image
            assert _image.is_file(), _cat
            
            self.im_ids.append(x)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["training"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["validation"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample['label'] = self.transform_target_masking(sample['label'])
        sample["image_name"] = str(self.im_ids[index])

        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            # SSUL uses fully annotated mask, while we don't
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)

    def __len__(self):
        return len(self.images)


class CitySegmentationIncrementalMemory(BaseDataset):
    def __init__(
        self,
        test=False, val=False, setting='overlap', step=0, classes_idx_new=[], classes_idx_old=[],
        transform=True, transform_args={}, masking_value=0, idxs_path=None,
    ):
        if setting not in ['sequential', 'disjoint', 'overlap']:
            raise ValueError('Wrong setting entered! Please use one of sequential, disjoint, overlap')

        super().__init__(
            transform_args=transform_args,
            base_dir=pathlib.Path(DATASETS_IMG_DIRS['city']),
            transform=transform,
        )
        self.setting = setting
        self.classes_idx_old = classes_idx_old
        self.classes_idx_new = classes_idx_new

        self.test = test
        self.val = val
        self.train = not (self.test or self.val)

        self.masking_value = masking_value

        if self.train:
            self.split = 'fine_train'
        else:
            self.split = 'fine_val'

        self._image_dir = self._base_dir / "images" / self.split
        self._cat_dir = self._base_dir / "annotations" / self.split

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(idxs_path, "r") as json_file:
            memory_list = json.load(json_file)

        file_names = memory_list[f"step_{step}"]["memory_list"]
        for x in file_names:
            _image = self._image_dir / f"{x}.jpg"  # .jpg
            _cat = self._cat_dir / f"{x}.png"  # .png
            assert _image.is_file(), _image
            assert _image.is_file(), _cat

            self.im_ids.append(x)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split in ["training"]:
                sample['image'], sample['label'] = self.transform_tr(sample)
            elif self.split in ["validation"]:
                sample['image'], sample['label'] = self.transform_val(sample)
        else:
            sample['image'], sample['label'] = self.transform_test(sample)

        # Target masking
        sample['label'] = self.transform_target_masking(sample['label'])
        sample["image_name"] = str(self.im_ids[index])

        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.RandomResizedCrop(
                    self.transform_args['crop_size'],
                    (0.5, 2.0)
                ),
                tr.RandomHorizontalFlip(),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_val(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.Resize(size=self.transform_args['crop_size']),
                tr.CenterCrop(self.transform_args['crop_size']),
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_test(self, sample):
        composed_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return composed_transforms(sample['image'], sample['label'])

    def transform_target_masking(self, target):
        if self.test:
            # Masking future class object
            # MiB: 255 / PLOP: 0
            label_list = list(set(self.classes_idx_old + self.classes_idx_new + [0, 255]))
            target_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
            )
            return target_transform(target)

        else:  # Train or Validation
            # Masking except current classes
            if self.masking_value is None:
                return target
            if self.setting in ['disjoint', 'overlap']:
                label_list = list(set(self.classes_idx_new + [0, 255]))
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: x if x in label_list else self.masking_value)
                )
                return target_transform(target)

    def __len__(self):
        return len(self.images)
