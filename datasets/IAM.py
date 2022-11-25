# import os
# import pickle
# import cv2
# import numpy as np
# import torch
# from utils import generate_distort, generate_perspective, generate_stretch
# from .general_hw import apply_data_augmentation
# from random import shuffle 

# iam_augmentation = {
#     "perspective": {
#         "proba": 0.2,
#         "min_factor": 0,
#         "max_factor": 0.3,
#     },
#     "elastic_distortion": {
#         "proba": 0.2,
#         "max_magnitude": 20,
#         "max_kernel": 3,
#     },
#     "random_transform": {
#         "proba": 0.2,
#         "max_val": 16,
#     },
#     "dilation_erosion": {
#         "proba": 0.2,
#         "min_kernel": 1,
#         "max_kernel": 3,
#         "iterations": 1,
#     },
#     "brightness": {
#         "proba": 0.2,
#         "min_factor": 0.01,
#         "max_factor": 1,
#     },
#     "contrast": {
#         "proba": 0.2,
#         "min_factor": 0.01,
#         "max_factor": 1,
#     },
#     "sign_flipping": {
#         "proba": 0.2,
#     },
# }

# class NewIAMDataset(torch.utils.data.Dataset):

#     def __init__(self, label_map, root_dir, mode='train', input_shape=[1, 64, 1200], p_aug=-1, text_img_distort=False):
#         super(NewIAMDataset, self).__init__()
#         c, conH, conW = input_shape 
#         self.conH = conH 
#         self.conW = conW 
#         self.p_aug = p_aug 
#         self.label_map = label_map 
#         assert mode in ['train', 'valid', 'test']
#         if p_aug > 0: assert mode == 'train'
#         self.text_img_distort = text_img_distort  # not for searching, nan 

#         self.img_paths = []
#         self.labels    = []
#         samples = pickle.load(open(os.path.join(root_dir, 'labels.pkl'), 'rb'))['ground_truth'][mode]
#         for image_path, label in samples.items():
#             image_path = os.path.join(root_dir, 'enh_%s' % mode, image_path)
#             assert os.path.isfile(image_path), 'check %s failed.' % image_path
#             self.img_paths.append(image_path)
#             self.labels.append(label['text'])

#     def __len__(self):
#         return len(self.img_paths)

#     def aug_norm_img(self, img):
#         if img is None: raise ValueError

#         h, w = img.shape
#         _conH = self.conH - (4 if self.p_aug > 0 else 0)
#         _conW = self.conW - (4 if self.p_aug > 0 else 0)
#         imgN = np.ones((_conH, _conW)) * 255 
#         beginH = int(abs(_conH - h) / 2)
#         beginW = int(abs(_conW - w) / 2)
#         if h <= _conH and w <= _conW:
#             imgN[beginH : beginH + h, beginW : beginW + w] = img 
#         elif (h / w) > (_conH / _conW):
#             newW = int(w * _conH / h)
#             beginW = int(abs(_conW - newW) / 2)
#             img = cv2.resize(img, (newW, _conH), interpolation=cv2.INTER_CUBIC)
#             imgN[:, beginW : beginW + newW] = img 
#         elif (h / w) <= (_conH / _conW):
#             newH = int(h * _conW / w)
#             beginH = int(abs(_conH - newH) / 2)
#             img = cv2.resize(img, (_conW, newH), interpolation=cv2.INTER_CUBIC)
#             imgN[beginH : beginH + newH, :] = img

#         def add_distort(img):
#             if np.random.rand() < self.p_aug:
#                 img = np.pad(img, ((5, 5), (5, 5)), mode='constant', constant_values=255)
#                 img = generate_distort(img, np.random.randint(8, 20))
#                 img = cv2.resize(img, (self.conW, self.conH), interpolation=cv2.INTER_CUBIC)
#             return img 

#         def add_stretch(img):
#             if np.random.rand() < self.p_aug:
#                 img = generate_stretch(img, np.random.randint(2, 20))
#             return img 

#         def add_perspective(img):
#             if np.random.rand() < self.p_aug:
#                 img = generate_perspective(img)
#             return img 

#         if self.p_aug > 0:
#             imgN = np.pad(imgN, ((2, 2), (2, 2)), mode='constant', constant_values=255).astype('uint8')

#             if self.text_img_distort:
#                 add_aug_fns = [add_distort, add_stretch, add_perspective]
#                 shuffle(add_aug_fns)
#                 for fn in add_aug_fns:
#                     imgN = fn(imgN)

#             if np.random.randn() < self.p_aug:
#                 imgN, _ = apply_data_augmentation(imgN, iam_augmentation)
#                 imgN = imgN[:, :, 0]

#         imgN = imgN.astype('float32') 
#         imgN = (imgN - 127.5) / 127.5 
#         return imgN 

#     def __getitem__(self, idx):
#         try:
#             img = cv2.imread(self.img_paths[idx], 0)
#             label = self.labels[idx]
#             img = self.aug_norm_img(img)
#         except:
#             print('corrupt')
#             return self[(idx + 1) % len(self.img_paths)]

#         if len(label) == 0:
#             return self[(idx + 1) % len(self.img_paths)]

#         img = img.reshape(1, self.conH, self.conW) 
#         label = self.label_map.encode(label) 
#         return torch.from_numpy(img), torch.IntTensor(label) 


import os
import pickle
from random import shuffle

import cv2
import numpy as np
import torch
from utils import generate_distort, generate_perspective, generate_stretch

from .general_hw import apply_data_augmentation

iam_augmentation = {
    "perspective": {
        "proba": 0.2,
        "min_factor": 0,
        "max_factor": 0.3,
    },
    "elastic_distortion": {
        "proba": 0.2,
        "max_magnitude": 20,
        "max_kernel": 3,
    },
    "random_transform": {
        "proba": 0.2,
        "max_val": 16,
    },
    "dilation_erosion": {
        "proba": 0.2,
        "min_kernel": 1,
        "max_kernel": 3,
        "iterations": 1,
    },
    "brightness": {
        "proba": 0.2,
        "min_factor": 0.01,
        "max_factor": 1,
    },
    "contrast": {
        "proba": 0.2,
        "min_factor": 0.01,
        "max_factor": 1,
    },
    "sign_flipping": {
        "proba": 0.2,
    },
}

class NewIAMDataset(torch.utils.data.Dataset):

    def __init__(self, label_map, root_dir, mode='train', input_shape=[1, 64, 1200], p_aug=-1):
        super(NewIAMDataset, self).__init__()
        c, conH, conW = input_shape 
        self.conH = conH 
        self.conW = conW 
        self.p_aug = p_aug 
        self.label_map = label_map 
        assert mode in ['train', 'valid', 'test']
        if p_aug > 0: assert mode == 'train'

        self.img_paths = []
        self.labels    = []
        samples = pickle.load(open(os.path.join(root_dir, 'labels.pkl'), 'rb'))['ground_truth'][mode]
        for image_path, label in samples.items():
            image_path = os.path.join(root_dir, 'enh_%s' % mode, image_path)
            assert os.path.isfile(image_path), 'check %s failed.' % image_path
            self.img_paths.append(image_path)
            self.labels.append(label['text'])

    def __len__(self):
        return len(self.img_paths)

    def aug_norm_img(self, img):
        if img is None: raise ValueError

        h, w = img.shape
        _conH = self.conH - (4 if self.p_aug > 0 else 0)
        _conW = self.conW - (4 if self.p_aug > 0 else 0)
        imgN = np.ones((_conH, _conW)) * 255 
        beginH = int(abs(_conH - h) / 2)
        beginW = int(abs(_conW - w) / 2)
        if h <= _conH and w <= _conW:
            imgN[beginH : beginH + h, beginW : beginW + w] = img 
        elif (h / w) > (_conH / _conW):
            newW = int(w * _conH / h)
            beginW = int(abs(_conW - newW) / 2)
            img = cv2.resize(img, (newW, _conH), interpolation=cv2.INTER_CUBIC)
            imgN[:, beginW : beginW + newW] = img 
        elif (h / w) <= (_conH / _conW):
            newH = int(h * _conW / w)
            beginH = int(abs(_conH - newH) / 2)
            img = cv2.resize(img, (_conW, newH), interpolation=cv2.INTER_CUBIC)
            imgN[beginH : beginH + newH, :] = img

        def add_distort(img):
            if np.random.rand() < self.p_aug:
                img = np.pad(img, ((5, 5), (5, 5)), mode='constant', constant_values=255)
                img = generate_distort(img, np.random.randint(8, 20))
                img = cv2.resize(img, (self.conW, self.conH), interpolation=cv2.INTER_CUBIC)
            return img 

        def add_stretch(img):
            if np.random.rand() < self.p_aug:
                img = generate_stretch(img, np.random.randint(2, 20))
            return img 

        def add_perspective(img):
            if np.random.rand() < self.p_aug:
                img = generate_perspective(img)
            return img 

        if self.p_aug > 0:
            imgN = np.pad(imgN, ((2, 2), (2, 2)), mode='constant', constant_values=255).astype('uint8')
            add_aug_fns = [add_distort, add_stretch, add_perspective]
            shuffle(add_aug_fns)
            for fn in add_aug_fns:
                imgN = fn(imgN)
            if np.random.randn() < self.p_aug:
                imgN, _ = apply_data_augmentation(imgN, iam_augmentation)
                imgN = imgN[:, :, 0]

        imgN = imgN.astype('float32') 
        imgN = (imgN - 127.5) / 127.5 
        return imgN 

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.img_paths[idx], 0)
            label = self.labels[idx]
            img = self.aug_norm_img(img)
        except:
            print('corrupt')
            return self[(idx + 1) % len(self.img_paths)]

        if len(label) == 0:
            return self[(idx + 1) % len(self.img_paths)]

        img = img.reshape(1, self.conH, self.conW) 
        label = self.label_map.encode(label) 
        return torch.from_numpy(img), torch.IntTensor(label) 

