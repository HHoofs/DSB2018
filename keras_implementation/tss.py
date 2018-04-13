from skimage.segmentation import mark_boundaries

from PIL import Image

import numpy as np
import os
import shutil

from skimage.segmentation import find_boundaries
from skimage.morphology import dilation
from scipy.ndimage.measurements import center_of_mass
from scipy.misc import imresize


def find_all_samples(path):
    all_samples = os.listdir(path)
    return all_samples


def create_mask(path, width, height):
    samples = find_all_samples(path)
    for sample in samples:
        if sample != '.DS_Store':
            sample_path = os.path.join(path, sample)
            sample_path_masks = os.path.join(sample_path, 'masks')
            masks = os.listdir(sample_path_masks)
            complete_mask = np.zeros((width, height), dtype=int)
            for mask in masks:
                with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                    _mask = _mask.resize((width, height))
                    _mask = np.array(_mask)
                    complete_mask = np.maximum(complete_mask, _mask)
            os.mkdir(os.path.join(sample_path, 'mask'))
            mask_image = Image.fromarray(complete_mask.astype('uint8'), 'L')
            mask_image.save(os.path.join(sample_path, 'mask', '{}.png'.format(sample)))


def create_border_mask(path, width, height):
    samples = find_all_samples(path)
    for sample in samples:
        if sample != '.DS_Store':
            sample_path = os.path.join(path, sample)
            sample_path_masks = os.path.join(sample_path, 'masks')
            masks = os.listdir(sample_path_masks)
            complete_mask = np.zeros((width, height), dtype=int)
            for i, mask in enumerate(masks):
                with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                    _array = np.array(_mask.resize((width, height)))
                    _array = np.array(_array > 0, dtype=int) * (i + 1)
                    complete_mask = np.add(complete_mask, _array)
            full_bounds = np.array(complete_mask, dtype=int)
            bound_array = find_boundaries(label_img = full_bounds, connectivity = 1, mode='outer', background=0)
            bound_array = np.array(bound_array, dtype=int)
            bound_array = dilation(bound_array)
            bound_array = bound_array * 255
            print(np.max(bound_array))

            # save image
            try:
                shutil.rmtree(os.path.join(sample_path, 'border'))
            except:
                pass
            os.mkdir(os.path.join(sample_path, 'border'))
            mask_image = Image.fromarray(bound_array.astype('uint8'), 'L')
            mask_image.save(os.path.join(sample_path, 'border', '{}.png'.format(sample)))


def foo(l, dtype=int):
    return map(dtype, l)

def create_border_hyper_mask(path, width, height):
    samples = find_all_samples(path)[:2]
    for sample in samples[:]:
        sample_path = os.path.join(path, sample)
        sample_path_masks = os.path.join(sample_path, 'masks')
        masks = os.listdir(sample_path_masks)
        complete_mask = np.zeros((width, height), dtype=int)
        complete_boun = np.zeros((width, height), dtype=int)
        for i, mask in enumerate(masks):
            with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                _array = np.array(_mask.resize((width, height)))
                bound_array = find_boundaries(label_img = _array, mode='outer', background=0)
                bound_array = dilation(bound_array)
                bound_array[_array == 1] = 0
                complete_boun = np.add(complete_boun, bound_array)
        complete_boun = complete_boun ** 2 / 9




        complete_boun = complete_boun * 255
        complete_boun[complete_boun > 255] = 255

        # save image
        try:
            shutil.rmtree(os.path.join(sample_path, 'smashing_border'))
        except:
            pass
        os.mkdir(os.path.join(sample_path, 'smashing_border'))
        mask_image = Image.fromarray(complete_boun.astype('uint8'), 'L')
        mask_image.save(os.path.join(sample_path, 'smashing_border', '{}.png'.format(sample)))

create_border_hyper_mask('C:/Users/huubh/Documents/DSB2018_bak/img', 256, 256)
# create_mask('/Users/HuCa/Documents/DSB2018/stage1_train', 256, 256)
# create_border_mask('/Users/HuCa/Documents/DSB2018/stage1_train', 256, 256)





ttt = np.array([[1,1,0,0],[1,0,0,0],[0,0,0,2]])
