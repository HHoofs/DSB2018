import os
import shutil


path = 'C:/Users/huubh/Documents/DSB2018_bak/img'

images =  os.listdir(path)

for image in images:
    subdirs = os.listdir(os.path.join(path, image))
    for subdir in subdirs:
        if subdir == 'border_small':
            shutil.copytree(os.path.join(path, image, subdir),
                            os.path.join('C:/Users/huubh/Documents/DSB2018_bak/img_no_masks', image, subdir))