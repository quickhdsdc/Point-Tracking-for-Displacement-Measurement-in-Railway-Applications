import numpy as np
import cv2
import os
from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names
import json
from random import randrange
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

## image corruption
def corruption(img_rgb):
    corruption_name = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur', 'snow', 'frost', 'fog', 'brightness'] 
    idx = randrange(9)
    severity = randrange(1)
    corruption = corruption_name[idx]
    img_corrupted = corrupt(img_rgb, corruption_name=corruption, severity=severity+1)
    # cv2.imshow('1',img_corrupted)
    return img_corrupted
## image augmentation
def augmentation(img_rgb,keypoints):
    # aug = [iaa.Affine(rotate=(-25, 25)), iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"), iaa.Cutout(), iaa.HorizontalFlip(), iaa.PerspectiveTransform(scale=0.08)]
    # idx = randrange(5)
    aug = [iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="edge"), iaa.Cutout()]
    idx = randrange(2)
    kps = [
    Keypoint(x=keypoints[0], y=keypoints[1]),   # ref_w
    Keypoint(x=keypoints[2], y=keypoints[3]),  # ref_r1
    Keypoint(x=keypoints[4], y=keypoints[5]) # ref_r2
    ]
    kpsoi = KeypointsOnImage(kps, shape=img_rgb.shape)
    img_aug, kpsoi_aug = aug[idx](image=img_rgb, keypoints=kpsoi)
    # ia.imshow(
    #     np.hstack([
    #         kpsoi.draw_on_image(img_rgb, size=7),
    #         kpsoi_aug.draw_on_image(img_aug, size=7)
    #     ])
    # )
    return img_aug,kpsoi_aug.get_coords_array()

# dirpath_img = './dataset/images/train256'
# dirpath_label = './dataset/labels/train256'

# dirpath_img = './dataset/images/test256'
# dirpath_label = './dataset/labels/test256'

dirpath_img = './dataset/images/valid256'
dirpath_label = './dataset/labels/valid256'

n = len(os.listdir(dirpath_img))
dir_img = os.listdir(dirpath_img)
dir_label = os.listdir(dirpath_label)

for i in range(n):
    # read data
    img_path = dirpath_img+'/'+dir_img[i]
    img_rgb = cv2.imread(img_path)
    label_path = dirpath_label+'/'+dir_label[i]
    keypoints = []
    with open(label_path) as json_file:
        data = json.load(json_file)
        ref_w = np.asarray(data['shapes'][0]['points'][0])        
        ref_r1 = np.asarray(data['shapes'][1]['points'][0])
        ref_r2 = np.asarray(data['shapes'][2]['points'][0])          
        keypoints.append(np.hstack((np.asarray(ref_w),np.asarray(ref_r1),np.asarray(ref_r2))))
    keypoints=np.reshape(np.asarray(keypoints),(6))
    # corruption and augmentation
    img_corrupted = corruption(img_rgb)
    # img_aug = img_corrupted
    # keypoints_aug = keypoints
    # keypoints_aug=np.reshape(np.asarray(keypoints_aug),(3,2))
    img_aug,keypoints_aug = augmentation(img_corrupted, keypoints)
    # save data
    img_aug_name = dir_img[i]
    # if dir_img[i].split('.')[0].split('_')[-1] == 'aug':
    #     img_aug_name = dir_img[i]
    # else:
    #     img_aug_name = dir_img[i].split('.')[0] + '_aug.png'
    img_aug_path = dirpath_img + '_aug' + '/' + img_aug_name
    cv2.imwrite(img_aug_path, img_aug)

    data['shapes'][0]['points'][0] = (int(keypoints_aug[0][0]),int(keypoints_aug[0][1]))
    data['shapes'][1]['points'][0] = (int(keypoints_aug[1][0]),int(keypoints_aug[1][1]))
    data['shapes'][2]['points'][0] = (int(keypoints_aug[2][0]),int(keypoints_aug[2][1]))
    data['imageHeight']=256
    data['imageWidth']=256 
    label_aug_name = dir_label[i]
    # if dir_label[i].split('.')[0].split('_')[-1] == 'aug':
    #     label_aug_name = dir_label[i]
    # else:
    #     label_aug_name = dir_label[i].split('.')[0] + '_aug.json'
    label_aug_path = dirpath_label + '_aug' + '/' + label_aug_name
    with open(label_aug_path, 'w') as json_file:
        json.dump(data, json_file)



