import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from utils.augmentations import horisontal_flip, strong_rand_aug
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(
        self, 
        list_path, 
        folder="images", 
        convert_channel=False, 
        need_old_wh=False, 
        img_size=(864, 864), 
        augment=True, 
        multiscale=True, 
        normalized_labels=True, 
        fp_dict={}
        ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace(folder, "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.rand_size_len = 5
        self.batch_count = 0
        self.fp_dict = fp_dict
        self.convert_channel = convert_channel
        self.need_old_wh = need_old_wh

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')
        if self.convert_channel:
            img = Image.fromarray(cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR))         
            
        # _, h, w = img.shape
        w, h = img.size

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            boxes = boxes[torch.where(boxes[:, 1]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 1]<=1)[0], :]
            boxes = boxes[torch.where(boxes[:, 2]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 2]<=1)[0], :]
            boxes = boxes[torch.where(boxes[:, 3]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 3]<=1)[0], :]
            boxes = boxes[torch.where(boxes[:, 4]>0)[0], :]
            boxes = boxes[torch.where(boxes[:, 4]<=1)[0], :]

            # Apply augmentations
            if self.augment:
                bboxes_for_aug = [[cx.item(), cy.item(), bw.item(), bh.item()] for cx, cy, bw, bh in boxes[:, 1:]]
                category_for_aug = [box[0].item() for box in boxes]
                img_for_aug = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                aug_input = {"image": img_for_aug, "bboxes": bboxes_for_aug, "category_id": category_for_aug}
                if w == h:
                    aug_output = strong_rand_aug(p=0.5, width=self.img_size[0], height=self.img_size[1])(**aug_input)
                else:
                    aug_output = strong_rand_aug(no_rotate=True, p=0.5, width=self.img_size[0], height=self.img_size[1])(**aug_input)
                img = aug_output['image']
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if len(bboxes_for_aug) != 0:
                    boxes = torch.cat([torch.FloatTensor([clas, cx, cy, bw, bh]).unsqueeze(0) for clas, (cx, cy, bw, bh ) in zip(aug_output['category_id'], aug_output['bboxes'])])

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        # if img.shape[1] == 1936:
        #     img = img[:, :-16, :]
        if not self.need_old_wh:
            return img_path, img, targets 
        else:
            return img_path, img, targets, w

    def collate_fn(self, batch):
        if not self.need_old_wh:
            paths, imgs, targets = list(zip(*batch))
        else:
            paths, imgs, targets, ws = list(zip(*batch))
        # Remove empty placeholder targets
        if targets != None:
            targets = [boxes for boxes in targets if boxes is not None]
            # Add sample index to targets
            for i, boxes in enumerate(targets):
                boxes[:, 0] = i
            if targets != []:
                targets = torch.cat(targets, 0)
            else:
                targets = torch.zeros((0, 6))
            # Concat FP
            fps = []
            for b_i, path in enumerate(paths):
                if path not in self.fp_dict:
                    self.fp_dict[path] = torch.FloatTensor(0, 5)
                fps.append(torch.cat((torch.FloatTensor(self.fp_dict[path].shape[0], 1).fill_(b_i), self.fp_dict[path]), 1))
            fps = torch.cat(fps)
            targets = torch.cat((targets, fps))
        new_img_size = self.img_size
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            rand_scale = random.choice(range(self.rand_size_len * -1, 1, 2))
            if self.img_size == (1216, 1936):
                new_img_size = (1216 + rand_scale*64, 1920 + rand_scale*64)
            else:
                new_img_size = (self.img_size[0] + rand_scale*64, self.img_size[1] + rand_scale*64)
    
        # Resize images to input shape
        imgs = torch.stack([resize(img, new_img_size) for img in imgs])

        if imgs.shape[-1] == 1936:
            imgs = imgs[..., :-16]
        self.batch_count += 1
        if not self.need_old_wh:
            return paths, imgs, targets
        else:
            return paths, imgs, targets, ws

    def __len__(self):
        return len(self.img_files)
