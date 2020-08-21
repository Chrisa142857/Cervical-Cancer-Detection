from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


def xyxy2xywh_pure(x):
    y = x.clone()
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    y[..., 0] = x[..., 0] + y[..., 2] / 2
    y[..., 1] = x[..., 1] + y[..., 3] / 2
    return y


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, folder='images', with_unsure=False, fp_flag=False, softnms=False):
    model.eval()
    PATCH_SAVE_TXT = 'Z:/wei/PCDD/test_txts/wei_CSPresnext_pos_step60k'+ str(img_size[0]) +'nms'+str(nms_thres)+'conf'+str(conf_thres)+'.txt'
    # Get dataloader
    dataset = ListDataset(path, folder=folder, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn
    )
    infer_times = []

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    save_str = ''
    for batch_i, (img_fns, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        new_img_size = (imgs.shape[3], imgs.shape[2])
        new_scale = (new_img_size[0] / 1024, new_img_size[1] / 1024)
        with torch.no_grad():
            dt1=datetime.datetime.now()
            outputs = model(imgs, fp_flag=fp_flag)
            # print("mean conf", outputs[0, :, 4].max())
            # input()
            dt2=datetime.datetime.now()
            print('Infer: ', dt2-dt1)
            infer_times.append(dt2-dt1)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            # print("after nms", [o.shape for o in outputs if o is not None])
            # input()

        dt1 = datetime.datetime.now()
        for b_i, (_o, img_fn) in enumerate(zip(outputs, img_fns)):
            img_fn = img_fn[img_fn.rfind('/') + 1:]
            dataId = img_fn[:-4]
            if _o is not None:
                global_outputs = _o.clone()
                global_outputs[:, 0] = (global_outputs[:, 0]) / new_scale[0]
                global_outputs[:, 1] = (global_outputs[:, 1]) / new_scale[1]
                global_outputs[:, 2] = (global_outputs[:, 2]) / new_scale[0]
                global_outputs[:, 3] = (global_outputs[:, 3]) / new_scale[1]
                xmins = global_outputs[:, 0].tolist()
                ymins = global_outputs[:, 1].tolist()
                xmaxs = global_outputs[:, 2].tolist()
                ymaxs = global_outputs[:, 3].tolist()
                scores = global_outputs[:, 4].tolist()
                for score, xmin, ymin, xmax, ymax in zip(scores, xmins, ymins, xmaxs, ymaxs):
                    save_str += '%s %f %f %f %f %f\n' % (dataId, score, xmin, ymin, xmax, ymax)

    with open(PATCH_SAVE_TXT, 'w') as f:
        f.write(save_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/csresnext50-panet-spp.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="CSPresnext50_pcdd1024/yolov3_ckpt_step_59998.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="Z:/wei/PyTorch-YOLOv3-master/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1024, help="size of each image dimension")
    parser.add_argument("--gpu_id", type=str, default="1")
    parser.add_argument("--folder", type=str, default="test_images1024_new")
    parser.add_argument("--fp_train", type=bool, default=False)
    parser.add_argument("--fp_max", type=float, default=0.5)
    parser.add_argument("--use_mish", type=bool, default=False)
    parser.add_argument("--softnms", type=bool, default=False)
    parser.add_argument("--mscale", type=bool, default=False)
    opt = parser.parse_args()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_path = 'Z:/wei/PCDD/data_sets/images1024_test_new.txt'

    # Initiate model
    model = Darknet(opt.model_def, fp_max=opt.fp_max).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    evaluate(
        model,
        folder=opt.folder,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=(opt.img_size, opt.img_size),
        batch_size=opt.batch_size,
        with_unsure=False,
        fp_flag=opt.fp_train,
    )
    # evaluate(
    #     model,
    #     folder=opt.folder,
    #     path=valid_path,
    #     iou_thres=opt.iou_thres,
    #     conf_thres=opt.conf_thres,
    #     nms_thres=opt.nms_thres,
    #     img_size=(1536, 1536),
    #     batch_size=opt.batch_size,
    #     with_unsure=False,
    #     fp_flag=opt.fp_train,
    # )