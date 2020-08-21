from sklearn.manifold import TSNE
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import *
from utils.datasets import *
from utils.utils import *
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# _model_def = "config/yolov3-custom.cfg"
_model_def = "config/csresnext50-panet-spp-sRMB.cfg"
# _w_path = "darknet53_AOT1024/yolov3_ckpt_step_1089998.pth"
_w_path = "csresnext50sRMB_AOT1024/yolov3_ckpt_step_149998.pth"
# _w_path = "weights/yolov3_ckpt_darknet53_fp_init.pth"
# _img_path = "Z:/wei/tianchi_cevical/whole_roi_detection/images_tianchi864_whole_roi/T2019_731_8139_9526_864.jpg"
# _img_path = "Z:/wei/PyTorch-YOLOv3-master/data/custom/images_8896/goldtest_1110441 0893050_9694_101795_21448_21448.jpg"
# _img_path = "Z:/wei/All_Of_Them/images1024/OURSFY3P_1179860_11108_1827_1024.jpg"
# _img_path = "Z:/wei/All_Of_Them/images1024/OURSFY3P_2018-11-01-203533-337_37856_4070_1024.jpg"
_img_path = 'Z:/wei/All_Of_Them/images1024/SDPCTJ6P_tj190619346_51549_112250_1663.jpg'
# _img_path = "Z:/wei/PyTorch-YOLOv3-master/data/custom/images_8896/patches/goldtest_1110441 0893050_9694_101795_21448_21448_3272_0_1936.jpg"
# _label_path = "Z:/wei/All_Of_Them/labels/OURSFY3P_1179860_11108_1827_1024.txt"
# _label_path = "Z:/wei/All_Of_Them/labels/OURSFY3P_2018-11-01-203533-337_37856_4070_1024.txt"
_label_path = "Z:/wei/All_Of_Them/labels/SDPCTJ6P_tj190619346_51549_112250_1663.txt"

fp_flag = False
dd = 'cpu'
x_pre_filters = "14"

def detect(model, img, targets):

    x = img.to(dd)
    img_dim = x.shape[2:4]
    layer_outputs, yolo_inputs, yolo_layers, fms = [], [], [], []
    loss = 0
    
    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        if module_def["type"] == "convolutional":
            if module_def["filters"] == x_pre_filters:
                x_pre = x.clone()
                fms.append(x_pre)
        if module_def["type"] in ["convolutional", "upsample", "maxpool", "sResMB"]:
            x = module(x)
        elif module_def["type"] == "reorg3d":
            x = reorg3d(x, int(module_def["stride"]))
        elif module_def["type"] == "route":
            x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
        elif module_def["type"] == "shortcut":
            layer_i = int(module_def["from"])
            x = layer_outputs[-1]
            a = layer_outputs[layer_i]
            nx = x.shape[1]
            na = a.shape[1]
            if nx == na:
                x = x + a
            elif nx > na:
                x[:, :na] = x[:, :na] + a
            else:
                x = x + a[:, :nx]
            x = module(x)
        elif module_def["type"] == "yolo":
            yolo_inputs.append(x)
            x, layer_loss = module[0](x, targets.to(dd), img_dim, fp_flag=fp_flag)
            loss += layer_loss
            yolo_layers.append(module[0])
        layer_outputs.append(x)
    print("loss: ", loss)
    return yolo_inputs, yolo_layers, fms

def calculate_dis(p, ps):

    disX = ps[:, 0] - p[0]
    disY = ps[:, 1] - p[1]
    dis = np.sqrt(disX ** 2 + disY ** 2)
    return dis

def main():
    img = Image.open(_img_path).convert('RGB')
    n_anchor = 2
    img_w = 1024
    img_h = 1024
    iw, ih = img.size
    plt.imshow(img)
    boxes = torch.from_numpy(np.loadtxt(_label_path).reshape(-1, 5))
    for i_i, (x,y,w,h) in enumerate(boxes[:,1:]):
        plt.gca().add_patch(
            plt.Rectangle(((x-w/2)*iw, (y-h/2)*ih), width=w*iw, height=h*ih, fill=False)
            )
    plt.show()
    # exit()

    boxes = torch.from_numpy(np.loadtxt(_label_path).reshape(-1, 5))
    targets = torch.zeros((len(boxes), 6))
    targets[:, 1:] = boxes
    img = transforms.ToTensor()(img) #[:,:,:-16]
    img = torch.stack([resize(img, (img_w, img_h))])
    print(img.shape)
    model = Darknet(_model_def, fp_max=0.5, old_version=False).to(dd)
    model.load_state_dict(torch.load(_w_path))
    model.eval()
    yolo_inputs, yolo_layers, fms = detect(model, img, targets)
    img = transforms.ToPILImage()(img[0])

    print("yolo nums: ", len(yolo_inputs))
    print("yolo nums: ", len(yolo_layers))
    print("yolo nums: ", len(fms))

    for i, (x, yolo_layer, x_pre) in enumerate(zip(yolo_inputs[::-1], yolo_layers[::-1], fms[::-1])):
        # if i == 1 :break
        grid_size = yolo_layer.grid_size
        grid_x = yolo_layer.grid_x[0][0].to("cpu").reshape(-1, 1)
        grid_y = yolo_layer.grid_y[0][0].to("cpu").reshape(-1, 1)
        stride_w = img_w/grid_size[0]
        stride_h = img_h/grid_size[1]

        prediction = (
            x.view(1, n_anchor, 7, grid_size[0], grid_size[1])
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # prediction = prediction[:, :,102:162 ,0:38, :]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_fp_conf = torch.sigmoid(prediction[..., 5])
        pred_cls = torch.sigmoid(prediction[..., 6:])
        if fp_flag:
            pred_final_conf = pred_conf * (1 - pred_fp_conf)
        else:
            pred_final_conf = pred_conf
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape)

        anchors = yolo_layer.anchors
        scaled_anchors = torch.FloatTensor([(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]).to(dd)

        class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, fp_mask, tfpconf = build_targets(pred_boxes, pred_cls, targets.to(dd), scaled_anchors, 0.5, label_adjustment=0.05, use_fp_score=True)
        
        masks = torch.round(pred_final_conf).bool()
        
        plt.figure()
        for n in range(n_anchor):
            plt.subplot(n_anchor,2,2*n+1)
            plt.imshow(obj_mask.to('cpu')[0][n].detach().numpy())
            plt.subplot(n_anchor,2,2*n+2)
            plt.imshow(masks.to('cpu')[0][n].detach().numpy())
        plt.show()

        obj_mask = obj_mask[0].reshape(n_anchor, -1).permute(1, 0).to('cpu').detach().numpy()
        print("obj_mask.shape: ", obj_mask.shape)
        pred_score = pred_final_conf[0].reshape(n_anchor, -1).permute(1, 0).to('cpu').detach().numpy()
        print("pred_score.shape: ", pred_score.shape)
        masks = masks[0].reshape(n_anchor, -1).permute(1, 0).to('cpu').detach().numpy()
        print("pred_mask.shape: ", masks.shape)
        _masks = ~masks

        # print(mask.shape, mask1, mask2, mask3)
        fm = x_pre.to('cpu')#[:, :, 0:38, 102:162]
        print("fm.shape: ", fm.shape)
        # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        fm = fm[0].reshape(fm.shape[1], -1).permute(1, 0).detach().numpy()
        # fm = fm[::-1, :]

        fm_embedded = TSNE(n_components=2, random_state=49).fit_transform(fm)
 
        print("fm_embedded.shape: ", fm_embedded.shape)
        pred_score = (pred_score*20 + 1)

        while input("Input 'n' to next yolo layer or Any other keys to...: ") != "n":
            plt.ion()
            plt.figure()
            for n in range(n_anchor):
                ids = np.where(_masks[:, n])
                plt.scatter(fm_embedded[ids, 0], fm_embedded[ids, 1], s=pred_score[ids, 0], color='RED')

            for n in range(n_anchor):
                ids = np.where(masks[:, n])
                plt.scatter(fm_embedded[ids, 0], fm_embedded[ids, 1], s=pred_score[ids, 0], color='BLUE')
            
            for n in range(n_anchor):
                ids = np.where(obj_mask[:, n])
                plt.scatter(fm_embedded[ids, 0], fm_embedded[ids, 1], linewidths=10, marker="+", color='GREEN')

            # plt.show()
            p = plt.ginput(1)[0]
            print("Click Point: ", p)
            dis = calculate_dis(p, fm_embedded)
            id = np.argmin(dis)
            patch_x = grid_x[id] * stride_w
            patch_y = grid_y[id] * stride_h
            x_min, y_min, x_max, y_max = int(patch_x.item()-3*stride_w), int(patch_y.item()-3*stride_h), int(patch_x+4*stride_w), int(patch_y+4*stride_h)
            print("Patch xy: ", (patch_x, patch_y))
            plt.figure()
            plt.imshow(img)
            plt.gca().add_patch(plt.Rectangle((patch_x, patch_y), width=stride_w, height=stride_h, fill=True, color="RED"))
            # patch = .crop((x_min, y_min, x_max, y_max))
            plt.ioff()
            plt.show()
            


if __name__ == "__main__":
    # try:
    main()
    # except:
    #     print("Exit~")