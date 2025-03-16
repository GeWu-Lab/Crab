import os
from os.path import join
import torch
from torch import nn
import torch.nn.functional as F


def metric_s_for_null(pred):
    # pred: 1,h,w
    assert len(pred.shape) == 3
    num_pixels = pred.view(-1).shape[0] 

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()

    x = torch.sum(pred.view(-1)) 
    s = torch.sqrt(x / num_pixels)

    return s


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou


def _eval_pr(y_pred, y, num, cuda_flag=False):
    if cuda_flag:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / \
            (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall



def Eval_Fmeasure(pred, gt, pr_num=255):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # print('=> eval [FMeasure]..')
    # =======================================[important]
    pred = torch.sigmoid(pred)
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)
    # print("{} videos in this batch".format(N))

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num

    return score.max().item()



def F1_Dice_loss(pred_masks, first_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)

    pred_mask = pred_masks.flatten(1)
    gt_mask = first_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()



def IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total ten frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total ten frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1)

    loss = nn.CrossEntropyLoss(reduction='none')(
        pred_mask, ten_gt_masks)  # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1)  # [bs*10]
    loss = loss * gt_temporal_mask_flag  # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss



def F5_Dice_loss(pred_mask, five_gt_masks):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = five_gt_masks.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask) # [bs*5, 1, 224, 224]
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss
    

'''
for ms and s4
'''
def IouSemanticAwareLoss(pred_mask, mask_feature, gt_mask, loss_type='F5_Dice_loss'):
    total_loss = 0

    if loss_type == 'bce':
        loss_func = F5_IoU_BCELoss
    elif loss_type == 'F5_Dice_loss':
        loss_func = F5_Dice_loss
    elif loss_type=='F1_Dice_loss':
        loss_func=F1_Dice_loss
    else:
        raise ValueError

    # loss_func = F5_Dice_loss

    iou_loss = 1.0 * loss_func(pred_mask, gt_mask)
    total_loss += iou_loss
    
    mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    mask_feature = F.interpolate(
        mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    mix_loss = 0.1*loss_func(mask_feature, gt_mask)
    total_loss += mix_loss

    return total_loss


'''
for avss
'''

def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total ten frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total ten frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1)

    loss = nn.CrossEntropyLoss(reduction='none')(
        pred_mask, ten_gt_masks)  # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1)  # [bs*10]
    loss = loss * gt_temporal_mask_flag  # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss


def Mix_Dice_loss(pred_mask, norm_gt_mask, gt_temporal_mask_flag):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = norm_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    loss = loss * gt_temporal_mask_flag
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
    return loss


def AVSSIouSemanticAwareLoss(pred_masks, mask_feature, gt_mask, gt_temporal_mask_flag):
    total_loss = 0
    loss_dict = {}

    iou_loss = 1.0 * \
        F10_IoU_BCELoss(pred_masks, gt_mask, gt_temporal_mask_flag)
    total_loss += iou_loss
    loss_dict['iou_loss'] = iou_loss.item()

    mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    mask_feature = F.interpolate(
        mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    one_mask = torch.ones_like(gt_mask)
    norm_gt_mask = torch.where(gt_mask > 0, one_mask, gt_mask)
    mix_loss = 0.1 * Mix_Dice_loss(mask_feature, norm_gt_mask, gt_temporal_mask_flag)
    total_loss += mix_loss
    
    return total_loss


'''
save avss mask
'''
import os
import numpy as np
from PIL import Image

def save_color_mask(pred_masks, save_base_path, video_name_list, filename, v_pallete, resize, resized_mask_size, T=10):
    # pred_mask: [bs*5, N_CLASSES, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    BT, N_CLASSES, H, W = pred_masks.shape
    bs = BT // T
    
    pred_masks = torch.softmax(pred_masks, dim=1)
    pred_masks = torch.argmax(pred_masks, dim=1) # [BT, 224, 224]
    pred_masks = pred_masks.cpu().numpy()
    
    pred_rgb_masks = np.zeros((pred_masks.shape + (3,)), np.uint8) # [BT, H, W, 3]
    for cls_idx in range(N_CLASSES):
        rgb = v_pallete[cls_idx]
        pred_rgb_masks[pred_masks == cls_idx] = rgb
    pred_rgb_masks = pred_rgb_masks.reshape(bs, T, H, W, 3)

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_rgb_masks[idx] # [5, 224, 224, 3]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            save_path = join(save_base_path,video_name,filename)
            im = Image.fromarray(one_mask)#.convert('RGB')
            if resize:
                im = im.resize(resized_mask_size)
            im.save(save_path, format='PNG')


def save_gt_mask(gt_masks, save_base_path, video_name_list, filename, v_pallete, resize, resized_mask_size, T=10):
    # gt_mask: [bs*5, 224, 224]
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    N_CLASSES = 71
    BT, H, W = gt_masks.shape
    bs = BT // T
    gt_masks = gt_masks.cpu().numpy()
    gt_rgb_masks = np.zeros((gt_masks.shape + (3,)), np.uint8) # [BT, H, W, 3]
    for cls_idx in range(N_CLASSES):
        rgb = v_pallete[cls_idx]
        gt_rgb_masks[gt_masks == cls_idx] = rgb
    gt_rgb_masks = gt_rgb_masks.reshape(bs, T, H, W, 3)

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = gt_rgb_masks[idx] # [5, 224, 224, 3]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            save_path = join(save_base_path,video_name,filename)
            im = Image.fromarray(one_mask)#.convert('RGB')
            if resize:
                im = im.resize(resized_mask_size)
            im.save(save_path, format='PNG')



'''
average 5 frames iou, then average all video iou.
'''
def compute_miou_from_jsonl(fp) -> dict:
    import jsonlines
    frame_nums = 0
    vid2miou = {}
    miou = 0.
    with jsonlines.open(fp,'r') as f:
        for idx, sample in enumerate(f):
            image_path = sample['image_path']
            vid = image_path.split('/')[-3]
            # frame_idx = int(image_path.split('/')[-1][:-4])
            iou = sample['iou']
            miou += iou
            frame_nums += 1
            if frame_nums == 5:
                miou = miou /5
                vid2miou[vid] = miou
                miou = 0
                frame_nums = 0
    
    miou = 0.
    for k,v in vid2miou.items():
        miou += v
    miou = miou / len(list(vid2miou.keys()))
    vid2miou['miou'] = miou
    return vid2miou


'''
compute miou and F-score for avss task
'''

def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))

    return ious, fscores, cls_count, vid_miou_list


def calc_color_miou_fscore(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """  
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    # miou, fscore, cls_count = _batch_miou_fscore(pred, target, nclass, T) 
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(pred, target, nclass, T) 
    return miou, fscore, cls_count, vid_miou_list


