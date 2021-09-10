import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import softmax

import VGG16Model
from proposalLayer import *


class RPN(nn.Module):
    anchor_scales=[8,16,32]
    _feat_stride = [16, ]
    def __init__(self):
        super(RPN, self).__init__()
        # 设输入的图像的尺寸为(M,N)
        self.features=VGG16Model() # (512,M/16,N/16)
        # feature map 后3*3的卷积层
        self.conv1=nn.Conv2d(512,512,(3,3),(1,1),padding=1)
        self.relu1=nn.ReLU()
        # 每个点对应9个anchor，每个anchor两个score
        self.score_conv=nn.Conv2d(512,len(self.anchor_scales)*3*2,(1,1)) # 经过此卷积后，矩阵shape为(18,M/16,N/16)
        # 每个点对应9个anchor，每个anchor对应4个坐标表示
        self.bbox_conv=nn.Conv2d(512,len(self.anchor_scales)*3*4,(1,1))  # 经过此卷积后，矩阵shape为(36,M/16,N/16)

        # loss
        self.cross_entropy=None
        self.loss_box=None

    def forward(self,im_data,im_info,gt_boxes=None,gt_ishard=None,doncare_areas=None):
        # 先reshape permute(0,3,1,2)
        features=self.features(im_data) # 得到feature map
        rpn_conv1=self.conv1(features)
        rpn_conv1=self.relu1(rpn_conv1)

        # rpn score, W=M/16,H=N/16
        rpn_cls_score=self.score_conv(rpn_conv1) # (1,18,W,H)
        # 计算softmax
        rpn_cls_prob=softmax(rpn_cls_score)  # (1,18.W,H)

        # rpn boxes
        rpn_bbox_pre=self.bbox_conv(rpn_conv1) # (1,36,W,H)

        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois=self.proposal_layer(rpn_cls_prob,rpn_bbox_pre,cfg_key,self._feat_stride,self.anchor_scales)

        # 生成训练标签，构建rpn loss
        if self.training:
            assert  gt_boxes is not None
            rpn_data

        # 返回feature map 和 得到的anchors
        return features,rois


    @staticmethod
    def proposal_layer(rpn_cls_prob,rpn_bbox_pred,im_info,cfg_key,_feat_stride,anchor_scales):
        rpn_cls_prob = rpn_cls_prob.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x=proposal_layer(rpn_cls_prob,rpn_bbox_pred,im_info,cfg_key,_feat_stride,anchor_scales)
        x=Variable(x,torch.from_numpy(x),torch.FloatTensor)
        x=x.cuda()
        return x

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
        """
        :param rpn_cls_score: (1,A*2,H,W) bg/fg scores
        :param gt_boxes: (G,5) [x1,y1,x2,y2,class] 真值
        :param gt_ishard:(G,1) 1 or 0 indicates difficult or not
        :param dontcare_areas:(D, 4), some areas may contains small objs but no labelling. D may be 0
        :param im_info:[高, 宽, 缩放比]
        :param _feat_stride: 降采样率
        :param anchor_scales:缩放比
        :return:
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)


