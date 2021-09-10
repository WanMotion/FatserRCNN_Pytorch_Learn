import numpy as np

from nms_wrapper import *
from generate_anchors import generate_anchors
from config import *


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride=[16, ],
                   anchor_scales=[8, 16, 32]):
    """
    :param rpn_cls_prob: 每个框的可能性，(1,18,W,H)
    :param rpn_bbox_pred: 预测出的每个框,(1,36,W,H)
    :param im_info: 图片信息，[高，宽，缩放倍数]
    :param cfg_key: TRAIN or TEST
    :param _feat_stride: 降采样倍数
    :param anchor_scales: anchor缩放倍数
    :return: (H*W*A,5),eg:[0,x1,y1,x2,y2]，A为feature map中每个点生成的anchor数，默认为9
    """
    # 根据anchor_scales生成的基础9个anchors
    _anchors = generate_anchors(scales=np.array(anchor_scales))  # (9,4)
    _num_anchors = _anchors.shape[0]

    scores = rpn_cls_prob[:, _num_anchors:, :, :]  # 背景评分
    bbox_deltas = rpn_bbox_pred  # 预测的前景框

    # 1.生成anchors
    # 生成W*H个偏移量，然后将基础anchors加上这些偏移量生成W*H*9个anchors
    height, width = scores.shape[-2:]  # (W,H)
    shift_x = np.arange(0, width) * _feat_stride  # 乘以降采样倍数，默认为16
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 构成 width * height 个点， shift_x为x轴取值，shift_y为y轴取值
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()  # (W*H,4)

    # 共有9个anchors，每个anchor有W*H个变换方式
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    A = _num_anchors  # 9
    K = shifts.shape[0]  # W*H
    anchors = _anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))  # 最终生成K*A个anchors，这些anchors对于每个图片来说都是一样的，需要bbox_deltas来加以修正

    # reshape
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))  # 原shape为(1,4,W,H)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # 通过传入的bbox_deltas修正得到anchors
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. 调整包含图像外侧的anchor
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. 过滤掉比min_size小的anchor
    keep = _filter_boxes(proposals, RPN_MIN_SIZE * im_info[2])  # RPN_NMS_THRESH*图片缩放倍数
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. 按照score排序获得的proposal
    # 5. 获取前RPN_PRE_NMS_TOP_N个anchor
    order = scores.ravel().argsort()[::-1]  # -1代表获取倒序
    if RPN_PRE_NMS_TOP_N>0:
        order=order[:RPN_PRE_NMS_TOP_N]
    proposals=proposals[order,:]
    scores=scores[order]

    # 6. 应用非极大值抑制算法，去除重合率大的冗余候选框，得到最具代表性的结果，以加快目标检测的效率
    keep=nms(np.hstack((proposals,scores)),RPN_NMS_THRESH)
    # 7. 获取前RPN_NMS_THRESH个应用了nms后的anchor
    if RPN_POST_NMS_TOP_N>0:
        keep=keep[:RPN_POST_NMS_TOP_N]
    proposals=proposals[keep,:]
    scores=scores[keep]

    batch_inds=np.zeros((proposals.shape[0],1),dtype=np.float32)
    blob=np.hstack((batch_inds,proposals.astype(np.float32,copy=False)))
    return blob


def bbox_transform_inv(boxes, deltas):
    """
    :param boxes: 生成的anchors
    :param deltas: 网络计算得出的deltas
    :return: 返回经过运算后偏移完成的anchors
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 0), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    # 得到宽度高度坐标
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_xs = boxes[:, 0] + widths * 0.5
    ctr_ys = boxes[:, 1] + heights * 0.5

    # 得到deltas
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # 生成预测的anchors
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_xs[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_ys[:np.newaxis]
    pre_w = np.exp(dw) * widths[:, np.newaxis]
    pre_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pre_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pre_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pre_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pre_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    调整包含图像外侧的anchor
    :param boxes: 所有anchor
    :param im_shape: 图像的shape
    :return: 调整后的anchor
    """
    if boxes.shape[0] == 0:
        return boxes

    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)


def _filter_boxes(boxes, min_size):
    """
    剔除所有比min_size小的anchor
    :param boxes: 所有的anchors
    :param min_size: 最小的size
    :return: 剔除后的anchor
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1.0
    hs = boxes[:, 3] - boxes[:, 1] + 1.0
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

