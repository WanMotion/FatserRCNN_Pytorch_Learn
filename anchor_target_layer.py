import numpy as np
from proposalLayer import generate_anchors
from utils.cython_bbox import bbox_overlaps, bbox_intersections
from config import *
import numpy.random as npr

def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride=[16, ],
                        anchor_scales=[4, 8, 16, 32]):
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    height,width=rpn_cls_score.shape[2:4]
    # 和proposal_layer基本一致，生成anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    total_anchors=int(K*A)

    # 只允许anchor出现在图片内部
    _allowed_border = 0
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]
    anchors=all_anchors[inds_inside,:]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels=np.empty((len(inds_inside),),dtype=np.float32) # (A,)
    labels.fill(-1)

    # 重叠
    # 计算anchors和gt_boxes之间重叠的比例，值=重叠部分面积/两者平面面积和
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)) # (A,G)
    # 获取每一行的最大值的编号,即，对于每个anchor，能在gt_box中能找到的最大重合度
    argmax_overlaps=overlaps.argmax(axis=1)
    max_overlaps=overlaps[np.arange(len(inds_inside)),argmax_overlaps]
    # 获取每一列的最大值的编号,即，对于每一个gt_box，能在所有anchor中能找到的最大重合度
    gt_argmax_overlaps=overlaps.argmax(axis=0)
    gt_max_overlaps=overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    # 在overlaps中找出和gt_max_overlaps有相同值的点
    gt_argmax_overlaps=np.where(overlaps==gt_max_overlaps)[0]

    if not RPN_CLOBBER_POSITIVES:
        labels[max_overlaps<RPN_NEGATIVE_OVERLAP]=0
    # fg label: for each gt, anchor with highest overlap 和真值相等的设为1
    labels[gt_argmax_overlaps]=1
    # fg label: above threshold IOU 大于阈值的设为1
    labels[max_overlaps>=RPN_POSITIVE_OVERLAP]=1

    if RPN_CLOBBER_POSITIVES:
        labels[max_overlaps<RPN_NEGATIVE_OVERLAP]=0

    # 处理无需关心的区域
    if dontcare_areas is not None and dontcare_areas.shape[0]>0:
        # bbox_intersections计算无需关心区域的面积与anchor的面积之比
        intersecs = bbox_intersections(
            np.ascontiguousarray(dontcare_areas, dtype=np.float),  # D x 4
            np.ascontiguousarray(anchors, dtype=np.float)  # A x 4
        )  # (D,A)
        # 计算出每一列的和，即：对于每一个anchor，计算它和所有无需关心区域重叠部分比例和
        intersecs_=intersecs.sum(axis=0) # (A,)
        # 将和大于阈值的anchor的label置为-1
        labels[intersecs_>DONTCARE_AREA_INTERSECTION_HI]=-1

    # 处理较为偶然的、截断的、难以识别的部分
    if PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0]>0:
        assert  gt_ishard.shape[0]==gt_boxes.shape[0]
        gt_ishard=gt_ishard.astype(int)
        gt_hardboxes=gt_boxes[gt_ishard==1,:]
        if gt_hardboxes.shape[0]>0:
            hard_overlaps=bbox_overlaps(# 计算重叠比例,H x A
                np.ascontiguousarray(gt_hardboxes, dtype=np.float),  # H x 4
                np.ascontiguousarray(anchors, dtype=np.float))  # A x 4
            hard_max_overlaps=hard_overlaps.max(axis=0) # 每一列(A,)
            labels[hard_max_overlaps>=RPN_POSITIVE_OVERLAP]=-1
            max_intersec_label_inds=hard_overlaps.argmax(axis=1) # (H,)
            labels[max_intersec_label_inds]=-1
    # 如果有过多的positive sample，再次采样
    num_fg=int(RPN_FG_FRACTION*RPN_BATCHSIZE)
    fg_inds=np.where(labels==1)[0]
    if len(fg_inds)>num_fg:
        disable_inds=npr.choice(fg_inds,size=(len(fg_inds)-num_fg),replace=False)
        labels[disable_inds]=-1
    # 如果有过多的negative label，再次筛选
    num_bg=RPN_BATCHSIZE-np.sum(labels==1)
    bg_inds=np.where(labels==0)[0]
    if len(bg_inds)>num_bg:
        disable_inds=npr.choice(bg_inds,size=(len(bg_inds)-num_bg),replace=False) # 从数组中随机抽取出size个元素
        labels[disable_inds]=-1

    # 计算回归目标
    bbox_targets=_compute_targets(anchors,gt_boxes[argmax_overlaps,:])
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels==1,:]=np.array(RPN_BBOX_INSIDE_WEIGHTS)
    bbox_outside_weights=np.zeros((len(inds_inside),4),dtype=np.float32)
    if RPN_POSITIVE_WEIGHT<0:
        positive_weights=np.ones((1,4))
        negative_weights=np.zeros((1,4))
    else:
        assert ((RPN_POSITIVE_WEIGHT>0)&(RPN_POSITIVE_WEIGHT<1))
        positive_weights=(RPN_POSITIVE_WEIGHT/(np.sum(labels==1))+1)
        negative_weights=((1.0-RPN_POSITIVE_WEIGHT)/(np.sum(labels==0))+1)

    bbox_outside_weights[labels==1,:]=positive_weights
    bbox_outside_weights[labels==0,:]=negative_weights

    # labels
    labels=labels.reshape((1,height,width,A))
    labels=labels.transepose(0,3,1,2)
    rpn_labels=labels.reshape((1,1,A*height,width)).transpose(0,2,3,1)

    # targets
    bbox_targets=bbox_targets.reshape((1,height,width,A*4)).transpose(0,3,1,2)
    rpn_bbox_tragets=bbox_targets

    # bbox_inside_weights
    bbox_inside_weights=bbox_inside_weights.reshape((1,height,width,A*4)).transpose(0,3,1,2)
    rpn_bbox_inside_weights=bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights=bbox_outside_weights.reshape((1,height,width,A*4)).transpose(0,3,1,2)
    rpn_bbox_outside_weights=bbox_outside_weights

    return rpn_labels,rpn_bbox_tragets,rpn_bbox_inside_weights,rpn_bbox_outside_weights

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)



def bbox_transform(ex_rois, gt_rois):
    """
    计算真值到given boxes的距离，返回deltas
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
    #     'Invalid boxes found: {} {}'. \
    #         format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret