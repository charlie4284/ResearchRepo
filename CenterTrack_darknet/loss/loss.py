import torch
import numpy as np


def focal_loss(hm_pred, hm_true):
    pos_mask = hm_true.eq(1).float()
    neg_mask = hm_true.lt(1).float()
    neg_weights = (1-hm_true).pow(4)

    pos_loss = -torch.log(hm_pred.clamp(1e-4, 1. - 1e-4)
                          ) * (1-hm_pred).pow(2) * pos_mask
    neg_loss = -torch.log((1 - hm_pred).clamp(1e-4, 1. - 1e-4)) * \
        hm_pred.pow(2) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    cls_loss = torch.where(num_pos.gt(
        0), (pos_loss + neg_loss)/num_pos, neg_loss)
    return cls_loss


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def reg_l1_loss(y_pred, y_true, indices, mask):
    b = y_pred.shape[0]
    k = indices.shape[1]
    c = y_pred.shape[-1]
    y_pred = y_pred.view(b, -1, c)
    indices = indices.int()
    y_pred = y_pred.gather(1, indices)
    expanded_dim = torch.unsqueeze(mask, -1)
    mask = tile(expanded_dim, 2, 2)
    # mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = torch.abs(y_true * mask - y_pred * mask).sum()
    reg_loss = total_loss / (mask.sum() + 1e-4)
    return reg_loss


def loss(heatmap_pred, wh_pred, reg_pred, heatmap_truth, wh_truth, reg_truth, reg_mask, indices):
    heatmap_loss = focal_loss(heatmap_pred, heatmap_truth)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_truth, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_truth, indices, reg_mask)
    total_loss = heatmap_loss + wh_loss + reg_loss
    return total_loss
