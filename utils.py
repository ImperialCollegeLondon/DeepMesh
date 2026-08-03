import torch
import numpy as np
import cv2
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
import pdb
import math
import vtk
from torch.autograd import Variable
from skimage.morphology import binary_dilation, disk
import imageio
import os


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum((differences+1e-6)**2, -1).sqrt()
    return distances

def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    return res



def weightedHausdorff_batch(prob_loc, prob_vec, gt, height, width, temper, status):
    max_dist = math.sqrt(height ** 2 + width ** 2)

    # print (gt.shape)
    # print (gt.sum())
    # print (prob_vec.sum())
    batch_size = prob_loc.shape[0]
    # print (batch_size)


    term_1 = []
    term_2 = []

    for i in range(batch_size):
        prob_vec_sele = prob_vec[i, :, 0][prob_vec[i, :, 0] > torch.exp(torch.tensor((-1) * temper).cuda())]
        idx_sele_x = prob_loc[i, :, 0][prob_vec[i, :, 0] > torch.exp(torch.tensor((-1) * temper).cuda())]
        idx_sele_y = prob_loc[i, :, 1][prob_vec[i, :, 0] > torch.exp(torch.tensor((-1) * temper).cuda())]
        idx_sele = torch.stack((idx_sele_x, idx_sele_y), 1)


        # For case GT=0
        if gt[i,:,:].sum() == 0:
            if prob_vec_sele.sum() < 1e-3:
                if status=='train':
                    term_1.append(Variable(torch.tensor(0.0).cuda(), requires_grad=True))
                    term_2.append(Variable(torch.tensor(0.0).cuda(), requires_grad=True))
                else:
                    term_1.append(torch.tensor(0.0).cuda())
                    term_2.append(torch.tensor(0.0).cuda())
            else:
                if status == 'train':
                    term_1.append(Variable(torch.tensor(0.0).cuda(), requires_grad=True))
                    term_2.append(Variable((torch.tensor(max_dist)).cuda(), requires_grad=True))
                else:
                    term_1.append(torch.tensor(0.0).cuda())
                    term_2.append(torch.tensor(max_dist).cuda())
        else:
            if prob_vec_sele.sum() < 1e-3:
                if status == 'train':
                    term_1.append(Variable((torch.tensor(max_dist)).cuda(), requires_grad=True))
                    term_2.append(Variable(torch.tensor(0.0).cuda(), requires_grad=True))
                else:
                    term_1.append(torch.tensor(max_dist).cuda())
                    term_2.append(torch.tensor(0.0).cuda())
            else:
                # find nonzero point in gt
                idx_gt = torch.nonzero(gt[i, :, :])
                d_matrix = cdist(idx_sele, idx_gt)
                # print (d_matrix.shape) # N*M


                term_1.append(
                    (1 / (prob_vec_sele.sum() + 1e-6)) * torch.sum(prob_vec_sele * torch.min(d_matrix, 1)[0]))
                p_replicated = prob_vec_sele.view(-1, 1).repeat(1, idx_gt.shape[0])
                weighted_d_matrix = (1 - p_replicated) * max_dist + p_replicated * d_matrix
                minn = generaliz_mean(weighted_d_matrix, p=-7, dim=0, keepdim=False)
                term_2.append(torch.mean(minn))


    # print (term_1)
    # print (term_2)
    term_1 = torch.stack(term_1)
    term_2 = torch.stack(term_2)

    res = term_1.mean()+term_2.mean()


    return res



def huber_loss_3d(x):
    bsize, csize, depth, height, width = x.size()
    d_x = torch.index_select(x, 4, torch.arange(1, width).cuda()) - torch.index_select(x, 4, torch.arange(width-1).cuda())
    d_y = torch.index_select(x, 3, torch.arange(1, height).cuda()) - torch.index_select(x, 3, torch.arange(height-1).cuda())
    d_z = torch.index_select(x, 2, torch.arange(1, depth).cuda()) - torch.index_select(x, 2, torch.arange(depth-1).cuda())
    err = torch.sum(torch.mul(d_x, d_x))/width + torch.sum(torch.mul(d_y, d_y))/height + torch.sum(torch.mul(d_z, d_z))/depth
    err /= bsize
    tv_err = torch.sqrt(0.01+err)
    return tv_err




def projection(voxels, z_target, temper):
    # voxels are transformed from meshes based on affine information of different target plane
    # z_target is the z coordinate of the target plane, e.g., SAX is 12,17,22,27,32,37,42,47,52, 2CH is 0, 4CH is 0
    v_idx = voxels[:,:,0:2]  # [bs, numer_of verties, x/y coordinate]
    v_probability = torch.exp((-1) * temper * torch.square(voxels[:, :, 2:3] - z_target)) # [bs, numer_of verties, probability]


    return v_idx, v_probability



def distance_metric(pts_A, pts_B, dx):
    # Measure the distance errors between the contours of two segmentations
    # The manual contours are drawn on 2D slices.
    # We calculate contour to contour distance for each slice.
    # pts_A is N*2, pts_B is M*2
    if pts_A.shape[0] > 0 and pts_B.shape[0] > 0:
        # Distance matrix between point sets
        M = np.zeros((pts_A.shape[0], pts_B.shape[0]))
        for i in range(pts_A.shape[0]):
            for j in range(pts_B.shape[0]):
                M[i, j] = np.linalg.norm(pts_A[i, :] - pts_B[j, :])

        # Mean distance and hausdorff distance
        md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
        hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
    else:
        md = None
        hd = None

    return md, hd


def slice_2D(v_hat_es_cp, slice_num):
    idx_x = v_hat_es_cp[0, :, 0][torch.abs(v_hat_es_cp[0, :, 2] - slice_num) < 0.3]
    idx_y = v_hat_es_cp[0, :, 1][torch.abs(v_hat_es_cp[0, :, 2] - slice_num) < 0.3]
    idx_x_t = np.round(idx_x.detach().cpu().numpy()).astype(np.int16)
    idx_y_t = np.round(idx_y.detach().cpu().numpy()).astype(np.int16)
    idx = np.stack((idx_x_t, idx_y_t), 1)

    return idx


def compute_sa_mcd_hd(v_sa_hat_es_cp, contour_sa_es, sliceall):
    mcd_sa_allslice = []
    hd_sa_allslice = []

    slice_number = [1,4,7]
    threeslice = [sliceall[slice_number[0]], sliceall[slice_number[1]], sliceall[slice_number[2]]]

    print (threeslice)
    for i in range(len(threeslice)):
        idx_sa = slice_2D(v_sa_hat_es_cp, threeslice[i])
        idx_sa_gt = np.stack(np.nonzero(contour_sa_es[slice_number[i], :, :]), 1)

        mcd_sa, hd_sa = distance_metric(idx_sa, idx_sa_gt, 1.25)
        if (mcd_sa != None) and (hd_sa != None):
            mcd_sa_allslice.append(mcd_sa)
            hd_sa_allslice.append(hd_sa)


    mean_mcd_sa_allslices = np.mean(mcd_sa_allslice) if mcd_sa_allslice else None
    mean_hd_sa_allslices = np.mean(hd_sa_allslice) if hd_sa_allslice else None

    return mean_mcd_sa_allslices, mean_hd_sa_allslices



def FBoundary(pred_contour, gt_contour, bound_th=2):
    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(pred_contour.shape))

    pred_dil = binary_dilation(pred_contour, disk(bound_pix))
    gt_dil = binary_dilation(gt_contour, disk(bound_pix))

    # Get the intersection
    gt_match = gt_contour * pred_dil
    pred_match = pred_contour * gt_dil

    # Area of the intersection
    n_pred = np.sum(pred_contour)
    n_gt = np.sum(gt_contour)

    # % Compute precision and recall
    if n_pred == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_pred > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_pred == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(pred_match) / float(n_pred)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        Fscore = None
    else:
        Fscore = 2 * precision * recall / (precision + recall)

    return Fscore

def compute_sa_Fboundary(v_sa_hat_es_cp, contour_sa_es, sliceall, height, width):

    bfscore_all = []
    for i in range(len(sliceall)):
        idx_sa = slice_2D(v_sa_hat_es_cp, sliceall[i])
        sa_pred = np.zeros(shape=(height, width))
        for j in range(idx_sa.shape[0]):
            sa_pred[idx_sa[j,0], idx_sa[j,1]] = 1

        Fscore_1 = FBoundary(sa_pred, contour_sa_es[i,:,:], 1)
        Fscore_2 = FBoundary(sa_pred, contour_sa_es[i,:,:], 2)
        Fscore_3 = FBoundary(sa_pred, contour_sa_es[i,:,:], 3)
        Fscore_4 = FBoundary(sa_pred, contour_sa_es[i,:,:], 4)
        Fscore_5 = FBoundary(sa_pred, contour_sa_es[i,:,:], 5)


        if (Fscore_1 != None):
            Fscore = (Fscore_1+Fscore_2+Fscore_3+Fscore_4+Fscore_5)/5.0
            bfscore_all.append(Fscore)

    mean_bfscore = np.mean(bfscore_all) if bfscore_all else None


    return mean_bfscore

def compute_la_Fboundary(pred_contour, gt_contour):

    Fscore_1 = FBoundary(pred_contour, gt_contour, 1)
    Fscore_2 = FBoundary(pred_contour, gt_contour, 2)
    Fscore_3 = FBoundary(pred_contour, gt_contour, 3)
    Fscore_4 = FBoundary(pred_contour, gt_contour, 4)
    Fscore_5 = FBoundary(pred_contour, gt_contour, 5)


    if (Fscore_1 != None):
        Fscore = (Fscore_1+Fscore_2+Fscore_3+Fscore_4+Fscore_5)/5.0
    else:
        Fscore = None


    return Fscore




def projection_weightHD_loss_SA(v_sa_hat_ed_cp, temper, height, width, depth, gt_mesh2seg_sa, status):

    weightHD_loss = []

    for i in range(depth-1):
        v_sa_idx_ed, w_sa_ed = projection(v_sa_hat_ed_cp, i, temper)
        slice_loss = weightedHausdorff_batch(v_sa_idx_ed, w_sa_ed, gt_mesh2seg_sa[:,:,:,i], height, width, temper, status)

        weightHD_loss.append(slice_loss)

    weightHD_loss = torch.stack(weightHD_loss)

    loss_aver = torch.mean(weightHD_loss)



    return loss_aver



























