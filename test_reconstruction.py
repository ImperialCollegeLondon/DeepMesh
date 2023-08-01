import torch.nn as nn
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pdb
import imageio
import os
import sys
import nibabel as nib
import neural_renderer as nr
import pyvista as pv

from network_reconstruction import *
from dataio_reconstruction import *
from utils import *
import vtk
import scipy.io
import csv
import pdb

n_class = 4
n_worker = 4
bs = 1
T_num = 50 # number of frames
width = 128
height = 128
depth = 64
temper = 2
sa_sliceall = [12, 17, 22, 27, 32, 37, 42, 47, 52]


model_save_path = './models/model_reconstruction'
# pytorch only saves the last model
Deform_save_path = os.path.join(model_save_path, 'deform.pth')
Motion_LA_save_path = os.path.join(model_save_path, 'multiview.pth')

def test(sub_path):
    DeformNet.eval()
    MV_LA.eval()

    hd_SA = []
    hd_2CH = []
    hd_4CH = []

    bfscore_SA = []
    bfscore_2CH = []
    bfscore_4CH = []


    for name in glob.glob(os.path.join(sub_path, '*')):

        sub_name = name.split('/')[-1]
        print (sub_name)

        image_sa_bank, image_2ch_bank, image_4ch_bank, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch = load_data(
        sub_path, sub_name, T_num, rand_frame=0)

        img_sa_ed = torch.from_numpy(image_sa_bank[1:2, :, :, :])
        img_2ch_t = torch.from_numpy(image_2ch_bank[0:1, :, :, :])
        img_2ch_ed = torch.from_numpy(image_2ch_bank[1:2, :, :, :])
        img_4ch_t = torch.from_numpy(image_4ch_bank[0:1, :, :, :])
        img_4ch_ed = torch.from_numpy(image_4ch_bank[1:2, :, :, :])

        with torch.no_grad():

            x_sa_ed = img_sa_ed.type(Tensor)
            x_2ch_t = img_2ch_t.type(Tensor)
            x_2ch_ed = img_2ch_ed.type(Tensor)
            x_4ch_t = img_4ch_t.type(Tensor)
            x_4ch_ed = img_4ch_ed.type(Tensor)


            aff_sa_inv = torch.from_numpy(affine_inv[0, :, :]).type(Tensor).unsqueeze(0)
            aff_sa = torch.from_numpy(affine[0, :, :]).type(Tensor).unsqueeze(0)
            aff_2ch_inv = torch.from_numpy(affine_inv[1, :, :]).type(Tensor).unsqueeze(0)
            aff_4ch_inv = torch.from_numpy(affine_inv[2, :, :]).type(Tensor).unsqueeze(0)

            origin_sa = torch.from_numpy(origin[0:1, :]).type(Tensor)
            origin_2ch = torch.from_numpy(origin[1:2, :]).type(Tensor)
            origin_4ch = torch.from_numpy(origin[2:3, :]).type(Tensor)

            vertex_tpl_0 = torch.from_numpy(vertex_tpl_ed).unsqueeze(0).permute(0, 2, 1).type(Tensor)  # [bs, 3, number of vertices]



            net_la = MV_LA(x_2ch_t, x_2ch_ed, x_4ch_t, x_4ch_ed)
            net_df = DeformNet(x_sa_ed, net_la['conv2s_2ch'], net_la['conv2s_4ch'])

            # ---------------sample from 3D motion fields
            # translate coordinate
            v_ed_o = torch.matmul(aff_sa_inv[:, :3, :3], vertex_tpl_0) + aff_sa_inv[:, :3, 3:4]
            v_ed = v_ed_o.permute(0, 2, 1) - origin_sa  # [bs, number of vertices,3]
            # normalize translated coordinate (image space) to [-1,1]
            v_ed_x = (v_ed[:, :, 0:1] - (width / 2)) / (width / 2)
            v_ed_y = (v_ed[:, :, 1:2] - (height / 2)) / (height / 2)
            v_ed_z = (v_ed[:, :, 2:3] - (depth / 2)) / (depth / 2)
            v_ed_norm = torch.cat((v_ed_x, v_ed_y, v_ed_z), 2)
            v_ed_norm_expand = v_ed_norm.unsqueeze(1).unsqueeze(1)  # [bs, 1, 1,number of vertices,3]

            # sample from 3D motion field
            pxx = F.grid_sample(net_df['out_def_ed'][:, 0:1], v_ed_norm_expand, align_corners=True).transpose(4, 3)
            pyy = F.grid_sample(net_df['out_def_ed'][:, 1:2], v_ed_norm_expand, align_corners=True).transpose(4, 3)
            pzz = F.grid_sample(net_df['out_def_ed'][:, 2:3], v_ed_norm_expand, align_corners=True).transpose(4, 3)
            delta_p = torch.cat((pxx, pyy, pzz), 4)
            # updata coor (image space)
            # print (v_ed.shape, delta_p.shape)
            v_0_norm_expand = v_ed_norm_expand + delta_p  # [bs, 1, 1,number of vertices,3]
            # t frame
            v_0_norm = v_0_norm_expand.squeeze(1).squeeze(1)
            v_0_x = v_0_norm[:, :, 0:1] * (width / 2) + (width / 2)
            v_0_y = v_0_norm[:, :, 1:2] * (height / 2) + (height / 2)
            v_0_z = v_0_norm[:, :, 2:3] * (depth / 2) + (depth / 2)
            v_0_crop = torch.cat((v_0_x, v_0_y, v_0_z), 2)
            # translate back to mesh space
            v_0 = v_0_crop + origin_sa  # [bs, number of vertices,3]
            pred_v_0 = torch.matmul(aff_sa[:, :3, :3], v_0.permute(0, 2, 1)) + aff_sa[:, :3,
                                                                               3:4]  # [bs, 3, number of vertices]

            # -------------- differentialable slicer

            # coordinate transformation np.dot(aff_sa_SR_inv[:3,:3], points_ED.T) + aff_sa_SR_inv[:3,3:4]
            v_sa_hat_ed_o = torch.matmul(aff_sa_inv[:, :3, :3], pred_v_0) + aff_sa_inv[:, :3, 3:4]
            v_sa_hat_ed = v_sa_hat_ed_o.permute(0, 2, 1) - origin_sa
            # print (v_sa_hat_t.shape)
            v_2ch_hat_ed_o = torch.matmul(aff_2ch_inv[:, :3, :3], pred_v_0) + aff_2ch_inv[:, :3, 3:4]
            v_2ch_hat_ed = v_2ch_hat_ed_o.permute(0, 2, 1) - origin_2ch
            v_4ch_hat_ed_o = torch.matmul(aff_4ch_inv[:, :3, :3], pred_v_0) + aff_4ch_inv[:, :3, 3:4]
            v_4ch_hat_ed = v_4ch_hat_ed_o.permute(0, 2, 1) - origin_4ch

            # project vertices satisfying threshood
            # project to SAX slices, project all vertices to a target plane,
            # vertices selection is moved to loss computation function
            v_sa_hat_ed_x = torch.clamp(v_sa_hat_ed[:, :, 0:1], min=0, max=height - 1)
            v_sa_hat_ed_y = torch.clamp(v_sa_hat_ed[:, :, 1:2], min=0, max=width - 1)
            v_sa_hat_ed_cp = torch.cat((v_sa_hat_ed_x, v_sa_hat_ed_y, v_sa_hat_ed[:, :, 2:3]), 2)


            # project to LAX 2CH view
            v_2ch_hat_ed_x = torch.clamp(v_2ch_hat_ed[:, :, 0:1], min=0, max=height - 1)
            v_2ch_hat_ed_y = torch.clamp(v_2ch_hat_ed[:, :, 1:2], min=0, max=width - 1)
            v_2ch_hat_ed_cp = torch.cat((v_2ch_hat_ed_x, v_2ch_hat_ed_y, v_2ch_hat_ed[:, :, 2:3]), 2)


            # project to LAX 4CH view
            v_4ch_hat_ed_x = torch.clamp(v_4ch_hat_ed[:, :, 0:1], min=0, max=height - 1)
            v_4ch_hat_ed_y = torch.clamp(v_4ch_hat_ed[:, :, 1:2], min=0, max=width - 1)
            v_4ch_hat_ed_cp = torch.cat((v_4ch_hat_ed_x, v_4ch_hat_ed_y, v_4ch_hat_ed[:, :, 2:3]), 2)



            # slicer

            mcd_sa, hd_sa = compute_sa_mcd_hd(v_sa_hat_ed_cp, contour_sa_ed, sa_sliceall)
            bfscore_sa = compute_sa_Fboundary(v_sa_hat_ed_cp, contour_sa_ed, sa_sliceall, height, width)


            idx_2ch = slice_2D(v_2ch_hat_ed_cp, 0)
            idx_2ch_gt = np.stack(np.nonzero(contour_2ch_ed), 1)
            mcd_2ch, hd_2ch = distance_metric(idx_2ch, idx_2ch_gt, 1.25)
            la_2ch_pred_con = np.zeros(shape=(height, width), dtype=np.uint8)
            for j in range(idx_2ch.shape[0]):
                la_2ch_pred_con[idx_2ch[j, 0], idx_2ch[j, 1]] = 1
            bfscore_2ch = compute_la_Fboundary(la_2ch_pred_con, contour_2ch_ed)


            idx_4ch = slice_2D(v_4ch_hat_ed_cp, 0)
            idx_4ch_gt = np.stack(np.nonzero(contour_4ch_ed), 1)
            mcd_4ch, hd_4ch = distance_metric(idx_4ch, idx_4ch_gt, 1.25)
            la_4ch_pred_con = np.zeros(shape=(height, width), dtype=np.uint8)
            for j in range(idx_4ch.shape[0]):
                la_4ch_pred_con[idx_4ch[j, 0], idx_4ch[j, 1]] = 1
            bfscore_4ch = compute_la_Fboundary(la_4ch_pred_con, contour_4ch_ed)


            if (hd_sa != None):
                hd_SA.append(hd_sa)
            if (hd_2ch != None):
                hd_2CH.append(hd_2ch)
            if (hd_4ch != None):
                hd_4CH.append(hd_4ch)

            if (bfscore_sa != None):
                bfscore_SA.append(bfscore_sa)
            if (bfscore_2ch != None):
                bfscore_2CH.append(bfscore_2ch)
            if (bfscore_4ch != None):
                bfscore_4CH.append(bfscore_4ch)


            print (hd_sa, hd_2ch, hd_4ch)
            print (bfscore_sa, bfscore_2ch, bfscore_4ch)



    print('SA HD: {:.4f}({:.4f}), 2CH HD: {:.4f}({:.4f}), 4CH HD: {:.4f}({:.4f})'
          .format(np.mean(hd_SA), np.std(hd_SA), np.mean(hd_2CH), np.std(hd_2CH), np.mean(hd_4CH), np.std(hd_4CH)))
    print('SA BFscore: {:.4f}({:.4f}), 2CH BFscore: {:.4f}({:.4f}), 4CH BFscore: {:.4f}({:.4f})'
          .format(np.mean(bfscore_SA), np.std(bfscore_SA), np.mean(bfscore_2CH), np.std(bfscore_2CH),
                  np.mean(bfscore_4CH), np.std(bfscore_4CH)))




test_data_path = '/test_data_path'


DeformNet = deformnet().cuda()
MV_LA = Mesh_2d().cuda()

DeformNet.load_state_dict(torch.load(Deform_save_path), strict=True)
MV_LA.load_state_dict(torch.load(Motion_LA_save_path), strict=True)

Tensor = torch.cuda.FloatTensor
TensorLong = torch.cuda.LongTensor


start = time.time()
test(test_data_path)
end = time.time()
print("testing took {:.8f}".format(end - start))
