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

from network_motion import *
from dataio_motion import *
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
temper = 3
sa_sliceall = [12, 17, 22, 27, 32, 37, 42, 47, 52]


model_save_path = './models/model_motion'
# pytorch only saves the last model
Motion_save_path = os.path.join(model_save_path, 'motionEst.pth')
Motion_LA_save_path = os.path.join(model_save_path, 'multiview.pth')



def test_all(sub_path):
    MotionNet.eval()
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

        image_sa_ES_bank, image_2ch_ES_bank, image_4ch_ES_bank, \
        contour_sa_es, contour_2ch_es, contour_4ch_es, vertex_ed, faces, affine_inv, affine, origin = load_data_ES(
            sub_path, sub_name)

        img_sa_es = torch.from_numpy(image_sa_ES_bank[0:1, :, :, :])
        img_sa_ed = torch.from_numpy(image_sa_ES_bank[1:2, :, :, :])
        img_2ch_es = torch.from_numpy(image_2ch_ES_bank[0:1, :, :, :])
        img_2ch_ed = torch.from_numpy(image_2ch_ES_bank[1:2, :, :, :])
        img_4ch_es = torch.from_numpy(image_4ch_ES_bank[0:1, :, :, :])
        img_4ch_ed = torch.from_numpy(image_4ch_ES_bank[1:2, :, :, :])



        with torch.no_grad():
            x_sa_es = img_sa_es.type(Tensor)
            x_sa_ed = img_sa_ed.type(Tensor)
            x_2ch_es = img_2ch_es.type(Tensor)
            x_2ch_ed = img_2ch_ed.type(Tensor)
            x_4ch_es = img_4ch_es.type(Tensor)
            x_4ch_ed = img_4ch_ed.type(Tensor)



            aff_sa_inv = torch.from_numpy(affine_inv[0, :, :]).type(Tensor).unsqueeze(0)
            aff_sa = torch.from_numpy(affine[0, :, :]).type(Tensor).unsqueeze(0)
            aff_2ch_inv = torch.from_numpy(affine_inv[1, :, :]).type(Tensor).unsqueeze(0)
            aff_4ch_inv = torch.from_numpy(affine_inv[2, :, :]).type(Tensor).unsqueeze(0)

            origin_sa = torch.from_numpy(origin[0, :]).type(Tensor).unsqueeze(0)
            origin_2ch = torch.from_numpy(origin[1, :]).type(Tensor).unsqueeze(0)
            origin_4ch = torch.from_numpy(origin[2, :]).type(Tensor).unsqueeze(0)

            vertex_0 = torch.from_numpy(vertex_ed).unsqueeze(0).permute(0, 2, 1).type(
                Tensor)  # [bs, 3, number of vertices]

            net_la = MV_LA(x_2ch_es, x_2ch_ed, x_4ch_es, x_4ch_ed)
            net_sa = MotionNet(x_sa_es, x_sa_ed, net_la['conv2_2ch'], net_la['conv2s_2ch'], net_la['conv2_4ch'],
                               net_la['conv2s_4ch'])
            # ---------------sample from 3D motion fields
            # translate coordinate
            v_ed_o = torch.matmul(aff_sa_inv[:, :3, :3], vertex_0) + aff_sa_inv[:, :3, 3:4]
            v_ed = v_ed_o.permute(0, 2, 1) - origin_sa  # [bs, number of vertices,3]
            v_ed_x = (v_ed[:, :, 0:1] - (width / 2)) / (width / 2)
            v_ed_y = (v_ed[:, :, 1:2] - (height / 2)) / (height / 2)
            v_ed_z = (v_ed[:, :, 2:3] - (depth / 2)) / (depth / 2)
            v_ed_norm = torch.cat((v_ed_x, v_ed_y, v_ed_z), 2)
            v_ed_norm_expand = v_ed_norm.unsqueeze(1).unsqueeze(1)  # [bs, 1, 1,number of vertices,3]

            # sample from 3D motion field
            pxx = F.grid_sample(net_sa['out'][:, 0:1], v_ed_norm_expand, align_corners=True).transpose(4, 3)
            pyy = F.grid_sample(net_sa['out'][:, 1:2], v_ed_norm_expand, align_corners=True).transpose(4, 3)
            pzz = F.grid_sample(net_sa['out'][:, 2:3], v_ed_norm_expand, align_corners=True).transpose(4, 3)
            delta_p = torch.cat((pxx, pyy, pzz), 4)
            # updata coor (image space)
            # print (v_ed.shape, delta_p.shape)
            v_es_norm_expand = v_ed_norm_expand + delta_p  # [bs, 1, 1,number of vertices,3]
            # t frame
            v_es_norm = v_es_norm_expand.squeeze(1).squeeze(1)
            v_es_x = v_es_norm[:, :, 0:1] * (width / 2) + (width / 2)
            v_es_y = v_es_norm[:, :, 1:2] * (height / 2) + (height / 2)
            v_es_z = v_es_norm[:, :, 2:3] * (depth / 2) + (depth / 2)
            v_es_crop = torch.cat((v_es_x, v_es_y, v_es_z), 2)
            # translate back to mesh space
            v_es = v_es_crop + origin_sa  # [bs, number of vertices,3]
            pred_vertex_es = torch.matmul(aff_sa[:, :3, :3], v_es.permute(0, 2, 1)) + aff_sa[:, :3,
                                                                                    3:4]  # [bs, 3, number of vertices]



            # --------------------compute segmentation evalutation

            # slicer

            # coordinate transformation np.dot(aff_sa_SR_inv[:3,:3], points_ED.T) + aff_sa_SR_inv[:3,3:4]
            v_sa_hat_es_o = torch.matmul(aff_sa_inv[:, :3, :3], pred_vertex_es) + aff_sa_inv[:, :3, 3:4]
            v_sa_hat_es = v_sa_hat_es_o.permute(0, 2, 1) - origin_sa
            # print (v_sa_hat_es[0,:,2])
            # print (v_sa_hat_t.shape)
            v_2ch_hat_es_o = torch.matmul(aff_2ch_inv[:, :3, :3], pred_vertex_es) + aff_2ch_inv[:, :3, 3:4]
            v_2ch_hat_es = v_2ch_hat_es_o.permute(0, 2, 1) - origin_2ch
            v_4ch_hat_es_o = torch.matmul(aff_4ch_inv[:, :3, :3], pred_vertex_es) + aff_4ch_inv[:, :3, 3:4]
            v_4ch_hat_es = v_4ch_hat_es_o.permute(0, 2, 1) - origin_4ch

            # project vertices satisfying threshood
            # project to SAX slices, project all vertices to a target plane,
            # vertices selection is moved to loss computation function
            v_sa_hat_es_x = torch.clamp(v_sa_hat_es[:, :, 0:1], min=0, max=height - 1)
            v_sa_hat_es_y = torch.clamp(v_sa_hat_es[:, :, 1:2], min=0, max=width - 1)
            v_sa_hat_es_cp = torch.cat((v_sa_hat_es_x, v_sa_hat_es_y, v_sa_hat_es[:, :, 2:3]), 2)



            mcd_sa, hd_sa = compute_sa_mcd_hd(v_sa_hat_es_cp, contour_sa_es, sa_sliceall)
            bfscore_sa = compute_sa_Fboundary(v_sa_hat_es_cp, contour_sa_es, sa_sliceall, height, width)


            # project to LAX 2CH view
            v_2ch_hat_es_x = torch.clamp(v_2ch_hat_es[:, :, 0:1], min=0, max=height - 1)
            v_2ch_hat_es_y = torch.clamp(v_2ch_hat_es[:, :, 1:2], min=0, max=width - 1)
            v_2ch_hat_es_cp = torch.cat((v_2ch_hat_es_x, v_2ch_hat_es_y, v_2ch_hat_es[:, :, 2:3]), 2)

            idx_2ch = slice_2D(v_2ch_hat_es_cp, 0)
            idx_2ch_gt = np.stack(np.nonzero(contour_2ch_es), 1)
            mcd_2ch, hd_2ch = distance_metric(idx_2ch, idx_2ch_gt, 1.25)

            la_2ch_pred_con = np.zeros(shape=(height, width), dtype=np.uint8)
            for j in range(idx_2ch.shape[0]):
                la_2ch_pred_con[idx_2ch[j,0], idx_2ch[j,1]] = 1

            bfscore_2ch = compute_la_Fboundary(la_2ch_pred_con, contour_2ch_es)

            # project to LAX 4CH view
            v_4ch_hat_es_x = torch.clamp(v_4ch_hat_es[:, :, 0:1], min=0, max=height - 1)
            v_4ch_hat_es_y = torch.clamp(v_4ch_hat_es[:, :, 1:2], min=0, max=width - 1)
            v_4ch_hat_es_cp = torch.cat((v_4ch_hat_es_x, v_4ch_hat_es_y, v_4ch_hat_es[:, :, 2:3]), 2)


            idx_4ch = slice_2D(v_4ch_hat_es_cp, 0)
            idx_4ch_gt = np.stack(np.nonzero(contour_4ch_es), 1)
            mcd_4ch, hd_4ch = distance_metric(idx_4ch, idx_4ch_gt, 1.25)
            la_4ch_pred_con = np.zeros(shape=(height, width), dtype=np.uint8)
            for j in range(idx_4ch.shape[0]):
                la_4ch_pred_con[idx_4ch[j,0], idx_4ch[j,1]] = 1

            bfscore_4ch = compute_la_Fboundary(la_4ch_pred_con, contour_4ch_es)


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
          .format(np.mean(bfscore_SA), np.std(bfscore_SA), np.mean(bfscore_2CH), np.std(bfscore_2CH), np.mean(bfscore_4CH), np.std(bfscore_4CH)))


    return



test_data_path = '/vol/bitbucket/qm216/UKBBcardiac/cardiacdata/ukbbSALA_CMR/test'

test_set = TestDataset(test_data_path)
testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker, batch_size=bs, shuffle=False)

MotionNet = MotionMesh_25d().cuda()
MV_LA = Mesh_2d().cuda()

MotionNet.load_state_dict(torch.load(Motion_save_path), strict=True)
MV_LA.load_state_dict(torch.load(Motion_LA_save_path), strict=True)

Tensor = torch.cuda.FloatTensor
TensorLong = torch.cuda.LongTensor



start = time.time()
test_all(test_data_path)
end = time.time()
print("testing took {:.8f}".format(end - start))