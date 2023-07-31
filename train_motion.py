import torch.nn as nn
import numpy as np
import itertools
import os
import sys

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import Meshes
from pytorch3d import loss

from network_motion import *
from dataio_motion import *
from utils import *


lr = 1e-4
n_worker = 4
bs = 8
n_epoch = 400
base_err = 1000

w_smooth = 150
w_reg = 20
w_h = 0.5
width = 128
height = 128
depth = 64
sa_idx = [12, 17, 22, 27, 32, 37, 42, 47, 52]
temper = 3



model_save_path = './models/model_motion'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# pytorch only saves the last model
Motion_save_path = os.path.join(model_save_path, 'motionEst.pth')
Motion_LA_save_path = os.path.join(model_save_path, 'multiview.pth')

flow_criterion = nn.MSELoss()
MotionNet = MotionMesh_25d().cuda()
MV_LA = Mesh_2d().cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                              itertools.chain(MotionNet.parameters(), MV_LA.parameters())), lr=lr)
Tensor = torch.cuda.FloatTensor
TensorLong = torch.cuda.LongTensor

# visualisation
writer = SummaryWriter('./runs/model_motion')



def train(epoch):
    MotionNet.train()
    MV_LA.train()

    epoch_loss = []
    epoch_seg_loss = []
    epoch_smooth_loss = []
    epoch_reg_loss = []
    epoch_huber_loss = []

    Myo_dice_sa = []
    Myo_dice_2ch = []
    Myo_dice_4ch = []


    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):

        img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, \
        contour_sa, contour_2ch, contour_4ch, \
        vertex_ed, faces, affine_inv, affine, origin = batch


        x_sa_t = Variable(img_sa_t.type(Tensor))
        x_sa_ed = Variable(img_sa_ed.type(Tensor))
        x_2ch_t = Variable(img_2ch_t.type(Tensor))
        x_2ch_ed = Variable(img_2ch_ed.type(Tensor))
        x_4ch_t = Variable(img_4ch_t.type(Tensor))
        x_4ch_ed = Variable(img_4ch_ed.type(Tensor))

        x_sa_t_5D = Variable(img_sa_t.unsqueeze(1).type(Tensor))
        x_sa_ed_5D = Variable(img_sa_ed.unsqueeze(1).type(Tensor))


        con_sa = Variable(contour_sa.type(TensorLong))  # [bs, slices, H, W]
        con_2ch = Variable(contour_2ch.type(TensorLong))  # [bs, H, W]
        con_4ch = Variable(contour_4ch.type(TensorLong))  # [bs, H, W]

        aff_sa_inv = Variable(affine_inv[:, 0,:,:].type(Tensor))
        aff_sa = Variable(affine[:, 0,:,:].type(Tensor))
        aff_2ch_inv = Variable(affine_inv[:, 1,:,:].type(Tensor))
        aff_4ch_inv = Variable(affine_inv[:, 2,:,:].type(Tensor))

        origin_sa = Variable(origin[:, 0:1, :].type(Tensor))
        origin_2ch = Variable(origin[:, 1:2, :].type(Tensor))
        origin_4ch = Variable(origin[:, 2:3, :].type(Tensor))

        vertex_0 = Variable(vertex_ed.permute(0,2,1).type(Tensor)) # [bs, 3, number of vertices]
        faces_0 = Variable(faces.type(Tensor)) # [bs, number of faces, 3]


        optimizer.zero_grad()

        net_la = MV_LA(x_2ch_t, x_2ch_ed, x_4ch_t, x_4ch_ed)
        net_sa = MotionNet(x_sa_t, x_sa_ed, net_la['conv2_2ch'], net_la['conv2s_2ch'], net_la['conv2_4ch'], net_la['conv2s_4ch'])

        # ---------------sample from 3D motion fields
        #translate coordinate
        v_ed_o = torch.matmul(aff_sa_inv[:, :3, :3], vertex_0) + aff_sa_inv[:, :3, 3:4]
        v_ed = v_ed_o.permute(0, 2, 1) - origin_sa  # [bs, number of vertices,3]

        # normalize translated coordinate (image space) to [-1,1]
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
        v_t_norm_expand = v_ed_norm_expand + delta_p  # [bs, 1, 1,number of vertices,3]
        # t frame
        v_t_norm = v_t_norm_expand.squeeze(1).squeeze(1)
        v_t_x = v_t_norm[:, :, 0:1] * (width / 2) + (width / 2)
        v_t_y = v_t_norm[:, :, 1:2] * (height / 2) + (height / 2)
        v_t_z = v_t_norm[:, :, 2:3] * (depth / 2) + (depth / 2)
        v_t_crop = torch.cat((v_t_x, v_t_y, v_t_z), 2)
        # translate back to mesh space
        v_t = v_t_crop + origin_sa  # [bs, number of vertices,3]
        pred_vertex_t = torch.matmul(aff_sa[:, :3, :3], v_t.permute(0,2,1)) + aff_sa[:, :3, 3:4] # [bs, 3, number of vertices]
        # print (pred_vertex_t.shape)


        pred_sa_ed = transform(x_sa_t_5D, net_sa['out'], mode='bilinear')

        # -------------- differentialable slicer

        # coordinate transformation np.dot(aff_sa_SR_inv[:3,:3], points_ED.T) + aff_sa_SR_inv[:3,3:4]
        v_sa_hat_t_o = torch.matmul(aff_sa_inv[:, :3, :3], pred_vertex_t) + aff_sa_inv[:, :3, 3:4]
        v_sa_hat_t = v_sa_hat_t_o.permute(0, 2, 1) - origin_sa
        # print (v_sa_hat_t.shape)
        v_2ch_hat_t_o = torch.matmul(aff_2ch_inv[:, :3, :3], pred_vertex_t) + aff_2ch_inv[:, :3, 3:4]
        v_2ch_hat_t = v_2ch_hat_t_o.permute(0, 2, 1) - origin_2ch
        v_4ch_hat_t_o = torch.matmul(aff_4ch_inv[:, :3, :3], pred_vertex_t) + aff_4ch_inv[:, :3, 3:4]
        v_4ch_hat_t = v_4ch_hat_t_o.permute(0,2, 1) - origin_4ch

        # project vertices satisfying threshood
        # project to SAX slices, project all vertices to a target plane,
        # vertices selection is moved to loss computation function
        v_sa_hat_t_x = torch.clamp(v_sa_hat_t[:, :, 0:1], min=0, max=height - 1)
        v_sa_hat_t_y = torch.clamp(v_sa_hat_t[:, :, 1:2], min=0, max=width - 1)
        v_sa_hat_t_cp = torch.cat((v_sa_hat_t_x, v_sa_hat_t_y, v_sa_hat_t[:, :, 2:3]), 2)

        v_sa_idx_t_0, w_sa_t_0 = projection(v_sa_hat_t_cp, 12, temper)
        # print (v_sa_idx_ed_0.shape, w_sa_ed_0.shape)
        v_sa_idx_t_1, w_sa_t_1 = projection(v_sa_hat_t_cp, 17, temper)
        v_sa_idx_t_2, w_sa_t_2 = projection(v_sa_hat_t_cp, 22, temper)
        v_sa_idx_t_3, w_sa_t_3 = projection(v_sa_hat_t_cp, 27, temper)
        v_sa_idx_t_4, w_sa_t_4 = projection(v_sa_hat_t_cp, 32, temper)
        v_sa_idx_t_5, w_sa_t_5 = projection(v_sa_hat_t_cp, 37, temper)
        v_sa_idx_t_6, w_sa_t_6 = projection(v_sa_hat_t_cp, 42, temper)
        v_sa_idx_t_7, w_sa_t_7 = projection(v_sa_hat_t_cp, 47, temper)
        v_sa_idx_t_8, w_sa_t_8 = projection(v_sa_hat_t_cp, 52, temper)

        # project to LAX 2CH view
        v_2ch_hat_t_x = torch.clamp(v_2ch_hat_t[:, :, 0:1], min=0, max=height - 1)
        v_2ch_hat_t_y = torch.clamp(v_2ch_hat_t[:, :, 1:2], min=0, max=width - 1)
        v_2ch_hat_t_cp = torch.cat((v_2ch_hat_t_x, v_2ch_hat_t_y, v_2ch_hat_t[:, :, 2:3]), 2)

        v_2ch_idx_t, w_2ch_t = projection(v_2ch_hat_t_cp, 0, temper)


        # project to LAX 4CH view
        v_4ch_hat_t_x = torch.clamp(v_4ch_hat_t[:, :, 0:1], min=0, max=height - 1)
        v_4ch_hat_t_y = torch.clamp(v_4ch_hat_t[:, :, 1:2], min=0, max=width - 1)
        v_4ch_hat_t_cp = torch.cat((v_4ch_hat_t_x, v_4ch_hat_t_y, v_4ch_hat_t[:, :, 2:3]), 2)

        v_4ch_idx_t, w_4ch_t = projection(v_4ch_hat_t_cp, 0, temper)



        # --------------------- Segmentation loss------------------
        loss_seg_sa_t_0 = weightedHausdorff_batch(v_sa_idx_t_0, w_sa_t_0, con_sa[:, 0, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_1 = weightedHausdorff_batch(v_sa_idx_t_1, w_sa_t_1, con_sa[:, 1, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_2 = weightedHausdorff_batch(v_sa_idx_t_2, w_sa_t_2, con_sa[:, 2, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_3 = weightedHausdorff_batch(v_sa_idx_t_3, w_sa_t_3, con_sa[:, 3, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_4 = weightedHausdorff_batch(v_sa_idx_t_4, w_sa_t_4, con_sa[:, 4, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_5 = weightedHausdorff_batch(v_sa_idx_t_5, w_sa_t_5, con_sa[:, 5, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_6 = weightedHausdorff_batch(v_sa_idx_t_6, w_sa_t_6, con_sa[:, 6, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_7 = weightedHausdorff_batch(v_sa_idx_t_7, w_sa_t_7, con_sa[:, 7, :, :], height, width, temper,
                                                  'train')
        loss_seg_sa_t_8 = weightedHausdorff_batch(v_sa_idx_t_8, w_sa_t_8, con_sa[:, 8, :, :], height, width, temper,
                                                  'train')
        loss_seg_2ch_t = weightedHausdorff_batch(v_2ch_idx_t, w_2ch_t, con_2ch, height, width, temper, 'train')
        loss_seg_4ch_t = weightedHausdorff_batch(v_4ch_idx_t, w_4ch_t, con_4ch, height, width, temper, 'train')

        loss_seg = (loss_seg_sa_t_0 + loss_seg_sa_t_1 + loss_seg_sa_t_2 + loss_seg_sa_t_3 +
                      loss_seg_sa_t_4 + loss_seg_sa_t_5 + loss_seg_sa_t_6 + loss_seg_sa_t_7 + loss_seg_sa_t_8) / 9.0 + \
                     loss_seg_2ch_t + loss_seg_4ch_t




        #----------------smoothness loss------------
        # print (pred_vertex_t.permute(0,2,1).shape)
        trg_mesh = Meshes(verts=list(pred_vertex_t.permute(0, 2, 1)), faces=list(faces_0))
        loss_smooth = loss.mesh_laplacian_smoothing(trg_mesh, method='uniform')

        # ----------------regularization loss------------

        # define image registration as a regularization term
        loss_reg = flow_criterion(pred_sa_ed, x_sa_ed_5D)

        loss_huber = huber_loss_3d(net_sa['out'])


        loss_all = loss_seg + w_reg*loss_reg + w_smooth * loss_smooth + w_h * loss_huber

        loss_all.backward()
        optimizer.step()


        epoch_loss.append(loss_all.item())
        epoch_seg_loss.append(loss_seg.item())
        epoch_smooth_loss.append(loss_smooth.item())
        epoch_reg_loss.append(loss_reg.item())
        epoch_huber_loss.append(loss_huber.item())



        # tensorboard visulisation
        writer.add_scalar("Loss/train", loss_all, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_seg", loss_seg, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_reg", loss_reg, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_smooth", loss_smooth, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_huber", loss_huber, epoch * len(training_data_loader) + batch_idx)


        if batch_idx % 40 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss all: {:.6f}, '
                  'Seg Loss: {:.6f}, Reg Loss: {:.6f}, Smooth Loss: {:.6f}, Huber Loss: {:.6f}'.format(
                epoch, batch_idx * len(img_sa_t), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), np.mean(epoch_loss),
                np.mean(epoch_seg_loss), np.mean(epoch_reg_loss), np.mean(epoch_smooth_loss), np.mean(epoch_huber_loss), np.mean(Myo_dice_sa), np.mean(Myo_dice_2ch), np.mean(Myo_dice_4ch)))

            # torch.save(model.state_dict(), model_save_path)
            # print("Checkpoint saved to {}".format(model_save_path))

def val(epoch):
    MotionNet.eval()
    MV_LA.eval()

    val_loss = []
    val_seg_loss = []
    val_smooth_loss = []
    val_reg_loss = []
    val_huber_loss = []

    global base_err
    for batch_idx, batch in tqdm(enumerate(val_data_loader, 1),
                                 total=len(val_data_loader)):

        img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, \
        contour_sa, contour_2ch, contour_4ch, \
        vertex_ed, faces, affine_inv, affine, origin = batch

        with torch.no_grad():

            x_sa_t = img_sa_t.type(Tensor)
            x_sa_ed = img_sa_ed.type(Tensor)
            x_2ch_t = img_2ch_t.type(Tensor)
            x_2ch_ed = img_2ch_ed.type(Tensor)
            x_4ch_t = img_4ch_t.type(Tensor)
            x_4ch_ed = img_4ch_ed.type(Tensor)

            x_sa_t_5D = img_sa_t.unsqueeze(1).type(Tensor)
            x_sa_ed_5D = img_sa_ed.unsqueeze(1).type(Tensor)


            con_sa = contour_sa.type(TensorLong)  # [bs, slices, H, W]
            con_2ch = contour_2ch.type(TensorLong)  # [bs, H, W]
            con_4ch = contour_4ch.type(TensorLong)  # [bs, H, W]

            aff_sa_inv = affine_inv[:, 0, :, :].type(Tensor)
            aff_sa = affine[:, 0, :, :].type(Tensor)
            aff_2ch_inv = affine_inv[:, 1, :, :].type(Tensor)
            aff_4ch_inv = affine_inv[:, 2, :, :].type(Tensor)

            origin_sa = origin[:, 0:1, :].type(Tensor)
            origin_2ch = origin[:, 1:2, :].type(Tensor)
            origin_4ch = origin[:, 2:3, :].type(Tensor)

            vertex_0 = vertex_ed.permute(0, 2, 1).type(Tensor)  # [bs, 3, number of vertices]
            faces_0 = faces.type(Tensor)  # [bs, number of faces, 3]


            net_la = MV_LA(x_2ch_t, x_2ch_ed, x_4ch_t, x_4ch_ed)
            net_sa = MotionNet(x_sa_t, x_sa_ed, net_la['conv2_2ch'], net_la['conv2s_2ch'], net_la['conv2_4ch'],
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
            v_t_norm_expand = v_ed_norm_expand + delta_p  # [bs, 1, 1,number of vertices,3]
            # t frame
            v_t_norm = v_t_norm_expand.squeeze(1).squeeze(1)
            v_t_x = v_t_norm[:, :, 0:1] * (width / 2) + (width / 2)
            v_t_y = v_t_norm[:, :, 1:2] * (height / 2) + (height / 2)
            v_t_z = v_t_norm[:, :, 2:3] * (depth / 2) + (depth / 2)
            v_t_crop = torch.cat((v_t_x, v_t_y, v_t_z), 2)
            # translate back to mesh space
            v_t = v_t_crop + origin_sa  # [bs, number of vertices,3]
            pred_vertex_t = torch.matmul(aff_sa[:, :3, :3], v_t.permute(0, 2, 1)) + aff_sa[:, :3,
                                                                                    3:4]  # [bs, 3, number of vertices]
            # print (pred_vertex_t.shape)


            pred_sa_ed = transform(x_sa_t_5D, net_sa['out'], mode='bilinear')

            # -------------- differentialable slicer

            # coordinate transformation np.dot(aff_sa_SR_inv[:3,:3], points_ED.T) + aff_sa_SR_inv[:3,3:4]
            v_sa_hat_t_o = torch.matmul(aff_sa_inv[:, :3, :3], pred_vertex_t) + aff_sa_inv[:, :3, 3:4]
            v_sa_hat_t = v_sa_hat_t_o.permute(0, 2, 1) - origin_sa
            # print (v_sa_hat_t.shape)
            v_2ch_hat_t_o = torch.matmul(aff_2ch_inv[:, :3, :3], pred_vertex_t) + aff_2ch_inv[:, :3, 3:4]
            v_2ch_hat_t = v_2ch_hat_t_o.permute(0, 2, 1) - origin_2ch
            v_4ch_hat_t_o = torch.matmul(aff_4ch_inv[:, :3, :3], pred_vertex_t) + aff_4ch_inv[:, :3, 3:4]
            v_4ch_hat_t = v_4ch_hat_t_o.permute(0, 2, 1) - origin_4ch

            # project vertices satisfying threshood
            # project to SAX slices, project all vertices to a target plane,
            # vertices selection is moved to loss computation function
            v_sa_hat_t_x = torch.clamp(v_sa_hat_t[:, :, 0:1], min=0, max=height - 1)
            v_sa_hat_t_y = torch.clamp(v_sa_hat_t[:, :, 1:2], min=0, max=width - 1)
            v_sa_hat_t_cp = torch.cat((v_sa_hat_t_x, v_sa_hat_t_y, v_sa_hat_t[:, :, 2:3]), 2)

            v_sa_idx_t_0, w_sa_t_0 = projection(v_sa_hat_t_cp, 12, temper)
            # print (v_sa_idx_ed_0.shape, w_sa_ed_0.shape)
            v_sa_idx_t_1, w_sa_t_1 = projection(v_sa_hat_t_cp, 17, temper)
            v_sa_idx_t_2, w_sa_t_2 = projection(v_sa_hat_t_cp, 22, temper)
            v_sa_idx_t_3, w_sa_t_3 = projection(v_sa_hat_t_cp, 27, temper)
            v_sa_idx_t_4, w_sa_t_4 = projection(v_sa_hat_t_cp, 32, temper)
            v_sa_idx_t_5, w_sa_t_5 = projection(v_sa_hat_t_cp, 37, temper)
            v_sa_idx_t_6, w_sa_t_6 = projection(v_sa_hat_t_cp, 42, temper)
            v_sa_idx_t_7, w_sa_t_7 = projection(v_sa_hat_t_cp, 47, temper)
            v_sa_idx_t_8, w_sa_t_8 = projection(v_sa_hat_t_cp, 52, temper)

            # project to LAX 2CH view
            v_2ch_hat_t_x = torch.clamp(v_2ch_hat_t[:, :, 0:1], min=0, max=height - 1)
            v_2ch_hat_t_y = torch.clamp(v_2ch_hat_t[:, :, 1:2], min=0, max=width - 1)
            v_2ch_hat_t_cp = torch.cat((v_2ch_hat_t_x, v_2ch_hat_t_y, v_2ch_hat_t[:, :, 2:3]), 2)

            v_2ch_idx_t, w_2ch_t = projection(v_2ch_hat_t_cp, 0, temper)

            # project to LAX 4CH view
            v_4ch_hat_t_x = torch.clamp(v_4ch_hat_t[:, :, 0:1], min=0, max=height - 1)
            v_4ch_hat_t_y = torch.clamp(v_4ch_hat_t[:, :, 1:2], min=0, max=width - 1)
            v_4ch_hat_t_cp = torch.cat((v_4ch_hat_t_x, v_4ch_hat_t_y, v_4ch_hat_t[:, :, 2:3]), 2)

            v_4ch_idx_t, w_4ch_t = projection(v_4ch_hat_t_cp, 0, temper)

            # --------------------- Segmentation loss------------------
            loss_seg_sa_t_0 = weightedHausdorff_batch(v_sa_idx_t_0, w_sa_t_0, con_sa[:, 0, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_1 = weightedHausdorff_batch(v_sa_idx_t_1, w_sa_t_1, con_sa[:, 1, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_2 = weightedHausdorff_batch(v_sa_idx_t_2, w_sa_t_2, con_sa[:, 2, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_3 = weightedHausdorff_batch(v_sa_idx_t_3, w_sa_t_3, con_sa[:, 3, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_4 = weightedHausdorff_batch(v_sa_idx_t_4, w_sa_t_4, con_sa[:, 4, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_5 = weightedHausdorff_batch(v_sa_idx_t_5, w_sa_t_5, con_sa[:, 5, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_6 = weightedHausdorff_batch(v_sa_idx_t_6, w_sa_t_6, con_sa[:, 6, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_7 = weightedHausdorff_batch(v_sa_idx_t_7, w_sa_t_7, con_sa[:, 7, :, :], height, width, temper,
                                                      'val')
            loss_seg_sa_t_8 = weightedHausdorff_batch(v_sa_idx_t_8, w_sa_t_8, con_sa[:, 8, :, :], height, width, temper,
                                                      'val')
            loss_seg_2ch_t = weightedHausdorff_batch(v_2ch_idx_t, w_2ch_t, con_2ch, height, width, temper, 'val')
            loss_seg_4ch_t = weightedHausdorff_batch(v_4ch_idx_t, w_4ch_t, con_4ch, height, width, temper, 'val')

            loss_seg = (loss_seg_sa_t_0 + loss_seg_sa_t_1 + loss_seg_sa_t_2 + loss_seg_sa_t_3 +
                        loss_seg_sa_t_4 + loss_seg_sa_t_5 + loss_seg_sa_t_6 + loss_seg_sa_t_7 + loss_seg_sa_t_8) / 9.0 + \
                       loss_seg_2ch_t + loss_seg_4ch_t

            # ----------------smoothness loss------------
            # print (pred_vertex_t.permute(0,2,1).shape)
            trg_mesh = Meshes(verts=list(pred_vertex_t.permute(0, 2, 1)), faces=list(faces_0))
            loss_smooth = loss.mesh_laplacian_smoothing(trg_mesh, method='uniform')

            # ----------------regularization loss------------

            loss_reg = flow_criterion(pred_sa_ed, x_sa_ed_5D)

            loss_huber = huber_loss_3d(net_sa['out'])

            loss_all = loss_seg + w_reg * loss_reg + w_smooth * loss_smooth + w_h * loss_huber


            val_loss.append(loss_all.item())
            val_seg_loss.append(loss_seg.item())
            val_smooth_loss.append(loss_smooth.item())
            val_reg_loss.append(loss_reg.item())
            val_huber_loss.append(loss_huber.item())

            if batch_idx == 1:
                # tensorboard visulisation
                writer.add_scalar("Loss/val", loss_all, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_seg", loss_seg, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_reg", loss_reg, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_smooth", loss_smooth, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_huber", loss_huber, epoch * len(training_data_loader) + batch_idx)


    if np.mean(val_loss) < base_err:
        torch.save(MotionNet.state_dict(), Motion_save_path)
        torch.save(MV_LA.state_dict(), Motion_LA_save_path)
        base_err = np.mean(val_loss)



data_path = '/train_data_path'
train_set = TrainDataset(data_path)
# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker, batch_size=bs, shuffle=True)

val_data_path = '/val_data_pathl'
val_set = ValDataset(val_data_path)
val_data_loader = DataLoader(dataset=val_set, num_workers=n_worker, batch_size=bs, shuffle=False)


for epoch in range(0, n_epoch + 1):
    start = time.time()
    train(epoch)
    end = time.time()
    print("training took {:.8f}".format(end-start))

    print('Epoch {}'.format(epoch))
    start = time.time()
    val(epoch)
    end = time.time()
    print("validation took {:.8f}".format(end - start))
