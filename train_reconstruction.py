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

from network_reconstruction import *
from dataio_reconstruction import *
from utils import *


lr = 1e-4
n_worker = 4
bs = 5
n_epoch = 400
base_err = 10000

w_smooth = 20
w_surface = 0.5
w_h = 0.5
width = 128
height = 128
depth = 64
temper = 3



model_save_path = './models/model_reconstruction'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# pytorch only saves the last model
Deform_save_path = os.path.join(model_save_path, 'deform.pth')
Motion_LA_save_path = os.path.join(model_save_path, 'multiview.pth')

DeformNet = deformnet().cuda()
MV_LA = Mesh_2d().cuda()


optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                              itertools.chain(DeformNet.parameters(), MV_LA.parameters())), lr=lr)
Tensor = torch.cuda.FloatTensor
TensorLong = torch.cuda.LongTensor

# visualisation
writer = SummaryWriter('./runs/model_reconstruction')



def train(epoch):
    DeformNet.train()
    MV_LA.train()

    epoch_loss = []
    epoch_seg_loss = []
    epoch_smooth_loss = []
    epoch_surface_loss = []
    epoch_huber_loss = []



    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):

        img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch = batch


        x_sa_ed = Variable(img_sa_ed.type(Tensor))
        x_2ch_t = Variable(img_2ch_t.type(Tensor))
        x_2ch_ed = Variable(img_2ch_ed.type(Tensor))
        x_4ch_t = Variable(img_4ch_t.type(Tensor))
        x_4ch_ed = Variable(img_4ch_ed.type(Tensor))


        aff_sa_inv = Variable(affine_inv[:, 0,:,:].type(Tensor))
        aff_sa = Variable(affine[:, 0,:,:].type(Tensor))
        aff_2ch_inv = Variable(affine_inv[:, 1,:,:].type(Tensor))
        aff_4ch_inv = Variable(affine_inv[:, 2,:,:].type(Tensor))

        origin_sa = Variable(origin[:, 0:1, :].type(Tensor))
        origin_2ch = Variable(origin[:, 1:2, :].type(Tensor))
        origin_4ch = Variable(origin[:, 2:3, :].type(Tensor))

        vertex_tpl_0 = Variable(vertex_tpl_ed.permute(0,2,1).type(Tensor)) # [bs, 3, number of vertices]
        faces_tpl_0 = Variable(faces_tpl.type(Tensor)) # [bs, number of faces, 3]
        vertex_0 = Variable(vertex_ed.permute(0, 2, 1).type(Tensor))  # [bs, 3, number of vertices]


        mesh2seg_sa_gt = Variable(mesh2seg_sa.type(Tensor))
        mesh2seg_2ch_gt = Variable(mesh2seg_2ch.type(Tensor))
        mesh2seg_4ch_gt = Variable(mesh2seg_4ch.type(Tensor))



        optimizer.zero_grad()

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
        # print (pxx.shape, pyy.shape, pzz.shape)
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
        pred_v_0 = torch.matmul(aff_sa[:, :3, :3], v_0.permute(0, 2, 1)) + aff_sa[:, :3,3:4]  # [bs, 3, number of vertices]
        # print (pred_vertex_t.shape)



        # -------------- differentialable slicer

        # coordinate transformation np.dot(aff_sa_SR_inv[:3,:3], points_ED.T) + aff_sa_SR_inv[:3,3:4]
        v_sa_hat_ed_o = torch.matmul(aff_sa_inv[:, :3, :3], pred_v_0) + aff_sa_inv[:, :3, 3:4]
        v_sa_hat_ed = v_sa_hat_ed_o.permute(0, 2, 1) - origin_sa
        # print (v_sa_hat_t.shape)
        v_2ch_hat_ed_o = torch.matmul(aff_2ch_inv[:, :3, :3], pred_v_0) + aff_2ch_inv[:, :3, 3:4]
        v_2ch_hat_ed = v_2ch_hat_ed_o.permute(0, 2, 1) - origin_2ch
        v_4ch_hat_ed_o = torch.matmul(aff_4ch_inv[:, :3, :3], pred_v_0) + aff_4ch_inv[:, :3, 3:4]
        v_4ch_hat_ed = v_4ch_hat_ed_o.permute(0,2, 1) - origin_4ch

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

        v_2ch_idx_ed, w_2ch_ed = projection(v_2ch_hat_ed_cp, 0, temper)


        # project to LAX 4CH view
        v_4ch_hat_ed_x = torch.clamp(v_4ch_hat_ed[:, :, 0:1], min=0, max=height - 1)
        v_4ch_hat_ed_y = torch.clamp(v_4ch_hat_ed[:, :, 1:2], min=0, max=width - 1)
        v_4ch_hat_ed_cp = torch.cat((v_4ch_hat_ed_x, v_4ch_hat_ed_y, v_4ch_hat_ed[:, :, 2:3]), 2)

        v_4ch_idx_ed, w_4ch_ed = projection(v_4ch_hat_ed_cp, 0, temper)



        # --------------------- Segmentation loss------------------
        loss_seg_sa_ed = projection_weightHD_loss_SA(v_sa_hat_ed_cp, temper, height, width, depth, mesh2seg_sa_gt, 'train')
        loss_seg_2ch_ed = weightedHausdorff_batch(v_2ch_idx_ed, w_2ch_ed, mesh2seg_2ch_gt, height, width, temper, 'train')
        loss_seg_4ch_ed = weightedHausdorff_batch(v_4ch_idx_ed, w_4ch_ed, mesh2seg_4ch_gt, height, width, temper, 'train')


        loss_seg = loss_seg_sa_ed + loss_seg_2ch_ed + loss_seg_4ch_ed


        #----------------smoothness loss------------
        trg_mesh_ed = Meshes(verts=list(pred_v_0.permute(0, 2, 1)), faces=list(faces_tpl_0))
        loss_laplacian_smooth = loss.mesh_laplacian_smoothing(trg_mesh_ed, method='uniform')

        loss_smooth = loss_laplacian_smooth

        # ------------------J loss---------------------
        loss_huber = huber_loss_3d(net_df['out_def_ed'])


        # ------------------Surface chamfer loss---------------------
        loss_surface, _ = loss.chamfer_distance(pred_v_0.permute(0, 2, 1), vertex_0.permute(0, 2, 1))


        loss_all = loss_seg + w_surface * loss_surface + w_smooth * loss_smooth + w_h * loss_huber

        loss_all.backward()
        optimizer.step()



        epoch_loss.append(loss_all.item())
        epoch_seg_loss.append(loss_seg.item())
        epoch_smooth_loss.append(loss_smooth.item())
        epoch_surface_loss.append(loss_surface.item())
        epoch_huber_loss.append(loss_huber.item())



        # tensorboard visulisation
        writer.add_scalar("Loss/train", loss_all, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_seg", loss_seg, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_smooth", loss_smooth, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_huber", loss_huber, epoch * len(training_data_loader) + batch_idx)
        writer.add_scalar("Loss/train_surface", loss_surface, epoch * len(training_data_loader) + batch_idx)



        if batch_idx % 40 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss all: {:.6f}, '
                  'Seg Loss: {:.6f}, Smooth Loss: {:.6f}, Surface Loss: {:.6f}, Huger Loss: {:.6f},'.format(
                epoch, batch_idx * len(img_sa_t), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), np.mean(epoch_loss),
                np.mean(epoch_seg_loss), np.mean(epoch_smooth_loss), np.mean(epoch_surface_loss), np.mean(epoch_huber_loss)))


def val(epoch):
    DeformNet.eval()
    MV_LA.eval()

    val_loss = []
    val_seg_loss = []
    val_smooth_loss = []
    val_surface_loss = []
    val_huber_loss = []


    global base_err
    for batch_idx, batch in tqdm(enumerate(val_data_loader, 1),
                                 total=len(val_data_loader)):

        img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch = batch

        with torch.no_grad():

            x_sa_ed = img_sa_ed.type(Tensor)
            x_2ch_t = img_2ch_t.type(Tensor)
            x_2ch_ed = img_2ch_ed.type(Tensor)
            x_4ch_t = img_4ch_t.type(Tensor)
            x_4ch_ed = img_4ch_ed.type(Tensor)


            aff_sa_inv = affine_inv[:, 0, :, :].type(Tensor)
            aff_sa = affine[:, 0, :, :].type(Tensor)
            aff_2ch_inv = affine_inv[:, 1, :, :].type(Tensor)
            aff_4ch_inv = affine_inv[:, 2, :, :].type(Tensor)


            origin_sa = origin[:, 0:1, :].type(Tensor)
            origin_2ch = origin[:, 1:2, :].type(Tensor)
            origin_4ch = origin[:, 2:3, :].type(Tensor)

            vertex_tpl_0 = vertex_tpl_ed.permute(0, 2, 1).type(Tensor)  # [bs, 3, number of vertices]
            faces_tpl_0 = faces_tpl.type(Tensor)  # [bs, number of faces, 3]
            vertex_0 = vertex_ed.permute(0, 2, 1).cuda()  # [bs, 3, number of vertices]

            mesh2seg_sa_gt = Variable(mesh2seg_sa.type(Tensor))
            mesh2seg_2ch_gt = Variable(mesh2seg_2ch.type(Tensor))
            mesh2seg_4ch_gt = Variable(mesh2seg_4ch.type(Tensor))


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

            v_2ch_idx_ed, w_2ch_ed = projection(v_2ch_hat_ed_cp, 0, temper)

            # project to LAX 4CH view
            v_4ch_hat_ed_x = torch.clamp(v_4ch_hat_ed[:, :, 0:1], min=0, max=height - 1)
            v_4ch_hat_ed_y = torch.clamp(v_4ch_hat_ed[:, :, 1:2], min=0, max=width - 1)
            v_4ch_hat_ed_cp = torch.cat((v_4ch_hat_ed_x, v_4ch_hat_ed_y, v_4ch_hat_ed[:, :, 2:3]), 2)

            v_4ch_idx_ed, w_4ch_ed = projection(v_4ch_hat_ed_cp, 0, temper)

            # --------------------- Segmentation loss------------------
            loss_seg_sa_ed = projection_weightHD_loss_SA(v_sa_hat_ed_cp, temper, height, width, depth, mesh2seg_sa_gt,
                                                         'val')

            loss_seg_2ch_ed = weightedHausdorff_batch(v_2ch_idx_ed, w_2ch_ed, mesh2seg_2ch_gt, height, width, temper,
                                                      'val')
            loss_seg_4ch_ed = weightedHausdorff_batch(v_4ch_idx_ed, w_4ch_ed, mesh2seg_4ch_gt, height, width, temper,
                                                      'val')

            loss_seg = loss_seg_sa_ed + loss_seg_2ch_ed + loss_seg_4ch_ed

            # ----------------smoothness loss------------
            # print (pred_vertex_t.permute(0,2,1).shape)
            trg_mesh_ed = Meshes(verts=list(pred_v_0.permute(0, 2, 1)), faces=list(faces_tpl_0))
            loss_laplacian_smooth = loss.mesh_laplacian_smoothing(trg_mesh_ed, method='uniform')

            loss_smooth = loss_laplacian_smooth

            # ------------------J loss---------------------
            loss_huber = huber_loss_3d(net_df['out_def_ed'])


            # ------------------Surface chamfer loss---------------------
            loss_surface, _ = loss.chamfer_distance(pred_v_0.permute(0, 2, 1), vertex_0.permute(0, 2, 1))

            loss_all = loss_seg + w_surface * loss_surface + w_smooth * loss_smooth + w_h * loss_huber


            val_loss.append(loss_all.item())
            val_seg_loss.append(loss_seg.item())
            val_smooth_loss.append(loss_smooth.item())
            val_surface_loss.append(loss_surface.item())
            val_huber_loss.append(loss_huber.item())

            if batch_idx == 1:
                # tensorboard visulisation
                writer.add_scalar("Loss/val", loss_all, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_seg", loss_seg, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_smooth", loss_smooth, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_huber", loss_huber, epoch * len(training_data_loader) + batch_idx)
                writer.add_scalar("Loss/val_surface", loss_surface, epoch * len(training_data_loader) + batch_idx)


    if np.mean(val_loss) < base_err:
        torch.save(DeformNet.state_dict(), Deform_save_path)
        torch.save(MV_LA.state_dict(), Motion_LA_save_path)
        base_err = np.mean(val_loss)



data_path = '/train_data_path'
train_set = TrainDataset(data_path)
# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker, batch_size=bs, shuffle=True)

val_data_path = '/val_data_path'
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
