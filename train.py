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

from network import *
from dataio import *
from utils import *

lr = 1e-4
n_worker = 4
bs = 3
n_epoch = 300
base_err = 1000

w_h = 5e-3
w_shape = 5
depth = 64
seg2d_depth = 9
width = 128
height = 128



model_save_path = './models/model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# pytorch only saves the last model
R_mesh_SA_save_path = os.path.join(model_save_path, 'R_mesh_SA_3d.pth')
Mesh_LA_save_path = os.path.join(model_save_path, 'Mesh_LA_2d.pth')

flow_criterion = nn.MSELoss()
seg_criterion = nn.CrossEntropyLoss()

R_mesh_SA = MotionMesh_25d().cuda()
Mesh_LA = Mesh_2d().cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                              itertools.chain(R_mesh_SA.parameters(), Mesh_LA.parameters())), lr=lr)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, R_SA.parameters()), lr=lr, weight_decay=5e-4)
Tensor = torch.cuda.FloatTensor
TensorLong = torch.cuda.LongTensor

# visualisation
writer = SummaryWriter('./runs/model')


def train(epoch):
    R_mesh_SA.train()
    Mesh_LA.train()

    epoch_loss = []
    epoch_shape_loss = []
    epoch_motion_loss = []
    Myo_dice = []


    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):

        img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, seg, seg_ed, \
        contour_sa, contour_2ch, contour_4ch, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        mask_sa_index, mask_2ch_ind_x, mask_2ch_ind_y, mask_2ch_ind_z, \
        mask_4ch_ind_x, mask_4ch_ind_y, mask_4ch_ind_z = batch

        x_sa_t = Variable(img_sa_t.type(Tensor))
        x_sa_ed = Variable(img_sa_ed.type(Tensor))
        x_2ch_t = Variable(img_2ch_t.type(Tensor))
        x_2ch_ed = Variable(img_2ch_ed.type(Tensor))
        x_4ch_t = Variable(img_4ch_t.type(Tensor))
        x_4ch_ed = Variable(img_4ch_ed.type(Tensor))

        x_sa_t_5D = Variable(img_sa_t.unsqueeze(1).type(Tensor))
        x_sa_ed_5D = Variable(img_sa_ed.unsqueeze(1).type(Tensor))
        seg_sa_ed_5D = Variable(seg_ed.unsqueeze(1).type(Tensor))

        con_sa_ed = Variable(contour_sa_ed.type(TensorLong)) # [bs, slices, H, W]
        con_2ch_ed = Variable(contour_2ch_ed.type(TensorLong)) # [bs, H, W]
        con_4ch_ed = Variable(contour_4ch_ed.type(TensorLong)) # [bs, H, W]
        con_sa = Variable(contour_sa.type(TensorLong))  # [bs, slices, H, W]
        con_2ch = Variable(contour_2ch.type(TensorLong))  # [bs, H, W]
        con_4ch = Variable(contour_4ch.type(TensorLong))  # [bs, H, W]

        sa_index = Variable(mask_sa_index.type(TensorLong))
        la_2ch_ind_x = Variable(mask_2ch_ind_x.type(TensorLong))
        la_2ch_ind_y = Variable(mask_2ch_ind_y.type(TensorLong))
        la_2ch_ind_z = Variable(mask_2ch_ind_z.type(TensorLong))
        la_4ch_ind_x = Variable(mask_4ch_ind_x.type(TensorLong))
        la_4ch_ind_y = Variable(mask_4ch_ind_y.type(TensorLong))
        la_4ch_ind_z = Variable(mask_4ch_ind_z.type(TensorLong))


        optimizer.zero_grad()

        net_la = Mesh_LA(x_2ch_t, x_2ch_ed, x_4ch_t, x_4ch_ed)
        net_sa = R_mesh_SA(x_sa_t, x_sa_ed, net_la['conv2_2ch'], net_la['conv2s_2ch'], net_la['conv2_4ch'], net_la['conv2s_4ch'])

        pred_sa = transform(x_sa_ed_5D, net_sa['out'], mode='bilinear')
        edge_sa_ed = net_sa['out_edge_ed']
        edge_sa = net_sa['out_edge']
        pred_edge_sa = transform(edge_sa_ed, net_sa['out'], mode='bilinear')



        # --------------------- Slicer (related to batch size)------------------
        slice_2ch = torch.reshape(edge_sa[torch.arange(edge_sa.shape[0]).unsqueeze(-1), :, la_2ch_ind_z, la_2ch_ind_x, la_2ch_ind_y].permute(0, 2, 1), (edge_sa.shape[0], 2, depth, width))
        slice_4ch = torch.reshape(edge_sa[torch.arange(edge_sa.shape[0]).unsqueeze(-1), :, la_4ch_ind_z, la_4ch_ind_x, la_4ch_ind_y].permute(0, 2, 1), (edge_sa.shape[0], 2, depth, width))
        slices_sa = torch.reshape(edge_sa[torch.arange(edge_sa.shape[0]).unsqueeze(-1), :, sa_index, :,:].permute(0,2,1,3,4), (edge_sa.shape[0], 2, seg2d_depth, width, height))

        slice_2ch_warp = torch.reshape(pred_edge_sa[torch.arange(pred_edge_sa.shape[0]).unsqueeze(-1), :, la_2ch_ind_z, la_2ch_ind_x, la_2ch_ind_y].permute(0, 2, 1), (pred_edge_sa.shape[0], 2, depth, width))
        slice_4ch_warp = torch.reshape(pred_edge_sa[torch.arange(pred_edge_sa.shape[0]).unsqueeze(-1), :, la_4ch_ind_z, la_4ch_ind_x, la_4ch_ind_y].permute(0, 2, 1), (pred_edge_sa.shape[0], 2, depth, width))
        slices_sa_warp = torch.reshape(pred_edge_sa[torch.arange(pred_edge_sa.shape[0]).unsqueeze(-1), :, sa_index, :, :].permute(0, 2, 1, 3, 4), (pred_edge_sa.shape[0], 2, seg2d_depth, width, height))

        slice_2ch_ed = torch.reshape(
            edge_sa_ed[torch.arange(edge_sa_ed.shape[0]).unsqueeze(-1), :, la_2ch_ind_z, la_2ch_ind_x,
            la_2ch_ind_y].permute(
                0, 2, 1), (edge_sa_ed.shape[0], 2, depth, width))
        slice_4ch_ed = torch.reshape(
            edge_sa_ed[torch.arange(edge_sa_ed.shape[0]).unsqueeze(-1), :, la_4ch_ind_z, la_4ch_ind_x, la_4ch_ind_y].permute(
                0, 2, 1), (edge_sa_ed.shape[0], 2, depth, width))
        slices_sa_ed = torch.reshape(
            edge_sa_ed[torch.arange(edge_sa_ed.shape[0]).unsqueeze(-1), :, sa_index, :, :].permute(0, 2, 1, 3, 4),
            (edge_sa_ed.shape[0], 2, seg2d_depth, width, height))

        #----------------3d motion loss------------
        loss_sa = flow_criterion(pred_sa, x_sa_t_5D) + w_h * huber_loss_3d(net_sa['out'])

        #----------------Shape loss------------------
        loss_shape = (seg_criterion(slices_sa, con_sa) + seg_criterion(slices_sa_warp, con_sa) +
                     seg_criterion(slice_2ch, con_2ch) + seg_criterion(slice_2ch_warp, con_2ch) +
                     seg_criterion(slice_4ch, con_4ch) + seg_criterion(slice_4ch_warp, con_4ch) +
                     seg_criterion(slices_sa_ed, con_sa_ed) + seg_criterion(slice_2ch_ed, con_2ch_ed) +
                     seg_criterion(slice_4ch_ed, con_4ch_ed)) / 9

        loss = loss_sa + w_shape * loss_shape

        loss.backward()
        optimizer.step()

        warped_seg_t = transform(seg_sa_ed_5D, net_sa['out'], mode='nearest')
        warped_seg_t = warped_seg_t.data.cpu().numpy()
        seg_t_1hot = convert_to_1hot_3D(seg.unsqueeze(1).numpy(), 4)
        warped_seg_t_1hot = convert_to_1hot_3D(warped_seg_t.astype(np.int16), 4)
        Myo_dice.append(categorical_dice(warped_seg_t_1hot, seg_t_1hot, 2))

        epoch_loss.append(loss.item())
        epoch_motion_loss.append(loss_sa.item())
        epoch_shape_loss.append(loss_shape.item())


        if batch_idx % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, '
                  'Motion Loss: {:.6f}, Shape Loss: {:.6f}, Myo Dice: {:.6f}'.format(
                epoch, batch_idx * len(img_sa_t), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), np.mean(epoch_loss),
                np.mean(epoch_motion_loss), np.mean(epoch_shape_loss), np.mean(Myo_dice)))


def val(epoch):
    R_mesh_SA.eval()
    Mesh_LA.eval()

    val_loss = []
    Myo_dice = []

    global base_err
    for batch_idx, batch in tqdm(enumerate(val_data_loader, 1),
                                 total=len(val_data_loader)):

        img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, seg, seg_ed, \
        contour_sa, contour_2ch, contour_4ch, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        mask_sa_index, mask_2ch_ind_x, mask_2ch_ind_y, mask_2ch_ind_z, \
        mask_4ch_ind_x, mask_4ch_ind_y, mask_4ch_ind_z = batch

        x_sa_t = Variable(img_sa_t.type(Tensor))
        x_sa_ed = Variable(img_sa_ed.type(Tensor))
        x_2ch_t = Variable(img_2ch_t.type(Tensor))
        x_2ch_ed = Variable(img_2ch_ed.type(Tensor))
        x_4ch_t = Variable(img_4ch_t.type(Tensor))
        x_4ch_ed = Variable(img_4ch_ed.type(Tensor))

        x_sa_t_5D = Variable(img_sa_t.unsqueeze(1).type(Tensor))
        x_sa_ed_5D = Variable(img_sa_ed.unsqueeze(1).type(Tensor))
        seg_sa_ed_5D = Variable(seg_ed.unsqueeze(1).type(Tensor))

        con_sa_ed = Variable(contour_sa_ed.type(TensorLong))  # [bs, slices, H, W]
        con_2ch_ed = Variable(contour_2ch_ed.type(TensorLong))  # [bs, H, W]
        con_4ch_ed = Variable(contour_4ch_ed.type(TensorLong))  # [bs, H, W]
        con_sa = Variable(contour_sa.type(TensorLong))  # [bs, slices, H, W]
        con_2ch = Variable(contour_2ch.type(TensorLong))  # [bs, H, W]
        con_4ch = Variable(contour_4ch.type(TensorLong))  # [bs, H, W]

        sa_index = Variable(mask_sa_index.type(TensorLong))
        la_2ch_ind_x = Variable(mask_2ch_ind_x.type(TensorLong))
        la_2ch_ind_y = Variable(mask_2ch_ind_y.type(TensorLong))
        la_2ch_ind_z = Variable(mask_2ch_ind_z.type(TensorLong))
        la_4ch_ind_x = Variable(mask_4ch_ind_x.type(TensorLong))
        la_4ch_ind_y = Variable(mask_4ch_ind_y.type(TensorLong))
        la_4ch_ind_z = Variable(mask_4ch_ind_z.type(TensorLong))

        net_la = Mesh_LA(x_2ch_t, x_2ch_ed, x_4ch_t, x_4ch_ed)
        net_sa = R_mesh_SA(x_sa_t, x_sa_ed, net_la['conv2_2ch'], net_la['conv2s_2ch'], net_la['conv2_4ch'],
                           net_la['conv2s_4ch'])

        pred_sa = transform(x_sa_ed_5D, net_sa['out'], mode='bilinear')
        edge_sa_ed = net_sa['out_edge_ed']
        edge_sa = net_sa['out_edge']
        # print (edge_sa.shape, net_sa['out'].shape)
        pred_edge_sa = transform(edge_sa_ed, net_sa['out'], mode='bilinear')

        # --------------------- Slicer (related to batch size)------------------
        slice_2ch = torch.reshape(
            edge_sa[torch.arange(edge_sa.shape[0]).unsqueeze(-1), :, la_2ch_ind_z, la_2ch_ind_x, la_2ch_ind_y].permute(
                0, 2, 1), (edge_sa.shape[0], 2, depth, width))
        slice_4ch = torch.reshape(
            edge_sa[torch.arange(edge_sa.shape[0]).unsqueeze(-1), :, la_4ch_ind_z, la_4ch_ind_x, la_4ch_ind_y].permute(
                0, 2, 1), (edge_sa.shape[0], 2, depth, width))
        slices_sa = torch.reshape(
            edge_sa[torch.arange(edge_sa.shape[0]).unsqueeze(-1), :, sa_index, :, :].permute(0, 2, 1, 3, 4),
            (edge_sa.shape[0], 2, seg2d_depth, width, height))

        slice_2ch_warp = torch.reshape(
            pred_edge_sa[torch.arange(pred_edge_sa.shape[0]).unsqueeze(-1), :, la_2ch_ind_z, la_2ch_ind_x,
            la_2ch_ind_y].permute(0, 2, 1), (pred_edge_sa.shape[0], 2, depth, width))
        slice_4ch_warp = torch.reshape(
            pred_edge_sa[torch.arange(pred_edge_sa.shape[0]).unsqueeze(-1), :, la_4ch_ind_z, la_4ch_ind_x,
            la_4ch_ind_y].permute(0, 2, 1), (pred_edge_sa.shape[0], 2, depth, width))
        slices_sa_warp = torch.reshape(
            pred_edge_sa[torch.arange(pred_edge_sa.shape[0]).unsqueeze(-1), :, sa_index, :, :].permute(0, 2, 1, 3, 4),
            (pred_edge_sa.shape[0], 2, seg2d_depth, width, height))

        slice_2ch_ed = torch.reshape(
            edge_sa_ed[torch.arange(edge_sa_ed.shape[0]).unsqueeze(-1), :, la_2ch_ind_z, la_2ch_ind_x,
            la_2ch_ind_y].permute(
                0, 2, 1), (edge_sa_ed.shape[0], 2, depth, width))
        slice_4ch_ed = torch.reshape(
            edge_sa_ed[torch.arange(edge_sa_ed.shape[0]).unsqueeze(-1), :, la_4ch_ind_z, la_4ch_ind_x,
            la_4ch_ind_y].permute(
                0, 2, 1), (edge_sa_ed.shape[0], 2, depth, width))
        slices_sa_ed = torch.reshape(
            edge_sa_ed[torch.arange(edge_sa_ed.shape[0]).unsqueeze(-1), :, sa_index, :, :].permute(0, 2, 1, 3, 4),
            (edge_sa_ed.shape[0], 2, seg2d_depth, width, height))

        # ----------------3d motion loss------------
        loss_sa = flow_criterion(pred_sa, x_sa_t_5D) + w_h * huber_loss_3d(net_sa['out'])

        # ----------------Shape loss------------------
        loss_shape = (seg_criterion(slices_sa, con_sa) + seg_criterion(slices_sa_warp, con_sa) +
                     seg_criterion(slice_2ch, con_2ch) + seg_criterion(slice_2ch_warp, con_2ch) +
                     seg_criterion(slice_4ch, con_4ch) + seg_criterion(slice_4ch_warp, con_4ch) +
                     seg_criterion(slices_sa_ed, con_sa_ed) + seg_criterion(slice_2ch_ed, con_2ch_ed) +
                     seg_criterion(slice_4ch_ed, con_4ch_ed)) / 9

        loss = loss_sa + w_shape * loss_shape

        warped_seg_t = transform(seg_sa_ed_5D, net_sa['out'], mode='nearest')
        warped_seg_t = warped_seg_t.data.cpu().numpy()
        seg_t_1hot = convert_to_1hot_3D(seg.unsqueeze(1).numpy(), 4)
        warped_seg_t_1hot = convert_to_1hot_3D(warped_seg_t.astype(np.int16), 4)
        Myo_dice.append(categorical_dice(warped_seg_t_1hot, seg_t_1hot, 2))


        val_loss.append(loss.item())

    print('Loss: {:.6f}, Myo Dice: {:.6f}'.format(np.mean(val_loss), np.mean(Myo_dice)))

    if np.mean(val_loss) < base_err:
        torch.save(R_mesh_SA.state_dict(), R_mesh_SA_save_path)
        torch.save(Mesh_LA.state_dict(), Mesh_LA_save_path)
        base_err = np.mean(val_loss)


data_path = '/vol/bitbucket/qm216/UKBBcardiac/cardiacdata/ukbbSALA_CMR/train'
train_set = TrainDataset(data_path)
# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker, batch_size=bs, shuffle=True)

val_data_path = '/vol/bitbucket/qm216/UKBBcardiac/cardiacdata/ukbbSALA_CMR/val'
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

