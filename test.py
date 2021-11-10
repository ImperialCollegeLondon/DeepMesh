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

from network import *
from dataio import *
import scipy.io



n_class = 4
n_worker = 4
bs = 1
T_num = 50 # number of frames

# model_save_path = './model'
R_mesh_SA_path = os.path.join(model_save_path, 'R_mesh_SA_3d.pth')
Mesh_LA_path = os.path.join(model_save_path, 'Mesh_LA_2d.pth')


def test():
    R_mesh_SA.eval()
    Mesh_LA.eval()

    Myo_dice = []


    for batch_idx, batch in tqdm(enumerate(testing_data_loader, 1),
                                 total=len(testing_data_loader)):
        img_sa_es, img_sa_ed, img_2ch_es, img_2ch_ed, img_4ch_es, img_4ch_ed, seg_es, seg_ed = batch

        x_sa_es = Variable(img_sa_es.type(Tensor))
        x_sa_ed = Variable(img_sa_ed.type(Tensor))
        x_2ch_es = Variable(img_2ch_es.type(Tensor))
        x_2ch_ed = Variable(img_2ch_ed.type(Tensor))
        x_4ch_es = Variable(img_4ch_es.type(Tensor))
        x_4ch_ed = Variable(img_4ch_ed.type(Tensor))

        seg_sa_ed = Variable(seg_ed.unsqueeze(1).type(Tensor))
        seg_sa_es = Variable(seg_es.unsqueeze(1).type(Tensor))

        x_sa_ed_5D = Variable(img_sa_ed.unsqueeze(1).type(Tensor))

        net_la = Mesh_LA(x_2ch_es, x_2ch_ed, x_4ch_es, x_4ch_ed)
        net_sa = R_mesh_SA(x_sa_es, x_sa_ed, net_la['conv2_2ch'], net_la['conv2s_2ch'], net_la['conv2_4ch'],
                           net_la['conv2s_4ch'])

        pred_sa = transform(x_sa_ed_5D, net_sa['out'], mode='bilinear')

        # the result that warp frame t to frame ED
        warped_seg_es_5D = transform(seg_sa_ed, net_sa['out'], mode='nearest')
        warped_seg_es_5D = warped_seg_es_5D.data.cpu().numpy()
        seg_es_5D_1hot = convert_to_1hot_3D(seg_es.unsqueeze(1).numpy(), 4)
        warped_seg_es_5D_1hot = convert_to_1hot_3D(warped_seg_es_5D.astype(np.int16), 4)

        # compute dice
        myodice = categorical_dice(warped_seg_es_5D_1hot, seg_es_5D_1hot, 2)

        Myo_dice.append(myodice)


    print('Myo DICE: {:.4f}({:.4f})'.format(np.mean(Myo_dice), np.std(Myo_dice)))


    return

test_data_path = '/testdata'
test_set = TestDataset(test_data_path)
testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker, batch_size=bs, shuffle=False)

R_mesh_SA = MotionMesh_25d().cuda()
Mesh_LA = Mesh_2d().cuda()

R_mesh_SA.load_state_dict(torch.load(R_mesh_SA_path), strict=True)
Mesh_LA.load_state_dict(torch.load(Mesh_LA_path), strict=True)

Tensor = torch.cuda.FloatTensor
flow_criterion = nn.MSELoss()

start = time.time()
test()
end = time.time()
print("testing took {:.8f}".format(end - start))

