import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import nibabel as nib
import glob
import cv2
from scipy import ndimage
from utils import *


class TrainDataset(data.Dataset):
    def __init__(self, data_path):
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input_sa, input_2ch, input_4ch, target, contour_sa, contour_2ch, contour_4ch, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        mask_sa_index, mask_2ch_ind_x, mask_2ch_ind_y, mask_2ch_ind_z, \
        mask_4ch_ind_x, mask_4ch_ind_y, mask_4ch_ind_z = load_data(self.data_path, self.filename[index], T_num=50)

        img_sa_t = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_t = input_2ch[0]
        img_2ch_ed = input_2ch[1]

        img_4ch_t = input_4ch[0]
        img_4ch_ed = input_4ch[1]

        seg = target[0]
        seg_ed = target[1]

        return img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, seg, seg_ed,\
               contour_sa, contour_2ch, contour_4ch, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
               mask_sa_index, mask_2ch_ind_x, mask_2ch_ind_y, mask_2ch_ind_z,\
               mask_4ch_ind_x, mask_4ch_ind_y, mask_4ch_ind_z

    def __len__(self):
        return len(self.filename)

class ValDataset(data.Dataset):
    def __init__(self, data_path):
        super(ValDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):

        input_sa, input_2ch, input_4ch, target, contour_sa, contour_2ch, contour_4ch, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        mask_sa_index, mask_2ch_ind_x, mask_2ch_ind_y, mask_2ch_ind_z, \
        mask_4ch_ind_x, mask_4ch_ind_y, mask_4ch_ind_z = load_data(self.data_path, self.filename[index], T_num=50,  rand_frame=23)

        img_sa_t = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_t = input_2ch[0]
        img_2ch_ed = input_2ch[1]

        img_4ch_t = input_4ch[0]
        img_4ch_ed = input_4ch[1]

        seg = target[0]
        seg_ed = target[1]

        return img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, seg, seg_ed, \
               contour_sa, contour_2ch, contour_4ch, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
               mask_sa_index, mask_2ch_ind_x, mask_2ch_ind_y, mask_2ch_ind_z, \
               mask_4ch_ind_x, mask_4ch_ind_y, mask_4ch_ind_z

    def __len__(self):
        return len(self.filename)

class TestDataset(data.Dataset):
    def __init__(self, data_path):
        super(TestDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]
        # print (self.filename)

    def __getitem__(self, index):
        input_sa, input_2ch, input_4ch, target = load_data_ES(self.data_path, self.filename[index])

        img_sa_es = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_es = input_2ch[0]
        img_2ch_ed = input_2ch[1]


        img_4ch_es = input_4ch[0]
        img_4ch_ed = input_4ch[1]

        seg_es = target[0]
        seg_ed = target[1]

        return img_sa_es, img_sa_ed, img_2ch_es, img_2ch_ed, img_4ch_es, img_4ch_ed, seg_es, seg_ed

    def __len__(self):
        return len(self.filename)

def get_data(path, fr):
    nim = nib.load(path)
    image = nim.get_data()[:, :, :, :]  # (h, w, slices, frame)
    image = np.array(image, dtype='float32')


    image_fr = image[..., fr]
    image_fr = image_fr[np.newaxis]
    image_ed = image[..., 0]
    image_ed = image_ed[np.newaxis]

    image_bank = np.concatenate((image_fr, image_ed), axis=0)
    image_bank = np.transpose(image_bank, (0, 3, 1, 2))


    return image_bank

def get_data_ES(path, path_ES):
    nim = nib.load(path)
    image = nim.get_data()[:, :, :, :]  # (h, w, slices, frame)
    image = np.array(image, dtype='float32')

    nim_ES = nib.load(path_ES)
    image_ES = nim_ES.get_data()[:, :, :, :]  # (h, w, slices, frame=0)
    image_ES = np.array(image_ES, dtype='float32')


    image_z_ed = image[..., 0]
    image_z_ed = image_z_ed[np.newaxis]
    image_z_es = image_ES[..., 0]
    image_z_es = image_z_es[np.newaxis]


    image_bank = np.concatenate((image_z_es, image_z_ed), axis=0)
    image_bank = np.transpose(image_bank, (0, 3, 1, 2))


    return image_bank

def load_data(data_path, filename, T_num, rand_frame=None):
    # Load images and labels
    img_sa_path = join(data_path, filename, 'cropimg_64', 'sa_img.nii.gz')  # (H, W, 1, frames)
    img_2ch_path = join(data_path, filename, 'cropimg_64', '2ch_img.nii.gz')
    img_4ch_path = join(data_path, filename, 'cropimg_64', '4ch_img.nii.gz')
    seg_sa_path = join(data_path, filename, 'cropimg_64', 'sa_seg.nii.gz')

    contour_sa_path = join(data_path, filename, 'cropimg_64', 'contour_sa.npy')
    contour_2ch_path = join(data_path, filename, 'cropimg_64', 'contour_2ch.npy')
    contour_4ch_path = join(data_path, filename, 'cropimg_64', 'contour_4ch.npy')

    mask_sa_index_path = join(data_path, filename, 'cropimg_64', 'sa_slice_index.npy')
    mask_2ch_index_path = join(data_path, filename, 'cropimg_64', 'mask_2ch_index.npz')
    mask_4ch_index_path = join(data_path, filename, 'cropimg_64', 'mask_4ch_index.npz')

    # generate random index for t and z dimension
    if rand_frame is not None:
        rand_t = rand_frame
    else:
        rand_t = np.random.randint(0, T_num)

    image_sa_bank = get_data(img_sa_path, rand_t)
    image_2ch_bank = get_data(img_2ch_path, rand_t)
    image_4ch_bank = get_data(img_4ch_path, rand_t)
    seg_sa_bank = get_data(seg_sa_path, rand_t)
    seg_sa_bank = np.array(seg_sa_bank, dtype='int16')  # (c, slices, H, W)

    contour_sa = np.transpose(np.load(contour_sa_path)[:,:,:,rand_t], (2,0,1))
    contour_2ch = np.load(contour_2ch_path)[:,:,rand_t] # [H,W,frame]
    contour_4ch = np.load(contour_4ch_path)[:,:,rand_t] # [H,W,frame]

    contour_sa_ed = np.transpose(np.load(contour_sa_path)[:, :, :, 0], (2, 0, 1))
    contour_2ch_ed = np.load(contour_2ch_path)[:, :, 0]  # [H,W,frame]
    contour_4ch_ed = np.load(contour_4ch_path)[:, :, 0]  # [H,W,frame]

    mask_sa_index = np.load(mask_sa_index_path) # [9]
    mask_2ch_ind_x = np.load(mask_2ch_index_path)['indx'][:,rand_t] # indx: [M, frame]
    mask_2ch_ind_y = np.load(mask_2ch_index_path)['indy'][:,rand_t] # indx: [M, frame]
    mask_2ch_ind_z = np.load(mask_2ch_index_path)['indz'][:,rand_t] # indx: [M, frame]
    mask_4ch_ind_x = np.load(mask_4ch_index_path)['indx'][:, rand_t]  # indx: [M, frame]
    mask_4ch_ind_y = np.load(mask_4ch_index_path)['indy'][:, rand_t]  # indx: [M, frame]
    mask_4ch_ind_z = np.load(mask_4ch_index_path)['indz'][:, rand_t]  # indx: [M, frame]


    return image_sa_bank, image_2ch_bank, image_4ch_bank, seg_sa_bank, contour_sa, contour_2ch, contour_4ch, \
           contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
           mask_sa_index, mask_2ch_ind_x, mask_2ch_ind_y, mask_2ch_ind_z, mask_4ch_ind_x, mask_4ch_ind_y, mask_4ch_ind_z

def load_data_ES(data_path, filename):
    # Load images and labels
    img_sa_path = join(data_path, filename, 'cropimg_64', 'sa_img.nii.gz')
    img_2ch_path = join(data_path, filename, 'cropimg_64', '2ch_img.nii.gz')
    img_4ch_path = join(data_path, filename, 'cropimg_64', '4ch_img.nii.gz')
    seg_sa_path = join(data_path, filename, 'cropimg_64', 'sa_seg.nii.gz')

    img_sa_ES_path = join(data_path, filename, 'cropimg_64', 'sa_ES_img.nii.gz')
    img_2ch_ES_path = join(data_path, filename, 'cropimg_64', '2ch_ES_img.nii.gz')
    img_4ch_ES_path = join(data_path, filename, 'cropimg_64', '4ch_ES_img.nii.gz')
    seg_sa_ES_path = join(data_path, filename, 'cropimg_64', 'sa_ES_seg.nii.gz')


    image_sa_ES_bank = get_data_ES(img_sa_path, img_sa_ES_path)
    image_2ch_ES_bank = get_data_ES(img_2ch_path, img_2ch_ES_path)
    image_4ch_ES_bank = get_data_ES(img_4ch_path, img_4ch_ES_path)
    seg_sa_ES_bank = get_data_ES(seg_sa_path, seg_sa_ES_path)
    seg_sa_ES_bank = np.array(seg_sa_ES_bank, dtype='int16')  # (c, slices, H, W)


    return image_sa_ES_bank, image_2ch_ES_bank, image_4ch_ES_bank, seg_sa_ES_bank


