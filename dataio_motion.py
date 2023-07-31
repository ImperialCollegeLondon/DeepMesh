import torch.utils.data as data
import torch
from os import listdir
from os.path import join
import numpy as np
import nibabel as nib
import glob
import neural_renderer as nr




class TrainDataset(data.Dataset):
    def __init__(self, data_path):
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input_sa, input_2ch, input_4ch, contour_sa, contour_2ch, contour_4ch, \
        vertex_ed, faces, affine_inv, affine, origin = load_data(self.data_path, self.filename[index], T_num=50)

        img_sa_t = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_t = input_2ch[0]
        img_2ch_ed = input_2ch[1]

        img_4ch_t = input_4ch[0]
        img_4ch_ed = input_4ch[1]

        return img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed,\
               contour_sa, contour_2ch, contour_4ch, \
               vertex_ed, faces, affine_inv, affine, origin

    def __len__(self):
        return len(self.filename)

class ValDataset(data.Dataset):
    def __init__(self, data_path):
        super(ValDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input_sa, input_2ch, input_4ch, contour_sa, contour_2ch, contour_4ch, \
        vertex_ed, faces, affine_inv, affine, origin = load_data(self.data_path, self.filename[index], T_num=50,  rand_frame=20)

        img_sa_t = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_t = input_2ch[0]
        img_2ch_ed = input_2ch[1]

        img_4ch_t = input_4ch[0]
        img_4ch_ed = input_4ch[1]


        return img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed,\
               contour_sa, contour_2ch, contour_4ch, \
               vertex_ed, faces, affine_inv, affine, origin

    def __len__(self):
        return len(self.filename)

class TestDataset(data.Dataset):
    def __init__(self, data_path):
        super(TestDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]
        # print (self.filename)

    def __getitem__(self, index):
        input_sa, input_2ch, input_4ch, contour_sa_es, contour_2ch_es, contour_4ch_es, \
        vertex_es, faces, affine_inv, affine, origin = load_data_ES(self.data_path, self.filename[index])

        img_sa_es = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_es = input_2ch[0]
        img_2ch_ed = input_2ch[1]


        img_4ch_es = input_4ch[0]
        img_4ch_ed = input_4ch[1]


        return img_sa_es, img_sa_ed, img_2ch_es, img_2ch_ed, img_4ch_es, img_4ch_ed, \
               contour_sa_es, contour_2ch_es, contour_4ch_es, vertex_es, faces, affine_inv, affine, origin

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

    contour_sa_path = join(data_path, filename, 'contourpara', 'contour_sa.npy') # (H, W, 9, frames)
    contour_2ch_path = join(data_path, filename, 'contourpara', 'contour_2ch.npy') # (H, W, 1, frames)
    contour_4ch_path = join(data_path, filename, 'contourpara', 'contour_4ch.npy')# (H, W, 1, frames)

    vertices_path = join(data_path, filename, 'pred_ED', 'pred_vertices_ED_new.npy')
    faces_path = join(data_path, filename, 'contourpara', 'faces_init_myo_ED.npy')
    affine_path = join(data_path, filename, 'contourpara', 'affine.npz')
    origin_path = join(data_path, filename, 'contourpara', 'origin.npz')

    # generate random index for t and z dimension
    if rand_frame is not None:
        rand_t = rand_frame
    else:
        rand_t = np.random.randint(0, T_num)

    image_sa_bank = get_data(img_sa_path, rand_t)
    image_2ch_bank = get_data(img_2ch_path, rand_t)
    image_4ch_bank = get_data(img_4ch_path, rand_t)

    contour_sa = np.transpose(np.load(contour_sa_path)[:,:,:,rand_t], (2,0,1)) # [H,W,slices,frame]
    contour_2ch = np.load(contour_2ch_path)[:,:, 0, rand_t] # [H,W,1, frame]
    contour_4ch = np.load(contour_4ch_path)[:,:, 0, rand_t] # [H,W,1, frame]


    # load mesh
    vertex_ed = np.load(vertices_path)
    faces = np.load(faces_path)

    # load affine
    aff_sa_inv = np.load(affine_path)['sainv']
    aff_2ch_inv = np.load(affine_path)['la2chinv']
    aff_4ch_inv = np.load(affine_path)['la4chinv']
    affine_inv = np.stack((aff_sa_inv, aff_2ch_inv, aff_4ch_inv), 0)
    aff_sa = np.load(affine_path)['sa']
    aff_2ch = np.load(affine_path)['la2ch']
    aff_4ch = np.load(affine_path)['la4ch']
    affine = np.stack((aff_sa, aff_2ch, aff_4ch), 0)
    # load origin
    origin_sa = np.load(origin_path)['sa']
    origin_2ch = np.load(origin_path)['la2ch']
    origin_4ch = np.load(origin_path)['la4ch']
    origin = np.stack((origin_sa, origin_2ch, origin_4ch), 0)



    return image_sa_bank, image_2ch_bank, image_4ch_bank, contour_sa, contour_2ch, contour_4ch, \
           vertex_ed, faces, affine_inv, affine, origin

def load_data_ES(data_path, filename):
    # Load images and labels
    img_sa_path = join(data_path, filename, 'cropimg_64', 'sa_img.nii.gz')
    img_2ch_path = join(data_path, filename, 'cropimg_64', '2ch_img.nii.gz')
    img_4ch_path = join(data_path, filename, 'cropimg_64', '4ch_img.nii.gz')

    img_sa_ES_path = join(data_path, filename, 'cropimg_64', 'sa_ES_img.nii.gz')
    img_2ch_ES_path = join(data_path, filename, 'cropimg_64', '2ch_ES_img.nii.gz')
    img_4ch_ES_path = join(data_path, filename, 'cropimg_64', '4ch_ES_img.nii.gz')

    contour_sa_path = join(data_path, filename, 'contourpara', 'contour_sa_es.npy')
    contour_2ch_path = join(data_path, filename, 'contourpara', 'contour_2ch_es.npy')
    contour_4ch_path = join(data_path, filename, 'contourpara', 'contour_4ch_es.npy')

    vertices_path = join(data_path, filename, 'pred_ED', 'pred_vertices_ED_new.npy')
    faces_path = join(data_path, filename, 'contourpara', 'faces_init_myo_ED.npy')
    affine_path = join(data_path, filename, 'contourpara', 'affine.npz')
    origin_path = join(data_path, filename, 'contourpara', 'origin.npz')

    # load obj
    vertex_ed = np.load(vertices_path)
    faces = np.load(faces_path)
    # load affine
    aff_sa_inv = np.load(affine_path)['sainv']
    aff_2ch_inv = np.load(affine_path)['la2chinv']
    aff_4ch_inv = np.load(affine_path)['la4chinv']
    affine_inv = np.stack((aff_sa_inv, aff_2ch_inv, aff_4ch_inv), 0)
    aff_sa = np.load(affine_path)['sa']
    aff_2ch = np.load(affine_path)['la2ch']
    aff_4ch = np.load(affine_path)['la4ch']
    affine = np.stack((aff_sa, aff_2ch, aff_4ch), 0)
    # load origin
    origin_sa = np.load(origin_path)['sa']
    origin_2ch = np.load(origin_path)['la2ch']
    origin_4ch = np.load(origin_path)['la4ch']
    origin = np.stack((origin_sa, origin_2ch, origin_4ch), 0)

    image_sa_ES_bank = get_data_ES(img_sa_path, img_sa_ES_path)
    image_2ch_ES_bank = get_data_ES(img_2ch_path, img_2ch_ES_path)
    image_4ch_ES_bank = get_data_ES(img_4ch_path, img_4ch_ES_path)


    contour_sa_es = np.transpose(np.load(contour_sa_path)[:, :, :, 0], (2, 0, 1))  # [H,W,slices,frame]
    contour_2ch_es = np.load(contour_2ch_path)[:, :, 0, 0]  # [H,W,frame]
    contour_4ch_es = np.load(contour_4ch_path)[:, :, 0, 0]  # [H,W,frame]


    return image_sa_ES_bank, image_2ch_ES_bank, image_4ch_ES_bank, \
           contour_sa_es, contour_2ch_es, contour_4ch_es, vertex_ed, faces, affine_inv, affine, origin

