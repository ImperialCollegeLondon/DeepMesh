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
        input_sa, input_2ch, input_4ch, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
        vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, \
        mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch = load_data(self.data_path, self.filename[index], T_num=50)

        img_sa_t = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_t = input_2ch[0]
        img_2ch_ed = input_2ch[1]

        img_4ch_t = input_4ch[0]
        img_4ch_ed = input_4ch[1]

        return img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
               vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch

    def __len__(self):
        return len(self.filename)

class ValDataset(data.Dataset):
    def __init__(self, data_path):
        super(ValDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input_sa, input_2ch, input_4ch, \
        vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, contour_sa_ed, contour_2ch_ed, contour_4ch_ed,  \
        mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch = load_data(self.data_path, self.filename[index], T_num=50,  rand_frame=20)

        img_sa_t = input_sa[0]
        img_sa_ed = input_sa[1]

        img_2ch_t = input_2ch[0]
        img_2ch_ed = input_2ch[1]

        img_4ch_t = input_4ch[0]
        img_4ch_ed = input_4ch[1]


        return img_sa_t, img_sa_ed, img_2ch_t, img_2ch_ed, img_4ch_t, img_4ch_ed, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
               vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch

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


def load_data(data_path, filename, T_num, rand_frame=None):
    # Load images and labels
    img_sa_path = join(data_path, filename, 'sa_img.nii.gz')  # (H, W, 1, frames)
    img_2ch_path = join(data_path, filename, '2ch_img.nii.gz')
    img_4ch_path = join(data_path, filename, '4ch_img.nii.gz')

    mesh2seg_SA_path = join(data_path, filename, 'proj_mesh_SA.npy') # (H, W, D)
    mesh2seg_2CH_path = join(data_path, filename, 'proj_mesh_2CH.npy') # (H, W)
    mesh2seg_4CH_path = join(data_path, filename, 'proj_mesh_4CH.npy') # (H, W)

    contour_sa_path = join(data_path, filename, 'contour_sa.npy')  # (H, W, 9, frames)
    contour_2ch_path = join(data_path, filename, 'contour_2ch.npy')  # (H, W, 1, frames)
    contour_4ch_path = join(data_path, filename, 'contour_4ch.npy')  # (H, W, 1, frames)

    vertices_path = join(data_path, filename, 'vertices_init_myo_ED_smooth.npy')
    faces_path = join(data_path, filename, 'faces_init_myo_ED_smooth.npy')
    affine_path = join(data_path, filename, 'affine.npz')
    origin_path = join(data_path, filename, 'origin.npz')
    vertices_gt_path = join(data_path, filename, 'vertices_resampled_ED.npy')


    # generate random index for t and z dimension
    if rand_frame is not None:
        rand_t = rand_frame
    else:
        rand_t = np.random.randint(0, T_num)

    image_sa_bank = get_data(img_sa_path, rand_t)
    image_2ch_bank = get_data(img_2ch_path, rand_t)
    image_4ch_bank = get_data(img_4ch_path, rand_t)

    contour_sa_ed = np.transpose(np.load(contour_sa_path)[:, :, :, 0], (2, 0, 1))  # [H,W,slices,frame]
    contour_2ch_ed = np.load(contour_2ch_path)[:, :, 0, 0]  # [H,W, 1, frame]
    contour_4ch_ed = np.load(contour_4ch_path)[:, :, 0, 0]  # [H,W, 1, frame]

    # load mesh
    vertex_tpl_ed = np.load(vertices_path)
    faces_tpl = np.load(faces_path)
    vertex_ed = np.load(vertices_gt_path)

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


    mesh2seg_sa = np.load(mesh2seg_SA_path)
    mesh2seg_2ch = np.load(mesh2seg_2CH_path)
    mesh2seg_4ch = np.load(mesh2seg_4CH_path)


    return image_sa_bank, image_2ch_bank, image_4ch_bank, contour_sa_ed, contour_2ch_ed, contour_4ch_ed, \
           vertex_tpl_ed, faces_tpl, affine_inv, affine, origin, vertex_ed, mesh2seg_sa, mesh2seg_2ch, mesh2seg_4ch


