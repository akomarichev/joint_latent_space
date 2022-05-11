""" Joint Latent Space

Author: A. Komarichev
"""

import os
import os.path
import numpy as np
import sys
from skimage import io,transform
import sklearn.preprocessing
import zlib

def save_as_obj(points, name='pred_points.obj'):
    out_filename = 'train_epoch_obj/' +name
    fout = open(out_filename, 'w')
    for i in range(points.shape[0]):
        fout.write('v %f %f %f\n' % \
						  (points[i,0], points[i,1], points[i,2]))
    fout.close()

def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    return cam_mat, cam_pos

class ShapeNetDataset():
    def __init__(self, file_list, batch_size = 32, npoints = 4096, split='train', cache_size=200000, shuffle=None):
        self.batch_size = batch_size
        self.npoints = npoints
        self.normal_channel = True  # always include normals
        
        self.pkl_list = []
        with open(file_list, 'r') as f:
            while(True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)
        self.index = 0
        self.number = len(self.pkl_list)

        # if split == 'train':
        self.pkl_list = [x for x in self.pkl_list if x.endswith('00.dat')]
        self.pkl_list_one_view = [x for x in self.pkl_list if x.startswith('Data/ShapeNetP2M/03001627')] # only chairs
        # self.pkl_list_one_view = [x for x in self.pkl_list if x.startswith('Data/ShapeNetP2M/02691156')] # only plane
        # self.pkl_list_one_view = [x for x in self.pkl_list if x.startswith('Data/ShapeNetP2M/02958343')] # only car
        self.number = len(self.pkl_list_one_view)

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (img, point_set) tuple

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset_one_view()


    def _get_item(self, index):
        if index in self.cache:
            # img_c, point_set_c, surface_area = self.cache[index]
            img_c = self.cache[index]
            img = np.fromstring(zlib.decompress(img_c), dtype='float32').reshape((128,128,3))
            # point_set = np.fromstring(zlib.decompress(point_set_c), dtype='float32').reshape((self.npoints,6))
        else:
            img = np.zeros((128, 128, 3),dtype='float32')
            # point_set = np.zeros((self.npoints, self.num_channel()),dtype='float32')
            pkl_path = self.pkl_list_one_view[index]
            label_path = pkl_path.replace('Data/ShapeNetP2M', '/media/artem/44360B093396D6F4/source_code/image2uniform_point_cloud/data_preparation/ShapeNet/ShapeNetMeshPointCloud')
            img_path = label_path.replace('ShapeNetMeshPointCloud', 'ShapeNetRendering')
            img_path = img_path.replace('rendering/00.dat', 'rendering_one_view/0.png')           
            img_buf = io.imread(img_path)
            img_buf[np.where(img_buf[:,:,3]==0)] = 255
            img_buf = transform.resize(img_buf, (128,128))
            
            # # Normalize point cloud and image to -1 and 1
            img[:] = 2.0 * img_buf[:,:,:3].astype('float32') - 1.0

            if len(self.cache) < self.cache_size:
                self.cache[index] = (zlib.compress(img))
        return img

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.pkl_list)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def reset(self):
        self.idxs = np.arange(0, self.number)
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (self.number+self.batch_size-1) // self.batch_size
        self.batch_idx = 0
    
    def reset_one_view(self):
        self.idxs = np.arange(0, self.number)
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (self.number+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def start_from_the_first_batch_again(self):
        self.batch_idx = 0

    def get_num_batches(self):
        return self.num_batches

    def get_num_batches_one_view(self):
        return self.number

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.number)
        bsize = end_idx - start_idx
        batch_img = np.zeros((bsize, 128, 128, 3),dtype='float32')
        for i in range(bsize):
            img = self._get_item(self.idxs[i+start_idx])
            batch_img[i] = img
        self.batch_idx += 1
        return batch_img
