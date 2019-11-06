import os

import soft_renderer.functional as srf
import torch
import torchvision
import numpy as np
import tqdm

from PIL import Image

class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}

class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        self.class_ids_map = class_ids_map

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(list(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])
            voxels.append(list(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels

    @property
    def class_ids_pair(self):
        class_names = [self.class_ids_map[i] for i in self.class_ids]
        return zip(self.class_ids, class_names)

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[data_ids_b].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = srf.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15)
        viewpoints_b = srf.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 15)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids

        distances = torch.ones(data_ids.size).float() * self.distance
        elevations = torch.ones(data_ids.size).float() * self.elevation
        viewpoints_all = srf.get_points_from_angles(distances, elevations, -torch.from_numpy(viewpoint_ids).float() * 15)

        for i in range((data_ids.size - 1) // batch_size + 1):
            images = torch.from_numpy(self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.)
            voxels = torch.from_numpy(self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] // 24].astype('float32'))
            yield images, voxels


class PtnShapeNet(object):
    """
    This class is ShapeNet loader provided PTN.
    please download the dataset from 'https://github.com/xcyan/nips16_PTN'.
    """

    def __init__(self, class_id=None, datapath='.', mode='train'):
        if mode not in ['train','val','test']:
            raise ValueError('value of mode is invalid.')


        self.class_id = class_id
        self.elevation = 30.
        self.distance = 2.732
        self.class_ids_map = class_ids_map

        ids_datapath  = os.path.join(datapath, 'shapenetcore_ids')
        view_datapath = os.path.join(datapath, 'shapenetcore_viewdata')

        images = []
        self.num_data = 0  # data num of one class
        self.pos = 0 # local id of class 
        count = 0 # num class counter

        self.id_path = os.path.join(ids_datapath, "{}_{}ids.txt".format(class_id, mode))
        self.instance_ids = []

        with open(self.id_path) as f:
            for s_line in f:
                id = s_line.split('/')[-1].rstrip('\n')
                self.instance_ids.append(id)

        #print(self.instance_ids)

        class_path = os.path.join(view_datapath, class_id) 
        print(class_path)
        self.instance_ids = [name for name in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, name))]
        self.num_data = len(self.instance_ids)

        self.imgs  = []
        self.masks = []

        # loop = tqdm.tqdm(range(self.num_data))
        # loop.set_description('Loading dataset')
        for instance_id in tqdm.tqdm(self.instance_ids):
            instance_path = os.path.join(class_path, str(instance_id))
            img_path  = os.path.join(instance_path, 'imgs') 
            mask_path = os.path.join(instance_path, 'masks')

            imgs_list = [name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]
            #print(imgs)

            for filename in imgs_list:
                img  = np.array(Image.open(os.path.join(img_path, filename)))[np.newaxis,:,:,:]
                mask = np.array(Image.open(os.path.join(mask_path, filename)))[np.newaxis,:,:,np.newaxis]

                #data = np.multiply(img,mask)
                #data = np.concatenate([img,mask], axis=3)

                #print("mask", mask.shape)
                #print("data", data.shape)

                self.imgs.append(img)
                self.masks.append(mask)

        #print(len(self.imgs))
        #print(len(self.masks))

        #self.imgs = np.concatenate(self.imgs, axis=0).reshape((-1, 64, 64, 3))
        self.imgs = np.concatenate(self.imgs, axis=0)
        self.imgs = np.ascontiguousarray(self.imgs)
        #self.masks = np.concatenate(self.masks, axis=0).reshape((-1, 64, 64))
        self.masks = np.concatenate(self.masks, axis=0)
        self.masks = np.ascontiguousarray(self.masks)

        #print(self.imgs.shape)
        #print(self.masks.shape)

            
    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            instance_id = np.random.randint(0, self.num_data)

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = instance_id*24 + viewpoint_id_a
            data_id_b = instance_id*24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

            #print(data_id_a)
            #print(viewpoint_id_a)

        images_a = torch.from_numpy(self.imgs[data_ids_a].astype('float32') / 255.).permute(0,3,1,2)
        images_b = torch.from_numpy(self.imgs[data_ids_b].astype('float32') / 255.).permute(0,3,1,2)

        masks_a = torch.from_numpy(self.masks[data_ids_a].astype('float32') / 255.).permute(0,3,1,2)
        masks_b = torch.from_numpy(self.masks[data_ids_b].astype('float32') / 255.).permute(0,3,1,2)

        masks_a = torch.where(masks_a>=0.25, torch.ones_like(images_a), torch.zeros_like(images_a))
        masks_b = torch.where(masks_b>=0.25, torch.ones_like(images_b), torch.zeros_like(images_b))

        images_a = torch.where(masks_a>=0.25, images_a, torch.zeros_like(images_a))
        images_b = torch.where(masks_b>=0.25, images_b, torch.zeros_like(images_b))

        images_a = torch.cat([images_a, masks_a], dim=1)
        images_b = torch.cat([images_b, masks_b], dim=1)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = srf.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15)
        viewpoints_b = srf.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 15)

        return images_a, images_b, viewpoints_a, viewpoints_b

if __name__ == "__main__":
    dataset = PtnShapeNet(class_id='02828884',
                          datapath='/home/gatheluck/Scratch/nips16_PTN/data',
                          mode='val')
    images_a, images_b, viewpoints_a, viewpoints_b = dataset.get_random_batch(6)
    torchvision.utils.save_image(images_a[:,0:3,:,:], './images_a_ptn.png')
    torchvision.utils.save_image(images_a[:,3,:,:].unsqueeze(1), './images_a_ptn_mask.png')


    # dataset = ShapeNet(directory='/home/gatheluck/Lib/softras/data/datasets', 
    #                    class_ids=['02828884'],
    #                    set_name='test')
    # images_a, images_b, viewpoints_a, viewpoints_b = dataset.get_random_batch(6)
    # print(images_a[0][-1])
    # torchvision.utils.save_image(images_a[:,3,:,:].unsqueeze(1), './images_a.png')