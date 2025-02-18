import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr
import soft_renderer as sr
import soft_renderer.functional as srf
import math

import open3d

from skimage import measure

class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return x


class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        # vertices_base, faces = srf.load_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])#vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])#faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces


class Model(nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        self.encoder = Encoder(im_size=args.image_size)
        self.decoder = Decoder(filename_obj)
        if args.renderer_type == 'softras':
            self.renderer = sr.SoftRenderer(image_size=args.image_size, sigma_val=args.sigma_val, 
                                            aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                            dist_eps=1e-10)
        elif args.renderer_type == 'nmr':
            self.renderer = nr.Renderer(image_size=args.image_size, camera_mode='look_at',  viewing_angle=15)
            #self.renderer.eye = [0, 0, -2.732]

        else:
            raise NotImplementedError

        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)

        self.renderer_type = args.renderer_type
        self.transform = args.transform

    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)

    def reconstruct(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces

    def predict_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b):
        batch_size = image_a.size(0)
        # [Ia, Ib]
        images = torch.cat((image_a, image_b), dim=0)
        # [Va, Va, Vb, Vb], set viewpoints
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        
        if self.renderer_type == 'softras':
            self.renderer.transform.set_eyes(viewpoints)
        elif self.renderer_type == 'nmr':
            self.transform.set_eyes(viewpoints)


        vertices, faces = self.reconstruct(images)
        laplacian_loss = self.laplacian_loss(vertices)
        flatten_loss = self.flatten_loss(vertices)

        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)

        # print(vertices.shape) # torch.Size([256, 642, 3])

        # [Raa, Rba, Rab, Rbb], cross render multiview images
        if self.renderer_type == 'softras':
            silhouettes = self.renderer(vertices, faces)
            #print(silhouettes.shape) # torch.Size([256, 4, 64, 64])
        elif self.renderer_type == 'nmr':
            # transform vert
            vertices = self.transform.transformer(vertices)
            silhouettes = self.renderer(vertices, faces, mode='silhouettes')
            silhouettes = silhouettes.unsqueeze(1).repeat(1,4,1,1) # this operation adjust output size of nmr to softras
            #print(silhouettes.shape) # torch.Size([256, 4, 64, 64])

        return silhouettes.chunk(4, dim=0), laplacian_loss, flatten_loss

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.reconstruct(images)

        # print("vertices:", vertices.shape) # torch.Size([100, 642, 3])

        # print("torch.mean(vertices, dim=1)")
        # print(torch.mean(vertices, dim=[0,1]))
        # print(torch.max(vertices)) # 0.5
        # print(torch.min(vertices)) # -0.5

        #print("voxels: ", voxels.shape) # voxels:  (100, 32, 32, 32) 

        ### 3D IoU ###
        faces_ = srf.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        
        
        ### F score ###
        # voxel to ptc
        # for i in range(voxels.shape[0]):
        #     voxel = voxels[i, ...] # (32, 32, 32)

        #     vert_gt, face, _, _ = measure.marching_cubes_lewiner(voxel, 0)
        #     vert_gt = torch.from_numpy(vert_gt.copy()).float()
        #     vert_pr = vertices[i,:,:].detach().cpu()

        #     vert_gt = (vert_gt / 32.0) - 0.5
        #     flip_z = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]]).float()
        #     vert_gt = torch.matmul(vert_gt, flip_z)
        #     vert_gt = torch.stack([vert_gt[:,1], vert_gt[:,0], vert_gt[:,2]], dim=-1)

        #     face_gt = faces[i,:,:]
        #     srf.save_obj('./vert_gt_{:03d}.obj'.format(i), vert_gt.float(), torch.from_numpy(face.copy()).long())
        #     srf.save_obj('./vert_pr_{:03d}.obj'.format(i), vert_pr.float(), face_gt.long())

        #     print("vert_gt: ", vert_gt.shape)
        #     print("vert_pr: ", vert_pr.shape)

        #     print("torch.max(vert_gt): ", torch.max(vert_gt))
        #     print("torch.max(vert_pr): ", torch.max(vert_pr))

        #     print("torch.min(vert_gt): ", torch.min(vert_gt))
        #     print("torch.min(vert_pr): ", torch.min(vert_pr))

        #     pcd_gt = open3d.geometry.PointCloud()
        #     pcd_gt.points = open3d.Vector3dVector(vert_gt)

        #     pcd_pr = open3d.geometry.PointCloud()
        #     pcd_pr.points = open3d.Vector3dVector(vert_pr)

        #     d1 = open3d.geometry.compute_point_cloud_to_point_cloud_distance(pcd_gt, pcd_pr)
        #     d2 = open3d.geometry.compute_point_cloud_to_point_cloud_distance(pcd_pr, pcd_gt)
            
        #     th = 0.01

        #     if len(d1) and len(d2):
        #         recall = float(sum(d < th for d in d2)) / float(len(d2))
        #         precision = float(sum(d < th for d in d1)) / float(len(d1))

        #         if recall+precision > 0:
        #             fscore = 2 * recall * precision / (recall + precision)
        #         else:
        #             fscore = 0
        #     else:
        #         fscore = 0
        #         precision = 0
        #         recall = 0
            
        #     print(fscore)

        #     if i == 10: break
            #raise NotImplementedError

        
        
        
        
        return iou, vertices, faces

    def forward(self, images=None, viewpoints=None, voxels=None, task='train'):
        if task == 'train':
            return self.predict_multiview(images[0], images[1], viewpoints[0], viewpoints[1])
        elif task == 'test':
            return self.evaluate_iou(images, voxels)
