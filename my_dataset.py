from typing import Callable
import torch
from torch.utils.data import Dataset
#from wisp.datasets.formats import load_nerf_standard_data, load_rtmv_data
from load_data import load_nerf_standard_data, load_rtmv_data
from wisp.core import Rays
from kaolin.render.camera.intrinsics import CameraFOV
from wisp.ops.raygen import generate_pinhole_rays
import numpy as np
from scipy import interpolate


class MyMultiviewDataset(Dataset):
    """This is a static multiview image dataset class.

    This class should be used for training tasks where the task is to fit a static 3D volume from
    multiview images.

    TODO(ttakikawa): Support single-camera dynamic temporal scenes, and multi-camera dynamic temporal scenes.
    TODO(ttakikawa): Currently this class only supports sampling per image, not sampling across the entire
                     dataset. This is due to practical reasons. Not sure if it matters...
    """

    def __init__(self, 
        dataset_path             : str,
        aabb_scale               : int      = 2,
        multiview_dataset_format : str      = 'standard',
        mip                      : int      = None,
        bg_color                 : str      = None,
        dataset_num_workers      : int      = -1,
        transform                : Callable = None,
        num_samples              : int       = 1024,
        **kwargs
    ):
        """Initializes the dataset class.

        Note that the `init` function to actually load images is separate right now, because we don't want 
        to load the images unless we have to. This might change later.

        Args: 
            dataset_path (str): Path to the dataset.
            multiview_dataset_format (str): The dataset format. Currently supports standard (the same format
                used for instant-ngp) and the RTMV dataset.
            mip (int): The factor at which the images will be downsampled by to save memory and such.
                       Will downscale by 2**mip.
            bg_color (str): The background color to use for images with 0 alpha.
            dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.root = dataset_path
        self.aabb_scale = aabb_scale
        self.multiview_dataset_format = multiview_dataset_format
        self.mip = mip
        self.bg_color = bg_color
        self.dataset_num_workers = dataset_num_workers
        self.transform = transform
        self.num_samples = num_samples

        self.coords = self.data = self.img_shape = self.num_imgs = self.coords_center = self.coords_scale = None
        self.init()

    def init(self):
        """Initializes the dataset. """
        # Get image tensors
        self.coords = None
        self.data = self.get_images()
        self.img_shape = self.data["imgs"].shape[1:3]
        self.num_imgs = self.data["imgs"].shape[0]

        self.data["imgs"] = self.data["imgs"]
        #self.data["rays"] = self.data["rays"].reshape(self.num_imgs, -1, 3)
        if "depths" in self.data:
            self.data["depths"] = self.data["depths"].reshape(self.num_imgs, -1, 1)
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(self.num_imgs, -1, 1)

    def get_images(self, split='train', mip=None):
        """Will return the dictionary of image tensors.

        Args:
            split (str): The split to use from train, val, test
            mip (int): If specified, will rescale the image by 2**mip.

        Returns:
            (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
        """
        if mip is None:
            mip = self.mip
        
        if self.multiview_dataset_format == "standard":
            data = load_nerf_standard_data(self.root, self.aabb_scale, split,
                                            bg_color=self.bg_color, num_workers=self.dataset_num_workers, mip=self.mip)
        elif self.multiview_dataset_format == "rtmv":
            if split == 'train':
                data = load_rtmv_data(self.root, split,
                                      return_pointcloud=True, mip=mip, bg_color=self.bg_color,
                                      normalize=True, num_workers=self.dataset_num_workers)
                self.coords = data["coords"]
                self.coords_center = data["coords_center"]
                self.coords_scale = data["coords_scale"]
            else:
                if self.coords is None:
                    assert False and "Initialize the dataset first with the training data!"
                
                data = load_rtmv_data(self.root, split,
                                      return_pointcloud=False, mip=mip, bg_color=self.bg_color,
                                      normalize=False)
                
                data["depths"] = data["depths"] * self.coords_scale
                data["rays"].origins = (data["rays"].origins - self.coords_center) * self.coords_scale

        return data

    def __len__(self):
        """Length of the dataset in number of rays.
        """
        return self.data["imgs"].shape[0]

    def __getitem__(self, idx : int):
        """Returns a ray.
        """
        camera = self.data['cameras'][idx]
        pixel_x = torch.rand(self.num_samples) * 2 - 1 
        pixel_y = torch.rand(self.num_samples) * 2 - 1
        ray_dir = torch.stack((pixel_x * camera.tan_half_fov(CameraFOV.HORIZONTAL),
                              -pixel_y * camera.tan_half_fov(CameraFOV.VERTICAL),
                              -torch.ones_like(pixel_x)), dim=-1)

        ray_dir = ray_dir.reshape(-1, 3)    
        ray_orig = torch.zeros_like(ray_dir)

        ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
        ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
        ray_orig, ray_dir = ray_orig[0], ray_dir[0]  

        rays = Rays(origins=ray_orig, dirs=ray_dir, dist_min=camera.near, dist_max=camera.far)
        rays = rays.to('cpu').to(dtype=torch.float)

        out = {}
        out['rays'] = rays
        out['imgs'] = self.bilinear_interpolation(pixel_x, pixel_y, self.data['imgs'][idx])

        if self.transform is not None:
            out = self.transform(out)

        out['idx'] = idx
        return out

    @staticmethod
    def bilinear_interpolation(x, y, image):        
        x = x * (image.shape[0] - 1)
        y = y * (image.shape[1] - 1)
        x_floor = torch.floor(x).to(dtype=torch.int64)
        x_ceil = torch.ceil(x).to(dtype=torch.int64)
        y_floor = torch.floor(y).to(dtype=torch.int64)
        y_ceil = torch.ceil(y).to(dtype=torch.int64)

        c0 = image[x_floor, y_floor]
        c1 = image[x_floor, y_ceil]
        c2 = image[x_ceil, y_floor]
        c3 = image[x_ceil, y_ceil]

        r1 = torch.frac(x)[:, None]
        r2 = torch.frac(y)[:, None]

        c = (1-r1)*(1-r2)*c0 + (1-r1)*(r2)*c1
        c += (r1)*(1-r2)*c2 + (r1)*(r2)*c3
        return c
    
    def get_img_samples(self, idx, batch_size):
        """Returns a batch of samples from an image, indexed by idx.
        """
        camera = self.data['cameras'][idx]
        pixel_x = torch.rand(batch_size) * 2 - 1 
        pixel_y = torch.rand(batch_size) * 2 - 1
        ray_dir = torch.stack((pixel_x * camera.tan_half_fov(CameraFOV.HORIZONTAL),
                              -pixel_y * camera.tan_half_fov(CameraFOV.VERTICAL),
                              -torch.ones_like(pixel_x)), dim=-1)

        ray_dir = ray_dir.reshape(-1, 3)    
        ray_orig = torch.zeros_like(ray_dir)

        ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
        ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
        ray_orig, ray_dir = ray_orig[0], ray_dir[0]  

        rays = Rays(origins=ray_orig, dirs=ray_dir, dist_min=camera.near, dist_max=camera.far)
        rays = rays.to('cpu').to(dtype=torch.float)

        out = {}
        out['rays'] = rays
        out['imgs'] = self.bilinear_interpolation(pixel_x, pixel_y, self.data['imgs'][idx])

        if self.transform is not None:
            out = self.transform(out)

        out['idx'] = idx
        return out
