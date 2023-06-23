from __future__ import annotations
import os
import re
import glob
import cv2
import json
from tqdm import tqdm
import logging as log
from typing import Callable, List, Dict, Union
import numpy as np
import torch
import torchvision
from torch.multiprocessing import Pool
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.ops.image import resize_mip, load_rgb
from wisp.datasets.base_datasets import MultiviewDataset
from wisp.datasets.batch import MultiviewBatch


from kaolin.render.camera.intrinsics import CameraFOV

import time


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, mask=None, mask_hands=None, depth=None, rays_per_sample=2048):
        
        self.mask_idx = self.mask_hands_idx = self.depth_idx = -1

        if mask is not None:
            self.mask_idx = imgs.shape[-1]
            imgs = torch.cat(([imgs, mask]), dim=-1)

        if mask_hands is not None:
            self.mask_hands_idx = imgs.shape[-1]
            imgs = torch.cat(([imgs, mask_hands]), dim=-1)

        if depth is not None:
            self.depth_idx = imgs.shape[-1]
            imgs = torch.cat(([imgs, depth]), dim=-1)
        

        self.num_imgs = imgs.shape[0]
        self.points = imgs.pin_memory() 
        self.rays_per_sample = rays_per_sample
        self.h = imgs.shape[1]
        self.w = imgs.shape[2]

    def __len__(self):
        return self.num_imgs
    
    def __getitem__(self, idx, num_rays=None, reject=False):

        img = self.points[idx].cuda(non_blocking=True)
        
        if num_rays is None:
            num_rays = self.rays_per_sample
        
        pos_y = torch.randint(0,img.shape[0],(num_rays,),dtype=torch.int64, device='cuda')
        pos_x = torch.randint(0,img.shape[1],(num_rays,),dtype=torch.int64, device='cuda')

        point = img[pos_y,pos_x]
        data = {'rgb':point[...,:3]}
        
        if self.mask_idx>-1:
            data['mask'] = point[...,self.mask_idx:self.mask_idx+1]

        if self.mask_hands_idx>-1:
            data['mask_hands'] = point[...,self.mask_hands_idx:self.mask_hands_idx+1]

        if self.depth_idx>-1:
            data['depth'] = point[...,self.depth_idx:self.depth_idx+1]


        return (data, 
                pos_x.type(torch.float32)+0.5, 
                pos_y.type(torch.float32)+0.5, 
                torch.empty(num_rays, dtype=torch.int64, device='cuda').fill_(idx))






class DatasetLoader():

    def __init__(self,):
        pass

    def load(self, dataset_path: str, 
                split: str=None, 
                bg_color: str='black', 
                with_mask: bool=False,
                with_mask_hands: bool=False,
                with_depth: bool=False,
                mip: int = 0,
                dataset_num_workers: int = -1):

        self.split = split
        self.dataset_num_workers = dataset_num_workers
        self.mip = mip
        self.with_mask = with_mask
        self.with_mask_hands = with_mask_hands
        self.with_depth = with_depth
        self.bg_color = bg_color

        self.coords = self.data = self.coords_center = self.coords_scale = None

        self.dataset_path = dataset_path
        self._transform_file = self._validate_and_find_transform()

        if self.dataset_num_workers > 0:
            return self.load_multiprocess()
        else:
            return self.load_singleprocess()


    def _validate_and_find_transform(self) -> str:
        """
        Validates the file structure and returns the filename for the dataset's split / transform.
        There are two pairs of standard file structures this dataset can parse:

        ```
        /path/to/dataset/transform.json
        /path/to/dataset/images/____.png
        ```

        or

        ```
        /path/to/dataset/transform_{split}.json
        /path/to/dataset/{split}/_____.png
        ```

        In the former case, the single transform file is assumed to be loaded as a train set,
        for the latter split is assumed to be any of: 'train', 'val', 'test'.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"NeRF dataset path does not exist: {self.dataset_path}")

        transforms = sorted(glob.glob(os.path.join(self.dataset_path, "*.json")))
        if len(transforms) == 0:
            raise RuntimeError(f"NeRF dataset folder has no transform *.json files with camera data: {self.dataset_path}")
        elif len(transforms) > 3 or len(transforms) == 2:
            raise RuntimeError(f"NeRF dataset folder has an unsupported number of splits, "
                               f"there should be ['test', 'train', 'val'], but found: {transforms}.")
        transform_dict = {}
        if len(transforms) == 1:
            transform_dict['train'] = transforms[0]
        elif len(transforms) == 3:
            fnames = [os.path.basename(transform) for transform in transforms]

            # Create dictionary of split to file path, probably there is simpler way of doing this
            for _split in ['test', 'train', 'val']:
                for i, fname in enumerate(fnames):
                    if _split in fname:
                        transform_dict[_split] = transforms[i]

        if self.split not in transform_dict:
            log.warning(
                f"WARNING: Split type ['{self.split}'] does not exist in the dataset. Falling back to train data.")
            self.split = 'train'
        return transform_dict[self.split]
    

    @staticmethod
    def _load_single_entry(frame, root, mip=None, with_mask=False, with_mask_hands=False, with_depth=False):
        """ Loads a single image: takes a frame from the JSON to load image and associated poses from json.
        This is a helper function which also supports multiprocessing for the standard dataset.

        Args:
            root (str): The root of the dataset.
            frame (dict): The frame object from the transform.json. The frame contains the metadata.
            mip (int): Optional, If set, rescales the image by 2**mip.

        Returns:
            (dict): Dictionary of the image and pose.
        """
        fpath = os.path.join(root, frame['file_path'].replace("\\", "/"))
        
        basename = os.path.basename(os.path.splitext(fpath)[0])

        if with_mask:
            mask_file = [name for name in os.listdir(os.path.join(root, 'masks')) if name.split('.')[0]==basename][0]
            mask_path = os.path.join(root, 'masks', mask_file)
        if with_mask_hands:
            mask_hands_file = [name for name in os.listdir(os.path.join(root, 'masks_hands')) if name.split('.')[0]==basename][0]
            mask_hands_path = os.path.join(root, 'masks_hands', mask_hands_file)
        if with_depth:
            depth_file = [name for name in os.listdir(os.path.join(root, 'depths')) if name.split('.')[0]==basename][0]
            depth_path = os.path.join(root, 'depths', depth_file)


        mask = None
        mask_hands = None
        depth = None
        
        if os.path.splitext(fpath)[1] == "":
            # Assume PNG file if no extension exists... the NeRF synthetic data follows this convention.
            fpath += '.png'

        # For some reason instant-ngp allows missing images that exist in the transform but not in the data.
        # Handle this... also handles the above case well too.
        if os.path.exists(fpath):
            img = load_rgb(fpath)
            if mip is not None:
                img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)

            # Load mask
            if with_mask and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mip is not None:
                    mask = resize_mip(mask, mip, interpolation=cv2.INTER_AREA)
                mask = torch.FloatTensor(mask > (mask.max()+mask.min())/2)

            if with_mask_hands and os.path.exists(mask_hands_path):
                mask_hands = cv2.imread(mask_hands_path, cv2.IMREAD_GRAYSCALE)
                if mip is not None:
                    mask_hands = resize_mip(mask_hands, mip, interpolation=cv2.INTER_AREA)
                mask_hands = torch.FloatTensor(mask_hands > (mask_hands.max()+mask_hands.min())/2)

            # Load depth map
            if with_depth and os.path.exists(depth_path):
                depth = load_rgb(depth_path)
                if mip is not None:
                    depth = resize_mip(depth, mip, interpolation=cv2.INTER_AREA)
                depth = torch.FloatTensor(depth)

            return dict(basename=basename,
                        img=torch.FloatTensor(img), 
                        pose=torch.FloatTensor(np.array(frame['transform_matrix'])), 
                        mask=mask,
                        mask_hands=mask_hands,
                        depth=depth)
        else:
            # log.info(f"File name {fpath} doesn't exist. Ignoring.")
            return None
        
    
    def load_singleprocess(self):
        """Standard parsing function for loading nerf-synthetic files on the main process.
        This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

        Returns:
            (dict of torch.FloatTensors): Channels of information from NeRF:
                - 'rays': a list of ray packs, each entry corresponds to a single camera view
                - 'rgb', 'masks': a list of torch.Tensors, each entry corresponds to a single gt image
                - 'cameras': a list of Camera objects, one camera per view
        """
        with open(self._transform_file, 'r') as f:
            metadata = json.load(f)

        imgs = []
        poses = []
        basenames = []
        masks = None
        masks_hands = None
        depths = None

        if self.with_mask:
            masks = []
        if self.with_mask_hands:
            masks_hands = []
        if self.with_depth:
            depths = []

        for frame in tqdm(metadata['frames'], desc='loading data'):
            _data = self._load_single_entry(frame, self.dataset_path, 
                                            mip=self.mip, 
                                            with_mask=self.with_mask, 
                                            with_mask_hands = self.with_mask_hands,
                                            with_depth=self.with_depth)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])
                if _data["mask"] is not None:
                    masks.append(_data["mask"])
                if _data["mask_hands"] is not None:
                    masks_hands.append(_data["mask_hands"])
                if _data["depth"] is not None:
                    depths.append(_data["depth"])
                assert not(self.with_mask and _data["mask"] is None), f"Error in Dataset: mask of image {_data['basename']} is missing from the dataset."
                assert not(self.with_depth and _data["depth"] is None), f"Error in Dataset: depth map of image {_data['basename']} is missing from the dataset."
            
        return self._collect_data_entries(metadata=metadata, 
                                          basenames=basenames, 
                                          imgs=imgs, 
                                          poses=poses,
                                          masks=masks,
                                          masks_hands = masks_hands,
                                          depths=depths)
    
    @staticmethod
    def _parallel_load_standard_imgs(args):
        """ Internal function used by the multiprocessing loader: allocates a single entry task for a worker.
        """
        torch.set_num_threads(1)
        result = DatasetLoader._load_single_entry(args['frame'], args['root'], 
                                                  mip=args['mip'], 
                                                  with_mask=args['with_mask'],
                                                  with_mask_hands=args['with_mask_hands'],
                                                  with_depth=args['with_depth'])
        if result is None:
            return dict(basename=None, img=None, pose=None)
        else:
            return dict(basename=result['basename'], 
                        img=result['img'], 
                        pose=result['pose'],
                        masks=result['masks'],
                        masks_hands=result['masks_hands'],
                        depths=result['depths'])

    def load_multiprocess(self):
        """Standard parsing function for loading nerf-synthetic files with multiple workers.
        This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

        Returns:
            (dict of torch.FloatTensors): Channels of information from NeRF:
                - 'rays': a list of ray packs, each entry corresponds to a single camera view
                - 'rgb', 'masks': a list of torch.Tensors, each entry corresponds to a single gt image
                - 'cameras': a list of Camera objects, one camera per view
        """
        with open(self._transform_file, 'r') as f:
            metadata = json.load(f)

        imgs = []
        poses = []
        basenames = []
        masks = None
        masks_hands = None
        depths = None

        if self.with_mask:
            masks = []
        if self.with_mask_hands:
            masks_hands = []
        if self.with_depth:
            depths = []

        p = Pool(self.dataset_num_workers)
        try:
            mp_entries = [dict(frame=frame, root=self.dataset_path, mip=self.mip, mask=self.mask)
                          for frame in metadata['frames']]
            iterator = p.imap(DatasetLoader._parallel_load_standard_imgs, mp_entries)

            for _ in tqdm(range(len(metadata['frames']))):
                result = next(iterator)
                if result['basename'] is not None:
                    basenames.append(result['basename'])
                if result['img'] is not None:
                    imgs.append(result['img'])
                if result['pose'] is not None:
                    poses.append(result['pose'])
                if result['masks'] is not None:
                    masks.append(result['masks'])
                if result['masks_hands'] is not None:
                    masks_hands.append(result['masks_hands'])
                if result['depth'] is not None:
                    depths.append(result['depth'])
                assert not(self.with_mask and result['masks'] is None), f"Error in Dataset: mask of image {result['basename']} is missing from the dataset."
                assert not(self.with_depth and result['depth'] is None), f"Error in Dataset: depth map of image {result['basename']} is missing from the dataset."
        finally:
            p.close()
            p.join()

        return self._collect_data_entries(metadata=metadata, 
                                          basenames=basenames, 
                                          imgs=imgs, 
                                          poses=poses,
                                          masks=masks,
                                          masks_hands=masks_hands,
                                          depths=depths)
    


    def _collect_data_entries(self, metadata, basenames, imgs, poses, masks, masks_hands, depths) -> Dict[str, Union[torch.Tensor, Rays, Camera]]:
        """ Internal function for aggregating the pre-loaded multi-views.
        This function will:
            1. Read the metadata & compute the intrinsic parameters of the camera view, (such as fov and focal length
                i.e., in the case of a pinhole camera).
            2. Apply various scaling and offsets transformations to the extrinsics,
                as specified by the metadata by parameters
               such as 'scale', 'offset' and 'aabb_scale'
            3. Create kaolin Camera objects out of the computed extrinsics and intrinsics.
            4. Invoke ray generation on each camera view.
            5. Stack the images pixel values and rays as per-view information entries.
        """

        # not implemented warnings
        if 'fix_premult' in metadata:
            log.info("WARNING: The dataset expects premultiplied alpha correction, "
                     "but the current implementation does not handle this.")

        if 'k1' in metadata:
            log.info \
                ("WARNING: The dataset expects distortion correction, but the current implementation does not handle this.")

        if 'rolling_shutter' in metadata:
            log.info("WARNING: The dataset expects rolling shutter correction,"
                     "but the current implementation does not handle this.")

        
        imgs = torch.stack(imgs)
        poses = torch.stack(poses)

        if masks is not None:
            masks = torch.stack(masks)[..., None]
        else:
            masks = torch.ones_like(imgs)[:, :, :, :1]
        if masks_hands is not None:
            masks_hands = torch.stack(masks_hands)[..., None]
        else:
            masks_hands = torch.ones_like(imgs)[:, :, :, :1]
        if depths is not None:
            depths = torch.stack(depths)[..., None]
        else:
            depths = torch.ones_like(imgs)[:, :, :, :1]

        h, w = imgs[0].shape[:2]

        # compute scaling factors
        if 'x_fov' in metadata:
            # Degrees
            x_fov = metadata['x_fov']
            fx = (0.5 * w) / np.tan(0.5 * float(x_fov) * (np.pi / 180.0))
            if 'y_fov' in metadata:
                y_fov = metadata['y_fov']
                fy = (0.5 * h) / np.tan(0.5 * float(y_fov) * (np.pi / 180.0))
            else:
                fy = fx
        elif 'fl_x' in metadata:
            fx = float(metadata['fl_x']) / float(2**self.mip)
            if 'fl_y' in metadata:
                fy = float(metadata['fl_y']) / float(2**self.mip)
            else:
                fy = fx
        elif 'camera_angle_x' in metadata:
            # Radians
            camera_angle_x = metadata['camera_angle_x']
            fx = (0.5 * w) / np.tan(0.5 * float(camera_angle_x))

            if 'camera_angle_y' in metadata:
                camera_angle_y = metadata['camera_angle_y']
                fy = (0.5 * h) / np.tan(0.5 * float(camera_angle_y))
            else:
                fy = fx
        else:
            fx = 0.0
            fy = 0.0


        # The principal point in wisp are always a displacement in pixels from the center of the image.
        x0 = 0.0
        y0 = 0.0



        # # The standard dataset generally stores the absolute location on the image to specify the principal point.
        # # Thus, we need to scale and translate them such that they are offsets from the center.
        # poses_ = poses.clone()
        # if 'cx' in metadata:
        #     x0_ = (float(metadata['cx']) / (2**self.mip)) - (w//2)
        # if 'cy' in metadata:
        #     y0_ = (float(metadata['cy']) / (2**self.mip)) - (h//2)

        # offset_ = metadata['offset'] if 'offset' in metadata else [0 ,0 ,0]
        # scale_ = metadata['scale'] if 'scale' in metadata else 1.0
        # aabb_scale_ = metadata['aabb_scale'] if 'aabb_scale' in metadata else 1.25

        # # TODO(ttakikawa): Actually scale the AABB instead? Maybe
        # poses_[..., :3, 3] /= aabb_scale_
        # poses_[..., :3, 3] *= scale_
        # poses_[..., :3, 3] += torch.FloatTensor(offset_)

        # # nerf-synthetic uses a default far value of 6.0
        # default_far_ = 6.0

        # rays_ = []

        # cameras_ = dict()
        # for i in range(1):
        #     view_matrix_ = torch.zeros_like(poses_[i])
        #     view_matrix_[:3, :3] = poses_[i][:3, :3].T
        #     view_matrix_[:3, -1] = torch.matmul(-view_matrix_[:3, :3], poses_[i][:3, -1])
        #     view_matrix_[3, 3] = 1.0
        #     camera = Camera.from_args(view_matrix=view_matrix_,
        #                               focal_x=fx,
        #                               focal_y=fy,
        #                               width=w,
        #                               height=h,
        #                               far=default_far_,
        #                               near=0.0,
        #                               x0=x0_,
        #                               y0=y0_,
        #                               dtype=torch.float64)
        #     camera.change_coordinate_system(blender_coords())
        #     cameras_[basenames[i]] = camera
        #     coords_grid_ = generate_centered_pixel_coords(camera.width, camera.height,
        #                                               camera.width, camera.height, device='cuda')
        #     pixel_y_, pixel_x_ = coords_grid_
        #     pixel_x_ = pixel_x_.to(camera.device, camera.dtype)
        #     pixel_y_ = pixel_y_.to(camera.device, camera.dtype)

        #     # Account for principal point (offsets from the center)
        #     pixel_x_1 = pixel_x_ - camera.x0
        #     pixel_y_1 = pixel_y_ + camera.y0

        #     # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
        #     pixel_x_2 = 2 * (pixel_x_1 / camera.width) - 1.0
        #     pixel_y_2 = 2 * (pixel_y_1 / camera.height) - 1.0

        #             # = 2* ( pixel_x_ - camera.x0)/ w - 1.0
        #             # = 2* ( pixel_x_ - (cx / (2**self.mip) + (w/2)))/ w - 1.0
        #             # = 2* ( pixel_x_ - (cx / (2**self.mip)) -(w/2))/ w - 1.0
        #             # = 2* ( pixel_x_ - (cx / (2**self.mip)) )/ w -2



        #     ray_dir_ = torch.stack((pixel_x_2 * camera.tan_half_fov(CameraFOV.HORIZONTAL),
        #                         -pixel_y_2 * camera.tan_half_fov(CameraFOV.VERTICAL),
        #                         -torch.ones_like(pixel_x_)), dim=-1)

        #     ray_dir_ = ray_dir_.reshape(-1, 3)    # Flatten grid rays to 1D array
        #     ray_orig_ = torch.zeros_like(ray_dir_)

        #     # Transform from camera to world coordinates
        #     ray_orig_f, ray_dir_f = camera.extrinsics.inv_transform_rays(ray_orig_, ray_dir_)
        #     ray_dir_f /= torch.linalg.norm(ray_dir_f, dim=-1, keepdim=True)
        #     ray_orig_e, ray_dir_e = ray_orig_f[0], ray_dir_f[0]  # Assume a single camera



        if 'cx' in metadata:
            x0 = (float(metadata['cx']) / (2**self.mip))
        if 'cy' in metadata:
            y0 = (float(metadata['cy']) / (2**self.mip))

        offset = metadata['offset'] if 'offset' in metadata else [0 ,0 ,0]
        scale = metadata['scale'] if 'scale' in metadata else 1.0
        aabb_scale = metadata['aabb_scale'] if 'aabb_scale' in metadata else 1.25

        # scale position of the cameras based on the aabb scale
        poses[..., :3, 3] /= aabb_scale
        poses[..., :3, 3] *= scale
        poses[..., :3, 3] += torch.FloatTensor(offset)


        # transf_matrix = torch.tensor([[1, 0, 0],
        #                             [0, 0, 1],
        #                             [0, -1, 0]]).float()
        # poses[:,:3,:3]= transf_matrix @ poses[:,:3,:3]
        # poses


        # for i in range(1):
        #     ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
        #                                               camera.width, camera.height, device='cpu')
            
        #     ray_orig = poses[ np.ones(w*h)*i, :3, -1] @ transf_matrix.T
        #     pixel_x = 2 * ( ((ray_grid[1].flatten() - x0 )/ w)) 
        #     pixel_y = 2 * ( ((ray_grid[0].flatten() + y0 )/ h)) - 2

        #     ray_dir0 = torch.stack((  pixel_x * (w / 2.0) / fx,
        #                             -pixel_y * (h / 2.0) / fy,
        #                             -torch.ones_like(pixel_x)), dim=-1)
            
        #     rotation_matrix = poses[ np.ones(w*h)*i, :3, :3]

        #     ray_dir = (rotation_matrix @ ray_dir0[...,None])[:,:,0]
        #     ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

        # modify colors if they are transparent (png)
        rgbs = imgs[... ,:3]
        alpha = imgs[... ,3:4]

        if alpha.numel() == 0:
            alpha_masks = torch.ones_like(rgbs[... ,0:1]).bool()
        else:
            alpha_masks = (alpha > 0.5).bool()

            if self.bg_color == 'black':
                rgbs[... ,:3] -= ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)
            else:
                rgbs[... ,:3] *= alpha
                rgbs[... ,:3] += ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)

        
        return {"cameras":{"fx": fx, "fy": fy, "cx": x0, "cy": y0, "width": w, "height": h, "poses":poses}, 
                "rgb": rgbs, "masks": masks, "masks_hands":masks_hands, "depths": depths}
    



