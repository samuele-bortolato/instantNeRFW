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
    def __init__(self, imgs, mask=None, depth=None, rays_per_sample=2048):
        
        if mask is not None:
            self.with_mask=True
            imgs = torch.cat(([imgs, mask]), dim=-1)
        else:
            self.with_mask=False

        if depth is not None:
            self.with_depth=True
            imgs = torch.cat(([imgs, depth]), dim=-1)
        else:
            self.with_depth=False

        self.num_imgs = imgs.shape[0]
        self.points = imgs
        self.rays_per_sample = rays_per_sample
        self.h = imgs.shape[1]
        self.w = imgs.shape[2]

    def __len__(self):
        return self.num_imgs
    
    def __getitem__(self, idx, num_rays=None, reject=False):

        img = self.points[idx].cuda()
        
        if num_rays is None:
            num_rays = self.rays_per_sample
        
        pos_y = torch.randint(0,img.shape[0],(num_rays,),dtype=torch.int64, device='cuda')
        pos_x = torch.randint(0,img.shape[1],(num_rays,),dtype=torch.int64, device='cuda')

        if self.with_mask and reject:
            wrong = img[pos_y,pos_x][...,3] != 1
            while wrong.sum()>0:
                pos_y[wrong] = torch.randint(0,img.shape[0],(wrong.sum(),),dtype=torch.int64, device='cuda')
                pos_x[wrong] = torch.randint(0,img.shape[1],(wrong.sum(),),dtype=torch.int64, device='cuda')
                wrong = img[pos_y,pos_x][...,3] != 1

        point = img[pos_y,pos_x]
        data = {'rgb':point[...,:3]}
        
        if self.with_mask:
            data['mask'] = point[...,3:4]
            if self.with_depth:
                data['depth'] = point[...,4:5]
        else:
            if self.with_mask:
                data['depth'] = point[...,3:4]

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
                with_depth: bool=False,
                mip: int = 0,
                dataset_num_workers: int = -1):

        self.split = split
        self.dataset_num_workers = dataset_num_workers
        self.mip = mip
        self.with_mask = with_mask
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
    def _load_single_entry(frame, root, mip=None, with_mask=False, with_depth=False):
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
        if with_depth:
            depth_file = [name for name in os.listdir(os.path.join(root, 'depths')) if name.split('.')[0]==basename][0]
            depth_path = os.path.join(root, 'depths', depth_file)


        mask = None
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
        depths = None

        if self.with_mask:
            masks = []
        if self.with_depth:
            depths = []

        for frame in tqdm(metadata['frames'], desc='loading data'):
            _data = self._load_single_entry(frame, self.dataset_path, 
                                            mip=self.mip, 
                                            with_mask=self.with_mask, 
                                            with_depth=self.with_depth)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])
                if _data["mask"] is not None:
                    masks.append(_data["mask"])
                if _data["depth"] is not None:
                    depths.append(_data["depth"])
                assert not(self.with_mask and _data["mask"] is None), f"Error in Dataset: mask of image {_data['basename']} is missing from the dataset."
                assert not(self.with_depth and _data["depth"] is None), f"Error in Dataset: depth map of image {_data['basename']} is missing from the dataset."
            
        return self._collect_data_entries(metadata=metadata, 
                                          basenames=basenames, 
                                          imgs=imgs, 
                                          poses=poses,
                                          masks=masks,
                                          depths=depths)
    
    @staticmethod
    def _parallel_load_standard_imgs(args):
        """ Internal function used by the multiprocessing loader: allocates a single entry task for a worker.
        """
        torch.set_num_threads(1)
        result = DatasetLoader._load_single_entry(args['frame'], args['root'], 
                                                  mip=args['mip'], 
                                                  with_mask=args['with_mask'],
                                                  with_depth=args['with_depth'])
        if result is None:
            return dict(basename=None, img=None, pose=None)
        else:
            return dict(basename=result['basename'], 
                        img=result['img'], 
                        pose=result['pose'],
                        masks=result['masks'],
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
        depths = None

        if self.with_mask:
            masks = []
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
                                          depths=depths)
    


    def _collect_data_entries(self, metadata, basenames, imgs, poses, masks, depths) -> Dict[str, Union[torch.Tensor, Rays, Camera]]:
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
                "rgb": rgbs, "masks": masks, "depths": depths}
    






class NeRFSyntheticDataset(MultiviewDataset):
    """ A dataset for files in the standard NeRF format, including extensions to the format
        supported by Instant Neural Graphics Primitives.
        See: https://github.com/NVlabs/instant-ngp
        NeRF-synthetic scenes include RGBA / RGB information.
    """

    def __init__(self, dataset_path: str, split: str, bg_color: str, mip: int = 0,
                 dataset_num_workers: int = -1, transform: Callable = None, num_samples=2**12):
        """ Loads the NeRF-synthetic data and applies dataset specific transforms required for compatibility with the
        framework.
        The loaded data is cached inside the `data` field.

        Args:
            dataset_path (str): The root directory of the dataset, where images and json files of a single multiview
                scene reside.
            split (str): The dataset split to use, corresponding to the transform file to load, when splits are
                available. In case of a single transform file, it will be considered as a single split of
                'train' by default.
                Options: 'train', 'val', 'test'.
            bg_color (str): The background color to use for when alpha=0.
                Options: 'black', 'white'.
            mip (int): If provided, will rescale images by 2**mip. Useful when large images are loaded.
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                If dataset_num_workers < 1, processing will take place on the main process.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        super().__init__(dataset_path=dataset_path, dataset_num_workers=dataset_num_workers,
                         transform=transform, split=split)
        self.mip = mip
        self.bg_color = bg_color

        self.coords = self.data = self.coords_center = self.coords_scale = None
        self._transform_file = self._validate_and_find_transform()
        self.data = self.load()

        self._img_shape = self.data["rgb"].shape[1:3]
        self.flatten_tensors()

        self.num_samples=num_samples

    def create_split(self, split: str, transform: Callable = None) -> NeRFSyntheticDataset:
        """ Creates a dataset with the same parameters and a different split.
        This is a convenient way of creating validation and test datasets, while making sure they're compatible
        with the train dataset.

        All settings except for split and transform will be copied from the current dataset.

        Args:
            split (str): The dataset split to use, corresponding to the transform file to load.
                Options: 'train', 'val', 'test'.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        return NeRFSyntheticDataset(
            dataset_path=self.dataset_path,
            split=split,
            bg_color=self.bg_color,
            mip=self.mip,
            dataset_num_workers=self.dataset_num_workers,
            transform=transform
        )

    def __getitem__(self, idx) -> MultiviewBatch:
        """Retrieve a batch of rays and their corresponding values.
        Rays are precomputed from the dataset's cameras, and are cached within the dataset.
        By default, rays are assumed to have corresponding rgb values, sampled from the dataset's images.

        Returns:
            (MultiviewBatch): A batch of rays and their rgb values. The fields can be accessed as a dictionary:
                "rays" - a wisp.core.Rays pack of ray origins and directions, pre-generated from the dataset camera.
                "rgb" - a torch.Tensor of rgb color which corresponds the gt image's pixel each ray intersects.
                "masks" - a torch.BoolTensor specifying if the ray hits a dense area or not.
                 This is estimated from the alpha channel of the gt image, where mask=True if alpha > 0.5.
        """
        out = dict(
            pos_x=self.data["pos_x"][idx],
            pos_y=self.data["pos_y"][idx],
            #rays=self.data["rays"][idx],
            rgb=self.data["rgb"][idx],
            masks=self.data["masks"][idx],
            idx=torch.ones(len(self.data["masks"][idx])).long()*idx
        )

        #if self.transform is not None:
        #    out = self.transform(out)

        device = self.data["pos_x"].device
        ray_idx = torch.randint(0, self.data["pos_x"][idx].shape[0], [self.num_samples], device=device)

        # Loop over ray values in this batch
        for channel_name, ray_value in out.items():
            out[channel_name] = ray_value[ray_idx].contiguous()

        return out

    @classmethod
    def is_root_of_dataset(cls, root: str, files_list: List[str]) -> bool:
        """ Each dataset may implement a simple set of rules to distinguish it from other datasets.
        Rules should be unique for this dataset type, such that given a general root path, Wisp will know
        to associate it with this dataset class.

        Datasets which don't implement this function should be created explicitly.

        Args:
                root (str): A path to the root directory of the dataset.
                files_list (List[str]): List of files within the dataset root, without their prefix path.
        Returns:
                True if the root folder points to content loadable by this dataset.
        """
        # NeRF-synthetic data is distinguished by the transform jsons and an additional image folder.
        try:
            regex = re.compile(r"transform.+\.json")
            transform_files = list(filter(regex.match, files_list))
            if 'transforms.json' in transform_files and 'images' in files_list:
                # Single transform file
                images_path = os.path.join(root, 'images')
                return os.path.isdir(images_path)
            elif 'transforms_train.json' in transform_files and 'train' in files_list:
                # Three transform files, with splits. Check only for train split
                train_path = os.path.join(root, 'train')
                return os.path.isdir(train_path)
            else:
                return False
        except ValueError:
            return False

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
    def _load_single_entry(frame, root, mip=None):
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
        if os.path.splitext(fpath)[1] == "":
            # Assume PNG file if no extension exists... the NeRF synthetic data follows this convention.
            fpath += '.png'

        # For some reason instant-ngp allows missing images that exist in the transform but not in the data.
        # Handle this... also handles the above case well too.
        if os.path.exists(fpath):
            img = load_rgb(fpath)
            if mip is not None:
                img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)
            return dict(basename=basename,
                        img=torch.FloatTensor(img), pose=torch.FloatTensor(np.array(frame['transform_matrix'])))
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

        for frame in tqdm(metadata['frames'], desc='loading data'):
            _data = self._load_single_entry(frame, self.dataset_path, mip=self.mip)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])

        return self._collect_data_entries(metadata=metadata, basenames=basenames, imgs=imgs, poses=poses)

    @staticmethod
    def _parallel_load_standard_imgs(args):
        """ Internal function used by the multiprocessing loader: allocates a single entry task for a worker.
        """
        torch.set_num_threads(1)
        result = NeRFSyntheticDataset._load_single_entry(args['frame'], args['root'], mip=args['mip'])
        if result is None:
            return dict(basename=None, img=None, pose=None)
        else:
            return dict(basename=result['basename'], img=result['img'], pose=result['pose'])

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

        p = Pool(self.dataset_num_workers)
        try:
            mp_entries = [dict(frame=frame, root=self.dataset_path, mip=self.mip)
                          for frame in metadata['frames']]
            iterator = p.imap(NeRFSyntheticDataset._parallel_load_standard_imgs, mp_entries)

            for _ in tqdm(range(len(metadata['frames']))):
                result = next(iterator)
                basename = result['basename']
                img = result['img']
                pose = result['pose']
                if basename is not None:
                    basenames.append(basename)
                if img is not None:
                    imgs.append(img)
                if pose is not None:
                    poses.append(pose)
        finally:
            p.close()
            p.join()

        return self._collect_data_entries(metadata=metadata, basenames=basenames, imgs=imgs, poses=poses)

    def _collect_data_entries(self, metadata, basenames, imgs, poses) -> Dict[str, Union[torch.Tensor, Rays, Camera]]:
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
        imgs = torch.stack(imgs)
        poses = torch.stack(poses)

        # TODO(ttakikawa): Assumes all images are same shape and focal. Maybe breaks in general...
        h, w = imgs[0].shape[:2]

        if 'x_fov' in metadata:
            # Degrees
            x_fov = metadata['x_fov']
            fx = (0.5 * w) / np.tan(0.5 * float(x_fov) * (np.pi / 180.0))
            if 'y_fov' in metadata:
                y_fov = metadata['y_fov']
                fy = (0.5 * h) / np.tan(0.5 * float(y_fov) * (np.pi / 180.0))
            else:
                fy = fx
        elif 'fl_x' in metadata and False:
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

        if 'fix_premult' in metadata:
            log.info("WARNING: The dataset expects premultiplied alpha correction, "
                     "but the current implementation does not handle this.")

        if 'k1' in metadata:
            log.info \
                ("WARNING: The dataset expects distortion correction, but the current implementation does not handle this.")

        if 'rolling_shutter' in metadata:
            log.info("WARNING: The dataset expects rolling shutter correction,"
                     "but the current implementation does not handle this.")

        # The principal point in wisp are always a displacement in pixels from the center of the image.
        x0 = 0.0
        y0 = 0.0
        # The standard dataset generally stores the absolute location on the image to specify the principal point.
        # Thus, we need to scale and translate them such that they are offsets from the center.
        if 'cx' in metadata:
            x0 = (float(metadata['cx']) / (2**self.mip)) - (w//2)
        if 'cy' in metadata:
            y0 = (float(metadata['cy']) / (2**self.mip)) - (h//2)

        offset = metadata['offset'] if 'offset' in metadata else [0 ,0 ,0]
        scale = metadata['scale'] if 'scale' in metadata else 1.0
        aabb_scale = metadata['aabb_scale'] if 'aabb_scale' in metadata else 1.25

        # TODO(ttakikawa): Actually scale the AABB instead? Maybe
        poses[..., :3, 3] /= aabb_scale
        poses[..., :3, 3] *= scale
        poses[..., :3, 3] += torch.FloatTensor(offset)

        # nerf-synthetic uses a default far value of 6.0
        default_far = 6.0

        rays = []

        cameras = dict()
        for i in range(imgs.shape[0]):
            view_matrix = torch.zeros_like(poses[i])
            view_matrix[:3, :3] = poses[i][:3, :3].T
            view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
            view_matrix[3, 3] = 1.0
            camera = Camera.from_args(view_matrix=view_matrix,
                                      focal_x=fx,
                                      focal_y=fy,
                                      width=w,
                                      height=h,
                                      far=default_far,
                                      near=0.0,
                                      x0=x0,
                                      y0=y0,
                                      dtype=torch.float64)
            camera.change_coordinate_system(blender_coords())
            cameras[basenames[i]] = camera
            ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                      camera.width, camera.height, device='cuda')
            rays.append \
                (generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, 3).to
                    ('cpu'))

        rays = Rays.stack(rays).to(dtype=torch.float)

        rgbs = imgs[... ,:3]
        alpha = imgs[... ,3:4]
        if alpha.numel() == 0:
            masks = torch.ones_like(rgbs[... ,0:1]).bool()
        else:
            masks = (alpha > 0.5).bool()

            if self.bg_color == 'black':
                rgbs[... ,:3] -= ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)
            else:
                rgbs[... ,:3] *= alpha
                rgbs[... ,:3] += ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)

        return {"rgb": rgbs, "masks": masks, "rays": rays, "cameras": cameras}

    def flatten_tensors(self) -> None:
        """ Flattens the cached data tensors to (NUM_VIEWS, NUM_RAYS, *).
        """
        num_imgs = len(self)
        self.data["rgb"] = self.data["rgb"].reshape(num_imgs, -1, 3)
        #self.data["pos_x"] = self.data["pos_x"].reshape(num_imgs, -1, 1)
        #self.data["pos_y"] = self.data["pos_y"].reshape(num_imgs, -1, 1)
        self.data["rays"] = self.data["rays"].reshape(num_imgs, -1, 3)
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(num_imgs, -1, 1)

    @property
    def img_shape(self) -> torch.Size:
        """ Returns the shape of the rescaled dataset images (cached values are flattened) """
        return self._img_shape

    @property
    def cameras(self) -> List[Camera]:
        """ Returns the list of camera views used to generate rays for this dataset. """
        return self.data["cameras"]

    @property
    def num_images(self) -> int:
        """ Returns the number of views this dataset stores. """
        return self.data["rgb"].shape[0]
