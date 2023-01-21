import os
import glob
import time
import cv2
import skimage
import imageio
import json
import copy
from tqdm import tqdm
import skimage.metrics
import logging as log
import numpy as np
import torch
from torch.multiprocessing import Pool, cpu_count
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.ops.image import resize_mip, load_exr
from wisp.ops.pointcloud import create_pointcloud_from_images, normalize_pointcloud

#############################################################################################
##################################### STANDARD ##############################################
#############################################################################################

# Local function for multiprocess. Just takes a frame from the JSON to load images and poses.
def _load_standard_imgs(frame, root, mip=None):
    """Helper for multiprocessing for the standard dataset. Should not have to be invoked by users.

    Args:
        root: The root of the dataset.
        frame: The frame object from the transform.json.
        mip: If set, rescales the image by 2**mip.

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
        img = imageio.imread(fpath)
        img = skimage.img_as_float32(img)
        if mip is not None:
            img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)
        return dict(basename=basename,
                    img=torch.FloatTensor(img), pose=torch.FloatTensor(np.array(frame['transform_matrix'])))
    else:
        # log.info(f"File name {fpath} doesn't exist. Ignoring.")
        return None

def _parallel_load_standard_imgs(args):
    """Internal function for multiprocessing.
    """
    torch.set_num_threads(1)
    result = _load_standard_imgs(args['frame'], args['root'], mip=args['mip'])
    if result is None:
        return dict(basename=None, img=None, pose=None)
    else:
        return dict(basename=result['basename'], img=result['img'], pose=result['pose'])

def load_nerf_standard_data(root, aabb_scale, split='train', bg_color='white', num_workers=-1, mip=None):
    """Standard loading function.

    This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

    There are two pairs of standard file structures this follows:

    ```
    /path/to/dataset/transform.json
    /path/to/dataset/images/____.png
    ```

    or

    ```
    /path/to/dataset/transform_{split}.json
    /path/to/dataset/{split}/_____.png
    ```

    Args:
        root (str): The root directory of the dataset.
        split (str): The dataset split to use from 'train', 'val', 'test'.
        bg_color (str): The background color to use for when alpha=0.
        num_workers (int): The number of workers to use for multithreaded loading. If -1, will not multithread.
        mip: If set, rescales the image by 2**mip.

    Returns:
        (dict of torch.FloatTensors): Different channels of information from NeRF.
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"NeRF dataset path does not exist: {root}")

    transforms = sorted(glob.glob(os.path.join(root, "*.json")))

    transform_dict = {}

    if mip is None:
        mip = 0

    if len(transforms) == 1:
        transform_dict['train'] = transforms[0]
    elif len(transforms) == 3:
        fnames = [os.path.basename(transform) for transform in transforms]

        # Create dictionary of split to file path, probably there is simpler way of doing this
        for _split in ['test', 'train', 'val']:
            for i, fname in enumerate(fnames):
                if _split in fname:
                    transform_dict[_split] = transforms[i]
    elif len(transforms) == 0:
        raise RuntimeError(f"NeRF dataset folder has no transform *.json files with camera data: {root}")
    else:
        raise RuntimeError(f"NeRF dataset folder has an unsupported number of splits, "
                           f"there should be ['test', 'train', 'val'], but found: {transforms}.")

    if split not in transform_dict:
        log.warning(f"WARNING: Split type ['{split}'] does not exist in the dataset. Falling back to train data.")
        split = 'train'

    for key in transform_dict:
        with open(transform_dict[key], 'r') as f:
            transform_dict[key] = json.load(f)

    imgs = []
    poses = []
    basenames = []

    if num_workers > 0:
        # threading loading images

        p = Pool(num_workers)
        try:
            iterator = p.imap(_parallel_load_standard_imgs,
                [dict(frame=frame, root=root, mip=mip) for frame in transform_dict[split]['frames']])
            for _ in tqdm(range(len(transform_dict[split]['frames']))):
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
    else:
        for frame in tqdm(transform_dict[split]['frames'], desc='loading data'):
            _data = _load_standard_imgs(frame, root, mip=mip)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])

    imgs = torch.stack(imgs)
    poses = torch.stack(poses)

    # TODO(ttakikawa): Assumes all images are same shape and focal. Maybe breaks in general...
    h, w = imgs[0].shape[:2]

    if 'x_fov' in transform_dict[split]:
        # Degrees
        x_fov = transform_dict[split]['x_fov']
        fx = (0.5 * w) / np.tan(0.5 * float(x_fov) * (np.pi / 180.0))
        if 'y_fov' in transform_dict[split]:
            y_fov = transform_dict[split]['y_fov']
            fy = (0.5 * h) / np.tan(0.5 * float(y_fov) * (np.pi / 180.0))
        else:
            fy = fx
    elif 'fl_x' in transform_dict[split] and False:
        fx = float(transform_dict[split]['fl_x']) / float(2**mip)
        if 'fl_y' in transform_dict[split]:
            fy = float(transform_dict[split]['fl_y']) / float(2**mip)
        else:
            fy = fx
    elif 'camera_angle_x' in transform_dict[split]:
        # Radians
        camera_angle_x = transform_dict[split]['camera_angle_x']
        fx = (0.5 * w) / np.tan(0.5 * float(camera_angle_x))

        if 'camera_angle_y' in transform_dict[split]:
            camera_angle_y = transform_dict[split]['camera_angle_y']
            fy = (0.5 * h) / np.tan(0.5 * float(camera_angle_y))
        else:
            fy = fx

    else:
        fx = 0.0
        fy = 0.0

    if 'fix_premult' in transform_dict[split]:
        log.info("WARNING: The dataset expects premultiplied alpha correction, "
                 "but the current implementation does not handle this.")

    if 'k1' in transform_dict[split]:
        log.info \
            ("WARNING: The dataset expects distortion correction, but the current implementation does not handle this.")

    if 'rolling_shutter' in transform_dict[split]:
        log.info("WARNING: The dataset expects rolling shutter correction,"
                 "but the current implementation does not handle this.")

    # The principal point in wisp are always a displacement in pixels from the center of the image.
    x0 = 0.0
    y0 = 0.0
    # The standard dataset generally stores the absolute location on the image to specify the principal point.
    # Thus, we need to scale and translate them such that they are offsets from the center.
    if 'cx' in transform_dict[split]:
        x0 = (float(transform_dict[split]['cx']) / (2**mip)) - (w//2)
    if 'cy' in transform_dict[split]:
        y0 = (float(transform_dict[split]['cy']) / (2**mip)) - (h//2)

    offset = transform_dict[split]['offset'] if 'offset' in transform_dict[split] else [0 ,0 ,0]
    scale = transform_dict[split]['scale'] if 'scale' in transform_dict[split] else 1.0

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

        if bg_color == 'black':
            rgbs[... ,:3] -= ( 1 -alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)
        else:
            rgbs[... ,:3] *= alpha
            rgbs[... ,:3] += ( 1 -alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)

    return {"imgs": rgbs, "masks": masks, "rays": rays, "cameras": cameras}


#############################################################################################
####################################### RTMV ################################################
#############################################################################################

def rescale_rtmv_intrinsics(camera, target_size, original_width, original_height):
    """ Rescale the intrinsics. """
    assert original_height == original_width, 'Current code assumes a square image.'
    resize_ratio = target_size * 1. / original_height
    camera.x0 *= resize_ratio
    camera.y0 *= resize_ratio
    camera.focal_x *= resize_ratio
    camera.focal_y *= resize_ratio

def load_rtmv_camera(path):
    """Loads a RTMV camera object from json metadata.
    """
    with open(path.replace("exr", "json"), 'r') as f:
        meta = json.load(f)

    # About RTMV conventions:
    # RTMV: meta['camera_data']['cam2world']), row major
    # kaolin cameras: cam2world is camera.inv_view_matrix(), column major
    # Therefore the following are equal up to a 1e-7 difference:
    # camera.inv_view_matrix() == torch.tensor(meta['camera_data']['cam2world']).T
    cam_data = meta['camera_data']
    camera = Camera.from_args(eye=torch.Tensor(cam_data['camera_look_at']['eye']),
                              at=torch.Tensor(cam_data['camera_look_at']['at']),
                              up=torch.Tensor(cam_data['camera_look_at']['up']),
                              width=cam_data['width'],
                              height=cam_data['height'],
                              focal_x=cam_data['intrinsics']['fx'],
                              focal_y=cam_data['intrinsics']['fy'],
                              x0=0.0,
                              y0=0.0,
                              near=0.0,
                              far=6.0, # inheriting default for nerf-synthetic
                              dtype=torch.float64,
                              device='cpu')

    # RTMV cameras use Blender coordinates, which are right handed with Z axis pointing upwards instead of Y.
    camera.change_coordinate_system(blender_coords())

    return camera.cpu()


def transform_rtmv_camera(camera, mip):
    """Transforms the RTMV camera according to the mip.
    """
    original_width, original_height = camera.width, camera.height
    if mip is not None:
        camera.width = camera.width // (2 ** mip)
        camera.height = camera.height // (2 ** mip)

    # assume no resizing
    rescale_rtmv_intrinsics(camera, camera.width, original_width, original_height)
    return camera


def _parallel_load_rtmv_data(args):
    """ A wrapper function to allow rtmv load faster with multiprocessing.
        All internal logic must therefore occur on the cpu.
    """
    torch.set_num_threads(1)
    with torch.no_grad():
        image, alpha, depth = load_exr(**args['exr_args'])
        camera = load_rtmv_camera(args['camera_args']['path'])
        transformed_camera = transform_rtmv_camera(copy.deepcopy(camera), mip=args['camera_args']['mip'])
        return dict(
            task_basename=args['task_basename'],
            image=image,
            alpha=alpha,
            depth=depth,
            camera=transformed_camera.cpu()
        )


def load_rtmv_data(root, split, mip=None, normalize=True, return_pointcloud=False, bg_color='white',
                   num_workers=0):
    """Load the RTMV data and applies dataset specific transforms required for compatibility with the framework.

    Args:
        root (str): The root directory of the dataset.
        split (str): The dataset split to use from 'train', 'val', 'test'.
        mip (int): If provided, will rescale images by 2**mip.
        normalize (bool): If True, will normalize the ray origins by the point cloud origin and scale.
        return_pointcloud (bool): If True, will also return the pointcloud and the scale and origin.
        bg_color (str): The background color to use for when alpha=0.
        num_workers (int): The number of workers to use for multithreaded loading. If -1, will not multithread.

    Returns:
        (dict of torch.FloatTensors): Different channels of information from RTMV.
    """
    json_files = sorted(glob.glob(os.path.join(root, '*.json')))

    # Hard-coded train-val-test splits for now (TODO(ttakikawa): pass as args?)
    train_split_idx = len(json_files) * 2 // 3
    eval_split_idx = train_split_idx + (len(json_files) * 10 // 300)

    if split == 'train':
        subset_idxs = np.arange(0, train_split_idx)
    elif split in 'val':
        subset_idxs = np.arange(train_split_idx, eval_split_idx)
    elif split == 'test':
        subset_idxs = np.arange(eval_split_idx, len(json_files))
    else:
        raise RuntimeError("Unimplemented split, check the split")

    images = []
    alphas = []
    depths = []
    rays = []
    cameras = dict()
    basenames = []

    json_files = [json_files[i] for i in subset_idxs]
    assert (len(json_files) > 0 and "No JSON files found")
    if num_workers > 0:
        # threading loading images

        p = Pool(num_workers)
        try:
            basenames = (os.path.splitext(os.path.basename(json_file))[0] for json_file in json_files)
            iterator = p.imap(_parallel_load_rtmv_data, [
                dict(
                    task_basename=basename,
                    exr_args=dict(
                        path=os.path.join(root, basename + '.exr'),
                        use_depth=True,
                        mip=mip,
                        srgb=True,
                        bg_color=bg_color),
                    camera_args=dict(
                        path=os.path.join(root, basename + '.json'),
                        mip=mip)) for basename in basenames])
            for _ in tqdm(range(len(json_files)), desc='loading data'):
                result = next(iterator)
                images.append(result["image"])
                alphas.append(result["alpha"])
                depths.append(result["depth"])
                cameras[result['task_basename']] = result["camera"]
        finally:
            p.close()
            p.join()
    else:
        for img_index, json_file in tqdm(enumerate(json_files), desc='loading data'):
            with torch.no_grad():
                basename = os.path.splitext(os.path.basename(json_file))[0]
                exr_path = os.path.join(root, basename + ".exr")
                image, alpha, depth = load_exr(exr_path, use_depth=True, mip=mip, srgb=True, bg_color=bg_color)
                json_path = os.path.join(root, basename + ".json")
                camera = load_rtmv_camera(path=json_path)
                transformed_camera = transform_rtmv_camera(copy.deepcopy(camera), mip=mip)

                images.append(image)
                alphas.append(alpha)
                depths.append(depth)
                cameras[basename] = transformed_camera

    for idx in cameras:
        camera = cameras[idx]
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height,
                                                  device='cuda')
        _rays = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(
            camera.height, camera.width, 3).to('cpu')
        rays.append(_rays.to(dtype=torch.float32))

    # Normalize
    
    if normalize:
        coords, _ = create_pointcloud_from_images(images, alphas, rays, depths)
        coords, coords_center, coords_scale = normalize_pointcloud(
            coords, return_scale=True)

        depths = torch.stack(depths)
        rays = Rays.stack(rays)
        depths = depths * coords_scale
        rays.origins = (rays.origins - coords_center) * coords_scale
        #depths = list(depths)
        #rays = list(rays)

        for cam_id, cam in cameras.items():
            cam.translate(-coords_center.to(cam.dtype))
            cam.t = cam.t * coords_scale.to(cam.dtype)

        #coords, _ = create_pointcloud_from_images(images, alphas, rays, depths)

    for idx in cameras:
        camera = cameras[idx]

    images = torch.stack(images)[..., :3]
    alphas = torch.stack(alphas)
    depths = torch.stack(depths)
    rays = Rays.stack(rays)

    output = {
        "imgs": images,
        "masks": alphas,
        "rays": rays,
        "depths": depths,
        "cameras": cameras
    }
    if return_pointcloud:
        output.update({
            "coords": coords,
            "coords_center": coords_center,
            "coords_scale": coords_scale
        })

    return output