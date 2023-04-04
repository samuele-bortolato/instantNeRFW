import torch

class Cameras(torch.nn.Module):

    def __init__(self, fx, fy, cx, cy, width, height, poses, trainable=False, near=0, far=6):
        super().__init__()
        self.fx = torch.nn.parameter(fx,trainable)
        self.fy = torch.nn.parameter(fy,trainable)
        self.cx = torch.nn.parameter(cx,trainable)
        self.cy = torch.nn.parameter(cy,trainable)
        self.poses = torch.nn.parameter(poses,trainable)
        self.width = width
        self.height = height
        self.near = near
        self.far = far


    def get_rays(self, cam_ids, pos_x, pos_y):
        
        #cam_ids have to by torch tensor of type long
        ray_orig = self.poses[ cam_ids, :3, -1]

        #normalize coordinates
        pixel_x = 2 * ( (pos_x - self.cx / self.width)) 
        pixel_y = 2 * ( (pos_y - self.cy / self.height))

        ray_dir = torch.stack((  pixel_x * (self.width / 2.0) / self.fx,
                                -pixel_y * (self.height / 2.0) / self.fy,
                                -torch.ones_like(pixel_x)), dim=-1)
        
        rotation_matrix = self.poses[ cam_ids, :3, :3]

        ray_dir = rotation_matrix @ ray_dir[...,None]
        ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

        return {"origins":ray_orig, "dirs":ray_dir, "dist_min":self.near, "dist_max":self.far}