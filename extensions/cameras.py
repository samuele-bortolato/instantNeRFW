import torch
from wisp.core import Rays
from wisp.ops.raygen import generate_centered_pixel_coords

class Cameras(torch.nn.Module):

    def __init__(self, fx, fy, cx, cy, width, height, poses, trainable=False, near=0, far=6):
        super().__init__()

        self.transform =  torch.tensor([[1, 0, 0],
                                        [0, 0, 1],
                                        [0, -1, 0]]).float()

        self.fx = torch.nn.Parameter(torch.tensor(fx),trainable)
        self.fy = torch.nn.Parameter(torch.tensor(fy),trainable)
        self.cx = torch.nn.Parameter(torch.tensor(cx),trainable)
        self.cy = torch.nn.Parameter(torch.tensor(cy),trainable)
        poses[:,:3,:3] = self.transform @ poses[:,:3,:3] 
        poses[:,:3,-1] = poses[:,:3,-1] @ self.transform.T
        self.poses = torch.nn.Parameter(torch.tensor(poses),trainable)
        self.width = width
        self.height = height
        self.near = near
        self.far = far

    def __len__(self,):
        return len(self.poses)

    def get_rays(self, cam_ids, pos_x, pos_y):
        
        #cam_ids have to by torch tensor of type long
        ray_orig = self.poses[ cam_ids, :3, -1]

        ray_dir = torch.stack((  (pos_x - self.cx) / self.fx,
                                -(pos_y - self.cy) / self.fy, #(-(pos_y + self.cy)+self.height) / self.fy, ??
                                -torch.ones_like(pos_x)), dim=-1)
        
        rotation_matrix = self.poses[ cam_ids, :3, :3]
        
        # third_dir = rotation_matrix[:,:,2]
        # first_dir = rotation_matrix[:,:,0]
        # first_dir = first_dir-third_dir*(third_dir[:,None]@first_dir[...,None])[:,0]
        # first_dir = first_dir/first_dir.norm(dim=1)[:,None]
        # second_dir = torch.cross(third_dir,first_dir, dim=1)

        # rotation_matrix = torch.stack([first_dir,second_dir,third_dir],2)
        
        ray_dir = (rotation_matrix @ ray_dir[...,None])[:,:,0]
        ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

        return Rays(origins=ray_orig, dirs=ray_dir, dist_min=self.near, dist_max=self.far)

#        return {"origins":ray_orig, "dirs":ray_dir, "dist_min":self.near, "dist_max":self.far}

    def get_n_rays_of_cam(self, cam_id, n=2**10):
        pos_x = torch.randint(0,self.width,(n,),dtype=torch.float32, device='cuda') + 0.5
        pos_y = torch.randint(0,self.width,(n,),dtype=torch.float32, device='cuda') + 0.5
        return self.get_rays( torch.empty(len(pos_x), dtype=torch.int64, device='cuda').fill_(cam_id), pos_x , pos_y)