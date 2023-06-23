# modified packed rf tracer:
#   - traced bothe the static and transient scene
#   - added support for renderning only the static scene
#   (- batched rendering to not crash the gpu with low memory)

import numpy as np
import torch
import torch.nn as nn
import kaolin.render.spc as spc_render
from wisp.core import RenderBuffer
from wisp.tracers import BaseTracer
from wisp.core import Rays
import inspect
import time

from extensions.raytrace import exponential_integration

from wisp.ops.raygen import generate_pinhole_rays

class Tracer(BaseTracer):
    """Tracer class for sparse (packed) radiance fields.
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is differentiable, and can be employed within training loops.

    This tracer class expects the neural field to expose a BLASGrid: a Bottom-Level-Acceleration-Structure Grid,
    i.e. a grid that inherits the BLASGrid class for both a feature structure and an occupancy acceleration structure).
    """
    def __init__(self, raymarch_type='voxel', num_steps=128, step_size=1.0, bg_color='white', rendering_threshold_density = 0.01):
        """Set the default trace() arguments.

        Args:
            raymarch_type (str): Sample generation strategy to use for raymarch.
                'voxel' - intersects the rays with the acceleration structure cells.
                    Then among the intersected cells, each cell is sampled `num_steps` times.
                'ray' - samples `num_steps` along each ray, and then filters out samples which falls outside of occupied
                    cells of the acceleration structure.
            num_steps (int): The number of steps to use for the sampling. The meaning of this parameter changes
                depending on `raymarch_type`:
                'voxel' - each acceleration structure cell which intersects a ray is sampled `num_steps` times.
                'ray' - number of samples generated per ray, before culling away samples which don't fall
                    within occupied cells.
                The exact number of samples generated, therefore, depends on this parameter but also the occupancy
                status of the acceleration structure.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (str): The background color to use.
        """
        super().__init__()
        self.raymarch_type = raymarch_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.bg_color = bg_color
        self.rendering_threshold_density = rendering_threshold_density

    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"depth", "hit", "rgb", "alpha"}

    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Returns:
            (set): Set of channel strings.
        """
        return {"rgb", "density"}

    def forward(self, nef, rays: Rays, idx=None, pos_x=None, pos_y=None, channels=None, **kwargs):
        """Queries the tracer with channels.

        Args:
            nef (BaseNeuralField): Neural field to be traced. The nef will be queried for decoded sample values.
            rays (Rays): Pack of rays to trace through the neural field.
            channels (str or list of str or set of str): Requested channel names.
            This list should include at least all channels in tracer.get_supported_channels(),
            and may include extra channels in addition.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        nef_channels = nef.get_supported_channels()
        unsupported_inputs = self.get_required_nef_channels() - nef_channels
        if unsupported_inputs:
            raise Exception(f"The neural field class {type(nef)} does not output the required channels {unsupported_inputs}.")

        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)
        extra_channels = requested_channels - self.get_supported_channels()
        unsupported_outputs = extra_channels - nef_channels
        if unsupported_outputs:
            raise Exception(f"Channels {unsupported_outputs} are not supported in the tracer {type(self)} or neural field {type(nef)}.")
    
        if extra_channels is None:
            requested_extra_channels = set()
        elif isinstance(extra_channels, str):
            requested_extra_channels = set([extra_channels])
        else:
            requested_extra_channels = set(extra_channels)

        argspec = inspect.getfullargspec(self.trace)

        # Skip self, nef, rays, channel, extra_channels
        required_args = argspec.args[:-len(argspec.defaults)][8:]   # TODO (operel): this is brittle
        optional_args = argspec.args[-len(argspec.defaults):]
        
        input_args = {}
        for _arg in required_args:
            # TODO(ttakiakwa): This doesn't actually format the string, fix :) 
            if _arg not in kwargs:
                raise Exception(f"Argument {_arg} not found as input to in {type(self)}.trace()")
            input_args[_arg] = kwargs[_arg]
        for _arg in optional_args:
            if _arg in kwargs:
                # By default, the function args will take priority
                input_args[_arg] = kwargs[_arg]
            else:
                # Check if default_args are set, and use them if they are.
                default_arg = getattr(self, _arg, None)
                if default_arg is not None:
                    input_args[_arg] = default_arg
        return self.trace(nef, rays, idx, pos_x, pos_y, requested_channels, requested_extra_channels, **input_args)



    def trace(self, nef, rays, idx, pos_x, pos_y, channels, extra_channels,
        lod_idx=None, raymarch_type='voxel', num_steps=64, step_size=1.0, bg_color='white', percentage=None):
        """Trace the rays against the neural field.

        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): If there are any extra channels requested, this tracer will by default
                                  perform volumetric integration on those channels.
            rays (wisp.core.Rays): Ray origins and directions of shape [N, 3]
            lod_idx (int): LOD index to render at. 
            raymarch_type (str): The type of raymarching algorithm to use. Currently we support:
                                 voxel: Finds num_steps # of samples per intersected voxel
                                 ray: Finds num_steps # of samples per ray, and filters them by intersected samples
            num_steps (int): The number of steps to use for the sampling.
            step_size (float): The step size between samples. Currently unused, but will be used for a new
                               sampling method in the future.
            bg_color (str): The background color to use. TODO(ttakikawa): Might be able to simplify / remove

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the render.
        """

        if rays is not None:
            N = rays.origins.shape[0]
        else:
            N = len(idx)
            #rays= [generate_pinhole_rays(nef.cameras[list(nef.cameras)[index]], (pos_x,pos_y)) for index in idx]
            #rays= Rays.stack(rays).to(dtype=torch.float).reshape(-1, 3)
            rays = nef.cameras.get_rays(idx, pos_x, pos_y)

        #TODO(ttakikawa): Use a more robust method
        assert nef.grid is not None and "this tracer requires a grid"
        
        if "depth" in channels:
            depth = torch.zeros(N, 1, device=rays.origins.device)
        else: 
            depth = None
        
        if bg_color == 'white':
            rgb = torch.ones(N, 3, device=rays.origins.device)
        else:
            rgb = torch.zeros(N, 3, device=rays.origins.device)
        var = torch.ones(N, 3, device=rays.origins.device) * 0.1
        hit = torch.zeros(N, device=rays.origins.device, dtype=torch.bool)
        out_alpha = torch.zeros(N, 1, device=rays.origins.device)

        if lod_idx is None:
            lod_idx = nef.grid.num_lods - 1

        
        # By default, PackedRFTracer will attempt to use the highest level of detail for the ray sampling.
        # This however may not actually do anything; the ray sampling behaviours are often single-LOD
        # and is governed by however the underlying feature grid class uses the BLAS to implement the sampling.
        raymarch_results = nef.grid.raymarch(rays,
                                            level=nef.grid.active_lods[lod_idx],
                                            num_samples=num_steps,
                                            raymarch_type=raymarch_type)
        ridx = raymarch_results.ridx
        samples = raymarch_results.samples
        deltas = raymarch_results.deltas
        boundary = raymarch_results.boundary
        depths = raymarch_results.depth_samples

        # Get the indices of the ray tensor which correspond to hits
        ridx_hit = ridx[spc_render.mark_pack_boundaries(ridx.int())]
        # Compute the color and density for each ray and their samples
        hit_ray_d = rays.dirs.index_select(0, ridx)

        # Compute the color and density for each ray and their samples
        num_samples = samples.shape[0]

        color, density = nef(coords=samples, ray_d=hit_ray_d, idx = idx[ridx] if torch.is_tensor(idx) else idx, lod_idx=lod_idx, channels=["rgb", "density"], percentage=percentage)

        if idx is None:
            density = density * (density >= self.rendering_threshold_density)
        
        density = density.reshape(num_samples, 1)    # Protect against squeezed return shape
        del ridx

        # Compute optical thickness
        tau = density * deltas
        del deltas
        # if idx is not None: # training
        #     alpha = 1.0 - torch.exp(-tau.contiguous())
        #     transm = torch.exp(-1.0 * spc_render.cumsum(tau.contiguous(), boundary.contiguous(), exclusive=True))
        #     extra_outputs['transmittance'] = transm
        #     transmittance = transm * alpha
        #     ray_colors = color
        # else: # gui
        ray_colors, transmittance = spc_render.exponential_integration(color, tau, boundary, exclusive=True)

        alpha = spc_render.sum_reduce(transmittance, boundary)
        out_alpha[ridx_hit] = alpha
        hit[ridx_hit] = alpha[...,0] > 0.0

        if "depth" in channels:
            ray_depth = spc_render.sum_reduce(depths.reshape(num_samples, 1) * transmittance, boundary)
            depth[ridx_hit, :] = ray_depth + 10*(1-alpha)

        # Populate the background
        # if bg_color == 'white':
        #     color = (1.0-alpha) + ray_colors
        # else:
        #     color = alpha * ray_colors
        color = (1.0-alpha)*torch.sigmoid(100*nef.backgroud_color) + alpha *ray_colors

        rgb[ridx_hit] = color

        if idx is not None: # use transients
            alpha_t = nef.sample_t(ray_d=rays.dirs, idx=idx, pos_x=pos_x, pos_y=pos_y)
            extra_outputs = {'density': density, 'alpha_t': alpha_t}
        else:
            extra_outputs = {}

        for channel in extra_channels:
            feats = nef(coords=samples,
                        ray_d=hit_ray_d,
                        lod_idx=lod_idx,
                        channels=channel)
            num_channels = feats.shape[-1]
            ray_feats, transmittance = spc_render.exponential_integration(
                feats.view(num_samples, num_channels), tau, boundary, exclusive=True
            )
            composited_feats = alpha * ray_feats
            out_feats = torch.zeros(N, num_channels, device=feats.device)
            out_feats[ridx_hit] = composited_feats
            extra_outputs[channel] = out_feats

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha, **extra_outputs)

