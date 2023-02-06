# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid
from typing import List

import math

import tinycudann as tcnn

import kaolin.ops.spc as spc_ops
from wisp.accelstructs import OctreeAS
from wisp.models.grids import BLASGrid


class FusedHashGrid(BLASGrid):

    def __init__(   self,
                    # Hash encodings
                    n_levels : int = 16,
                    n_features_per_level: int = 2,
                    log2_hashmap_size: int = 15,
                    base_resolution: int = 16,
                    per_level_scale: float = 1.5,
                    multiscale_type: str = 'cat',
                    # accel struct
                    blas_level: int = 7):
        super().__init__()
        self.blas_level = blas_level
        self.hash_encoding_config=  {
                                        "otype": "HashGrid",
                                        "n_levels": n_levels,
                                        "n_features_per_level": n_features_per_level,
                                        "log2_hashmap_size": log2_hashmap_size,
                                        "base_resolution": base_resolution,
                                        "per_level_scale": per_level_scale
                                    }

        octree=torch.zeros(torch.sum(torch.pow(torch.pow(2,torch.arange(7)),3)), dtype=torch.uint8)-1
        self.blas=OctreeAS(octree.cuda())
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points,
                                                               self.blas.pyramid,
                                                               self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.zeros(self.num_cells)

        self.feature_dim = n_features_per_level
        self.multiscale_type = multiscale_type
        self.codebook_bitwidth = log2_hashmap_size

        self.num_lods = n_levels
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

        self.codebook = tcnn.Encoding(n_input_dims = 3,  encoding_config = self.hash_encoding_config)


    def freeze(self):
        self.codebook.requires_grad_(False)

    def interpolate(self, coords, lod_idx):
        output_shape = coords.shape[:-1]
        if coords.ndim == 3:    # flatten num_samples dim with batch for cuda call
            batch, num_samples, coords_dim = coords.shape  # batch x num_samples
            coords = coords.reshape(batch * num_samples, coords_dim)
            
        feats = self.codebook(coords).float()

        if self.multiscale_type == 'cat':
            return feats.reshape(*output_shape, feats.shape[-1])
        elif self.multiscale_type == 'sum':
            return feats.reshape(*output_shape, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2)
        else:
            raise NotImplementedError

    def raymarch(self, rays, raymarch_type, num_samples, level=None):
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=self.blas_level)

    def name(self) -> str:
        return "Fused Hash Grid"


class FusedNeuralRadianceField(BaseNeuralField):
    """Model for encoding Neural Radiance Fields (Mildenhall et al. 2020), e.g., density and view dependent color.
    Different to the original NeRF paper, this implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(   self,
                    grid,
                    #Frequency embedder (direction)
                    view_multires: int = 4,
                    # fused neural networks
                    n_neurons: int = 64,
                    n_hidden_layers_density: int = 1,
                    n_hidden_layers_color: int = 2,
                    # pruning args
                    prune_density_decay: float = 0.95,
                    prune_min_density: float = 0.01 * 1024 / math.sqrt(3)
                    ):
        
        super().__init__()
        
        self.density_network_config={
                                        "otype": "FullyFusedMLP",
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": n_neurons,
                                        "n_hidden_layers": n_hidden_layers_density
                                    }
        self.color_network_config=  {
                                        "otype": "FullyFusedMLP",
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": n_neurons,
                                        "n_hidden_layers": n_hidden_layers_color
                                    }
        # self.grid=accelGrid(accel_blas_level)
        # self.embedding_and_density_net = tcnn.NetworkWithInputEncoding( n_input_dims=3, 
        #                                                             n_output_dims=16, 
        #                                                             encoding_config=self.hash_encoding_config, 
        #                                                             network_config=self.density_network_config
        #                                                             ).cuda()
                                                                    
        # self.view_embedder, self.view_embed_dim = get_positional_embedder(frequencies=view_multires, include_input=True)
        # self.color_net = tcnn.Network(   n_input_dims= 16 + self.view_embed_dim, 
        #                             n_output_dims=3, 
        #                             network_config=self.color_network_config)

        self.grid = grid
        # Init Embedders
        self.decoder_density = tcnn.Network(self.grid.codebook.n_output_dims, 16, network_config = self.density_network_config)
        self.view_embedder, self.view_embed_dim = get_positional_embedder(frequencies = view_multires, include_input = True)
        self.decoder_color = tcnn.Network(16 + self.view_embed_dim, 3, network_config = self.color_network_config)

        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density

        torch.cuda.empty_cache()


    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ["density", "rgb"])

    def rgba(self, coords, ray_d, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, 3]
                - Density tensor of shape [batch, 1]
        """
        
        batch, _ = coords.shape
        print(coords.shape)

        feats = self.grid.interpolate(coords, lod_idx).reshape(batch, -1)

        density_feats = self.decoder_density(feats)
        
        # Concatenate embedded view directions.
        embedded_dir = self.view_embedder(-ray_d).view(batch, self.view_embed_dim)
        fdir = torch.cat([density_feats, embedded_dir], dim=-1)

        # Colors are values [0, 1] floats
        # colors ~ (batch, 3)
        colors = torch.sigmoid(self.decoder_color(fdir))

        # Density is [particles / meter], so need to be multiplied by distance
        # density ~ (batch, 1)
        density = torch.relu(density_feats[...,0:1])
        return dict(rgb=colors, density=density)

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.grid is not None:
            if isinstance(self.grid, HashGrid):
                density_decay = self.prune_density_decay
                min_density = self.prune_min_density

                self.grid.occupancy = self.grid.occupancy.cuda()
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points.cuda()
                res = 2.0**self.grid.blas_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
                with torch.no_grad():
                    density = self.forward(coords=samples, ray_d=sample_views, channels="density")
                self.grid.occupancy = torch.stack([density[:, 0], self.grid.occupancy], -1).max(dim=-1)[0]

                mask = self.grid.occupancy > min_density

                _points = points[mask]

                if _points.shape[0] == 0:
                    return

                if hasattr(self.grid.blas.__class__, "from_quantized_points"):
                    self.grid.blas = self.grid.blas.__class__.from_quantized_points(_points, self.grid.blas_level)
                else:
                    raise Exception(f"The BLAS {self.grid.blas.__class__.__name__} does not support initialization " 
                                     "from_quantized_points, which is required for pruning.")

            else:
                raise NotImplementedError(f'Pruning not implemented for grid type {self.grid}')

    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    def density_net_input_dim(self):
        return self.effective_feature_dim() + self.pos_embed_dim

    def color_net_input_dim(self):
        return 16 + self.view_embed_dim