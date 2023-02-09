# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import inspect
import torch
from typing import Dict, Any, List
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid

class Nef(BaseNeuralField):
    """Model for encoding Neural Radiance Fields (Mildenhall et al. 2020), e.g., density and view dependent color.
    Different to the original NeRF paper, this implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(self,
                 grid: BLASGrid = None,
                 grid_t: List[BLASGrid] = None,
                 appearence_embedding: torch.Tensor = None,
                 # embedder args
                 pos_embedder: str = 'none',
                 view_embedder: str = 'none',
                 pos_t_embedder: str = 'none',
                 pos_multires: int = 10,
                 view_multires: int = 4,
                 position_input: bool = False,
                 # decoder args
                 activation_type: str = 'relu',
                 layer_type: str = 'none',
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 beta_min: float = 0.01,
                 # pruning args
                 prune_density_decay: float = None,
                 prune_min_density: float = None,
                 ):
        """
        Creates a new NeRF instance, which maps 3D input coordinates + view directions to RGB + density.

        This neural field consists of:
         * A feature grid (backed by an acceleration structure to boost raymarching speed)
         * Color & density decoders
         * Optional: positional embedders for input position coords & view directions, concatenated to grid features.

         This neural field also supports:
          * Aggregation of multi-resolution features (more than one LOD) via summation or concatenation
          * Pruning scheme for HashGrids

        Args:
            grid: (BLASGrid): represents feature grids in Wisp. BLAS: "Bottom Level Acceleration Structure",
                to signify this structure is the backbone that captures
                a neural field's contents, in terms of both features and occupancy for speeding up queries.
                Notable examples: OctreeGrid, HashGrid, TriplanarGrid, CodebookGrid.

            pos_embedder (str): Type of positional embedder to use for input coordinates.
                Options:
                 - 'none': No positional input is fed into the density decoder.
                 - 'identity': The sample coordinates are fed as is into the density decoder.
                 - 'positional': The sample coordinates are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the density decoder.
            view_embedder (str): Type of positional embedder to use for view directions.
                Options:
                 - 'none': No positional input is fed into the color decoder.
                 - 'identity': The view directions are fed as is into the color decoder.
                 - 'positional': The view directions are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the color decoder.
            pos_multires (int): Number of frequencies used for 'positional' embedding of pos_embedder.
                 Used only if pos_embedder is 'positional'.
            view_multires (int): Number of frequencies used for 'positional' embedding of view_embedder.
                 Used only if view_embedder is 'positional'.
            position_input (bool): If True, the input coordinates will be passed into the decoder.
                 For 'positional': the input coordinates will be concatenated to the embedded coords.
                 For 'none' and 'identity': the embedder will behave like 'identity'.
            activation_type (str): Type of activation function to use in BasicDecoder:
                 'none', 'relu', 'sin', 'fullsort', 'minmax'.
            layer_type (str): Type of MLP layer to use in BasicDecoder:
                 'none' / 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'.
            hidden_dim (int): Number of neurons in hidden layers of both decoders.
            num_layers (int): Number of hidden layers in both decoders.
            prune_density_decay (float): Decay rate of density per "prune step",
                 using the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
            prune_min_density (float): Minimal density allowed for "cells" before they get pruned during a "prune step".
                 Used within the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
        """
        super().__init__()
        self.grid = grid
        self.grid_t = grid_t
        self.appearence_embedding = appearence_embedding
        self.appearence_feat = appearence_embedding.shape[1]

        # Init Embedders
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires,
                                                                   include_input=position_input)
        self.view_embedder, self.view_embed_dim = self.init_embedder(view_embedder, view_multires,
                                                                     include_input=True)
        

        # Init Decoder
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.decoder_density = BasicDecoder(input_dim=self.density_net_input_dim(),
                                       output_dim=16,
                                       activation=get_activation_class(activation_type),
                                       bias=True,
                                       layer=get_layer_class(layer_type),
                                       num_layers=num_layers,
                                       hidden_dim=hidden_dim,
                                       skip=[])
        self.decoder_density.lout.bias.data[0] = 0.1 
        
        self.decoder_color = BasicDecoder(input_dim=self.color_net_input_dim(),
                                     output_dim=3,
                                     activation=get_activation_class(activation_type),
                                     bias=True,
                                     layer=get_layer_class(layer_type),
                                     num_layers=num_layers + 1,
                                     hidden_dim=hidden_dim,
                                     skip=[])
        if grid_t is not None:
            self.decoder_transient = BasicDecoder(input_dim=self.transient_net_input_dim(),
                                        output_dim=5,
                                        activation=get_activation_class(activation_type),
                                        bias=True,
                                        layer=get_layer_class(layer_type),
                                        num_layers=num_layers + 1,
                                        hidden_dim=hidden_dim,
                                        skip=[])
            self.decoder_transient.lout.bias.data[0] = 0.1

        self.beta_min = beta_min

        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density

        self.backgroud_color=torch.nn.parameter.Parameter(torch.ones(3)*0.01)

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, include_input=False):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none' and not include_input:
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity' or (embedder_type == 'none' and include_input):
            embedder, embed_dim = torch.nn.Identity(), 3    # Assumes pos / view input is always 3D
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, include_input=include_input)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralRadianceField: {embedder_type}')
        return embedder, embed_dim


    def prune(self):
        """Prunes the blas based on current state.
        """
        print('prune')
        if not isinstance(self.grid, HashGrid):
            raise NotImplementedError(f'Pruning not implemented for grid type {self.grid}')
        for grid in self.grid_t:
            if not isinstance(self.grid, HashGrid):
                raise NotImplementedError(f'Pruning not implemented for grid type {self.grid}')

        if self.grid_t is not None:
            density_decay = self.prune_density_decay
            min_density = self.prune_min_density

            self.grid.occupancy = self.grid.occupancy.cuda()
            self.grid.occupancy = self.grid.occupancy * density_decay
            points = self.grid.dense_points.cuda()
            res = 2.0**self.grid.blas_level

            for i, grid in enumerate(self.grid_t):

                grid.occupancy = grid.occupancy.cuda()
                grid.occupancy = grid.occupancy * density_decay
                points_t = grid.dense_points.cuda()

                samples = torch.rand(points_t.shape[0], 3, device=points_t.device)
                samples = points_t.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0

                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points_t.device)
                with torch.no_grad():
                    density, density_t = self.forward(coords=samples, idx=i, ray_d=sample_views, channels=["density","density_t"])
                    density_t = density + density_t
                
                self.grid.occupancy = torch.stack([density[:, 0], grid.occupancy], -1).max(dim=-1)[0]
                grid.occupancy = torch.stack([density_t[:, 0], grid.occupancy], -1).max(dim=-1)[0]

                del density, density_t

                mask_t = grid.occupancy > min_density
                _points_t = points_t[mask_t]
                grid.blas = grid.blas.__class__.from_quantized_points(_points_t, grid.blas_level)
            
            mask = self.grid.occupancy > min_density
            _points = points[mask]
            self.grid.blas = self.grid.blas.__class__.from_quantized_points(_points, grid.blas_level)

    def forward(self, channels=None, **kwargs):
        """Queries the neural field with channels.

        Args:
            channels (str or list of str or set of str): Requested channels. See return value for details.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (list or dict or torch.Tensor): 
                If channels is a string, will return a tensor of the request channel. 
                If channels is a list, will return a list of channels.
                If channels is a set, will return a dictionary of channels.
                If channels is None, will return a dictionary of all channels.
        """
        kwargs['channels'] = channels
        if not (isinstance(channels, str) or isinstance(channels, list) or isinstance(channels, set) or channels is None):
            raise Exception(f"Channels type invalid, got {type(channels)}." \
                            "Make sure your arguments for the nef are provided as keyword arguments.")
        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)

        unsupported_channels = requested_channels - self.get_supported_channels()
        if unsupported_channels:
            raise Exception(f"Channels {unsupported_channels} are not supported in {self.__class__.__name__}")
        
        return_dict = {}
        for fn in self._forward_functions:

            output_channels = self._forward_functions[fn]
            # Filter the set of channels supported by the current forward function
            supported_channels = output_channels & requested_channels

            # Check that the function needs to be executed
            if len(supported_channels) != 0:

                # Filter args to the forward function and execute
                argspec = inspect.getfullargspec(fn)
                required_args = argspec.args[:-len(argspec.defaults)][1:] # Skip first element, self
                optional_args = argspec.args[-len(argspec.defaults):]
                
                input_args = {}
                for _arg in required_args:
                    # TODO(ttakiakwa): This doesn't actually format the string, fix :) 
                    if _arg not in kwargs:
                        raise Exception(f"Argument {_arg} not found as input to in {self.__class__.__name__}.{fn.__name__}()")
                    input_args[_arg] = kwargs[_arg]
                for _arg in optional_args:
                    if _arg in kwargs:
                        input_args[_arg] = kwargs[_arg]
                output = fn(**input_args)

                for channel in supported_channels:
                    return_dict[channel] = output[channel]
        
        if isinstance(channels, str):
            if channels in return_dict:
                return return_dict[channels]
            else:
                return None
        elif isinstance(channels, list):
            return [return_dict[channel] for channel in channels]
        else:
            return return_dict

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.sample, ["density", "rgb", "rgb_t", "density_t", "beta_t"])


    def sample(self, coords, ray_d=None, idx=None, lod_idx=None, channels=None):
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
        
        require_color = "rgb" in channels
        require_transient =  ("rgb_t" in channels ) or ( "density_t" in channels) or ("beta_t" in channels)

        original_shape=coords.shape
       
        threshold = 0.5
        if lod_idx is not None:
            mask = torch.sum(torch.square(coords), 1) < threshold
        else:
            mask=torch.ones(original_shape[0],dtype=torch.bool)
        coords = coords[mask]
        ray_d = ray_d[mask]

        if len(coords)>0:

            if lod_idx is None:
                lod_idx = len(self.grid.active_lods) - 1
            batch, _ = coords.shape

            # Embed coordinates into high-dimensional vectors with the grid.
            feats = self.grid.interpolate(coords, lod_idx).reshape(batch, self.effective_feature_dim())
            if require_transient:
                if idx is None:
                    idx=0
                feats_t = self.grid_t[0].interpolate(   coords + torch.tensor([10*idx,0,0], dtype=coords.dtype,device=coords.device), 
                                                        lod_idx).reshape(batch, self.effective_feature_dim())

            # Optionally concat the positions to the embedding
            if self.pos_embedder is not None:
                embedded_pos = self.pos_embedder(coords).view(batch, self.pos_embed_dim)
                feats = torch.cat([feats, embedded_pos], dim=-1)
                if require_transient:
                    feats_t = torch.cat([feats_t, embedded_pos], dim=-1)

            # Decode high-dimensional vectors to density features.
            density_feats = self.decoder_density(feats)

            # Density is [particles / meter], so need to be multiplied by distance
            # density ~ (batch, 1)
            density = torch.zeros(original_shape[0], 1, device = density_feats.device, dtype=density_feats.dtype)
            density[mask] = torch.relu(density_feats[...,0:1])

            if require_color:

                if ray_d is None:
                    raise Exception(f"Ray direction is required to compute static color")

                if idx is not None:
                    appearence = torch.broadcast_to(self.appearence_embedding[idx:idx+1], (batch, self.appearence_feat))
                else:
                    appearence = torch.broadcast_to(self.appearence_embedding[:1], (batch, self.appearence_feat))

                # Concatenate embedded view directions.
                if self.view_embedder is not None:
                    embedded_dir = self.view_embedder(-ray_d).view(batch, self.view_embed_dim)
                    fdir = torch.cat([density_feats, appearence, embedded_dir], dim=-1)
                else:
                    fdir = torch.cat([density_feats, appearence], dim=-1)

                # Colors are values [0, 1] floats
                # colors ~ (batch, 3)
                c = torch.sigmoid(self.decoder_color(fdir))
                rgb = torch.zeros(original_shape[0], 3, device = c.device, dtype=c.dtype)
                rgb[mask] = c

            if require_transient:
                transient = self.decoder_transient(torch.concat([feats_t, density_feats],-1))

                c_t = torch.sigmoid(transient[...,2:5])
                d_t = torch.nn.functional.softplus(transient[...,0:1])
                b_t = self.beta_min + torch.nn.functional.softplus(transient[...,1:2])

                rgb_t = torch.zeros(original_shape[0], 3, device = c_t.device, dtype=c_t.dtype)
                density_t = torch.zeros(original_shape[0], 1, device = d_t.device, dtype=d_t.dtype)
                beta_t = torch.zeros(original_shape[0], 1, device = b_t.device, dtype=b_t.dtype)

                rgb_t[mask] = c_t
                density_t[mask] = d_t
                beta_t[mask] = b_t
        else:
            density = torch.zeros(original_shape[0], 1, device = coords.device, dtype=coords.dtype)
            if require_color:
                rgb = torch.zeros(original_shape[0], 3, device = coords.device, dtype=coords.dtype)
            if require_transient:
                rgb_t = torch.zeros(original_shape[0], 3, device = coords.device, dtype=coords.dtype)
                density_t = torch.zeros(original_shape[0], 1, device = coords.device, dtype=coords.dtype)
                beta_t = torch.zeros(original_shape[0], 1, device = coords.device, dtype=coords.dtype)



        out_dict = {}
        out_dict['density'] = density
        if require_color:
            out_dict['rgb'] = rgb
        if require_transient:
            out_dict['rgb_t'] = rgb_t
            out_dict['density_t'] = density_t
            out_dict['beta_t'] = beta_t

        return out_dict

    def effective_feature_dim(self, grid = None):
        if grid is None:
            grid=self.grid
    
        if grid.multiscale_type == 'cat':
            effective_feature_dim = grid.feature_dim * grid.num_lods
        else:
            effective_feature_dim = grid.feature_dim
        return effective_feature_dim

    def density_net_input_dim(self):
        return self.effective_feature_dim() + self.pos_embed_dim

    def color_net_input_dim(self):
        return 16 + self.appearence_feat + self.view_embed_dim

    def transient_net_input_dim(self):
        return 16 + self.effective_feature_dim(self.grid_t[0]) + self.pos_embed_dim

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {
            "Grid": self.grid,
            "Pos. Embedding": self.pos_embedder,
            "View Embedding": self.view_embedder,
            "Decoder (density)": self.decoder_density,
            "Decoder (color)": self.decoder_color
        }
        if self.prune_density_decay is not None:
            properties['Pruning Density Decay'] = self.prune_density_decay
        if self.prune_min_density is not None:
            properties['Pruning Min Density'] = self.prune_min_density
        return properties
