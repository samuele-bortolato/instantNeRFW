# instantNeRFW

This project fuses [NeRF in the Wild](https://nerf-w.github.io/) with [instant NeRF](https://nvlabs.github.io/instant-ngp/) to speed up the generation of NeRFs with occlusions. 
The goal is to be able to reconstruct hand held objects without the hands or the background.
The project was developed using the library [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp) from nvidia.


## Installation

For the installation folloe the instructions [here](resources\installation.md)

## Differences from NeRFW

This work takes builds on the work of NeRF in the Wild to create NeRFs able to handle occlusions, but we made multiple changes in order to speed up the training and rendering. 

The most substantial are:
- removal of the spatial integration of the transient
- removal of the uncertainty factor
- static scene modelled with hash encoding

Every transient field is integrated always from the point of view of the corresponding camera, so there is really no point in computing the integration every time. 
Instead we propose to learn a two dimensional transient and density to be added in front of the camera.

If there is a transient object in front of a ray we stil want the color of the transient to be the same color of the image pixel, so we chose to further save the computation by not computing the transient colors but onòy the densities of the transient. This can be just be seen as a discount factor for the RGB loss worh the points that are occluded.

When trying to reconstruct hand held objects rotated in front of the camera the background tends to stay the same and match too much in all the images. This causes the creation of a fog around the object in most cases. More rarely it matches more than the object we are tryng to reconstruct, that is then modelled as a hole in the fog or just plainly put in the trainsients.
In orderd to aleviate this problem we introduced two regularizers:
- entropy regularizer
- empty background regularizer

For further details see the technical paper

## Computing the positions of the cameras


