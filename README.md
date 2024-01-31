# Topological Generation using Stable Diffusion - Environment Generator, Encoder, and Decoder

This repo contains the code for the accompanying [research paper ]([url](https://drive.google.com/file/d/1hfSenvkE6SxhtQsS47allLh3wAr-YC-F/view?usp=sharing)). I would highly recommend reading it to better understand the context with which this code is written. 

Repo incudes: 
- Randomized Topological Environment Generator 
  - Procedural voxel generation of natural terrain with customizable generation of special features
  - Interactive 3D visualization using Pyvista
  - Automatic labeling and generated environments
- Voxel-to-Image Encoder and Image-to-Voxel Decoder
  - 3D Environment can be encoded into a 2D Image in 3 custom designed algorithms.
  - Decoder for each encoding scheme to decode the images back to 3D data
  - Designed to be resistant to color corruption.

voxelGeneration.py contains the bulk of the 3D environent code generaion and animation. 
voxelDecoder.py contains the code to decode the outputs of voxelGeneration

topologyLabelGeneration.py and encoding_decoder.py contain the 2D variant of the same problem.

The utils contains:
- script to generate hilbert indicies
- GUI to visualize color correction and rounding
- image augmentation script to test color correctio.  
