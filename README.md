# Topological Generation using Stable Diffusion - Environment Generator, Encoder, and Decoder

This repo contains the code for the accompanying [research paper](https://github.com/AdityaAngajala/FRI_2_Project/files/14118071/Goal_Oriented_3D_Environment_Generation_Final.pdf). I would highly recommend reading it to better understand the context with which this code is written. 

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

Animations:
![Slices](https://github.com/AdityaAngajala/FRI_2_Project/assets/53411299/3bec09f8-8dd2-4a07-8a97-1b3aad63f551)

[Hilbert](https://github.com/AdityaAngajala/FRI_2_Project/assets/53411299/8ca5a7bf-33f8-46bc-8dec-10acefde963d)


