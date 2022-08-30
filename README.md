# Path tracer on CUDA

![MainImg](./Img/Render/DiffuseRoom_MS8x2048spp8bounce_22min.png)

Simple path tracer renderer using CUDA.



# Features

- Path tracer on CUDA
  - BRDF
    - Diffuse reflection (Lambertian)
    - Specular reflection
  - Non recursive implementation
  - Monte Carlo integration
    - NEE
- BVH Acceleration
  - BVH construction using SAH
  - [DEPRECATED] BVH construction using clustering algorithm
    - ![BVH](./Img/Doc/BVH.png)
  
  - Non recursive BVH traverse
  
- Simple scene viewer on OpenGL 4.6
  - For now,  rendering cluster of polygons is all.


### To be implemented later...

- Refraction,...
- optimizations.,
  - metropolis light transformation?

- Scene loading and exportation using json file.

# License

This project contains code from following open source software

1. **[yocto-gl](https://github.com/xelatihy/yocto-gl)**
   1. License : MIT license
   2. Download Site : [xelatihy/yocto-gl: Yocto/GL: Tiny C++ Libraries for Data-Driven Physically-based Graphics (github.com)](https://github.com/xelatihy/yocto-gl)

# Rendered images...

1. Image for comparision of image with NEE and one without. (16spp)

![](./Img/Render/16spp_woNEE.png) ![](./Img/Render/16spp_NEE.png)

2. Diffuse room

![](./Img/Render/DiffuseRoom_MS8x2048spp_13min.png)

3. Mirror bunny

![][./Img/Render/bunny.png]![][./Img/Render//Render/bunny2.png]
