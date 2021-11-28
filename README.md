[comment]: <> (<div align="center">)

[comment]: <> (  <img width="500px" src="https://github.com/taichi-dev/taichi/raw/master/misc/logo.png">)

[comment]: <> (   <h3> <a href="https://docs.taichi.graphics/"> Tutorial </a> | <a href="https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples"> Examples </a> | <a href="https://forum.taichi.graphics/"> Forum </a><!-- | <a href="http://hub.taichi.graphics/"> Playground </a> --></h3>)

[comment]: <> (  <h3> <a href="https://docs.taichi.graphics/"> Documentation </a> | <a href="https://docs.taichi.graphics/zh-Hans/"> 简体中文文档 </a> | <a href="https://docs.taichi.graphics/lang/articles/contribution/contributor_guide"> Contributor Guidelines </a> </h3>)

[comment]: <> (</div>)
## Overview

This project is for **CS596**. 

### What and Why

I'm interested in how to turn a pure physics PDE into something that I can visualize and understand.
Since I do not have prior knowledge about numerical method and modern physics,
a good start point is to do a simple fluid simulation. 
I choose to implement an (outdated) [APIC](https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf) method as the beginning of my learning journey. 

### Recent Study and Tools

* APIC: uses combined Eulerian and Lagrangian description to do the fluid simulation. It has several [steps](https://www.bilibili.com/video/BV1ZK411H7Hc?p=7):
  * Particles to Grids: distribute  particle physical information to the nearby grids
  * Grid operation: do the boundary conditions and pressure projection
  * Grids to Particles: distribute grid physical information to nearby particles
  * Particle Operations: move particles. update material.
* Framework: **Taichi** [太极](https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples/simulation) is a young parallel programming language for high-performance numerical computations. It is embedded in **Python**.
  * It has many examples. Newbie friendly.
  * It is written in python. Short to write. Easy to debug in the local environment.
  * It compiled python code into cuda utilising GPU.

### What I have done

* Learn fluid dynamics model. Turn the [basic 2D fluid model](https://forum.taichi.graphics/t/0-mls-mpm/1619) into 3D to generate ply files
* Learn Houdini. Bake the point cloud into the polygon mesh and generate animation
  * It took my mac 1hr 30 minutes to render these few frames. 
* Learn MPI4Py. Rewrite the basic model to support MPI. Each computation node manages a range of grids. Deploy to USC CARC system to distribute computation work.
* Do the minimum performance sampling. The bottleneck is the message synchronization and serialization.
  * One node is 10 time faster than other node. But, because of message synchronization, the slowest node decides the overall efficiency.
  * Loop in python to converting taichi tensor object to python object to interact with MPI costs a lot of time.

### What is the next

* Learn taichi source code to find intermediate proxy to quickly do the conversion.
* Dynamic grid. Let fast nodes do more computation.


<div style="display:inline-block; width:30%">
<figure style="width: 100%">
<img src="https://i.imgur.com/cewFObO.gif" />
<figcaption>Basic Model</figcaption>
</figure>
</div> 
<div style="display:inline-block; width: 30%">
<figure  style="width: 100%">
<img src="https://i.imgur.com/2dHhn7R.gif"/>
<figcaption>3D visualization</figcaption>
</figure>
</div>
<div style="display:inline-block; width:30%">
<figure  style="width: 100%">
<img src="https://i.imgur.com/6rOKDTT.gif"/>
<figcaption>4-nodes MPI-GPU computatoin</figcaption>
</figure>
</div>

| computation model (4096 particles, 128 grids, 5000 steps) | time (Wall clock in secs) |
| --- | ----------- |
| local macbook (gpu)  | 3.25 |
| 2 nodes (64 grids each, horizontal splitting) | more than 2hr |
| 4 nodes (32 grids) | 4777 |
| 8 nodes (16 grids) | 3457 |

| sub method cost (4 nodes, sample first few frames) | time (Wall clock in secs) |
| --- | ----------- |
| particle to grid (gpu computation)  | 1e-4 |
| sync grid information (MPI message) | 0.04 |
| grid operation (gpu computation) | 3e-3 |
| grid to particle (gpu computation) | 4e-3 |
| transfer particle (taichi tensor -> python obj ->taichi tensor, MPI message)| 0.47 | 
