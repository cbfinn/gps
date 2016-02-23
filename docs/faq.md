FAQ
===

**What does this codebase include?**

This codebase implements the BADMM-based guided policy search algorithm, including LQG-based trajectory optimization, fitting
local dynamics model using a Gaussian Mixture Model prior, and the cost functions used our work.

For ease of applying the method, we also provide interfaces with three simulation and robotic platforms: Box2D for its ease of access,
Mujoco for its more accurate and capable 3D physics simulation, and ROS for interfacing with real-world robots.

**What does this codebase *not* include?**

The codebase is a work in progress.

It includes the algorithm detailed in ([Levine, Finn et al., 2015](http://arxiv.org/pdf/1504.00702.pdf)) but does not yet include support for images and convolutional networks, which is under development.

It does not include the constrained guided policy search algorithm in ([Levine et al., 2015](http://rll.berkeley.edu/icra2015gps/robotgps.pdf); [Levine and Abbeel, 2014](http://www.eecs.berkeley.edu/~svlevine/papers/mfcgps.pdf)). For a discussion on the
differences between the BADMM and constrained versions of the algorithm, see ([Levine, Finn et al. 2015](http://arxiv.org/pdf/1504.00702.pdf)).

Other extensions on the algorithm, including ([Finn et al., 2016](http://arxiv.org/pdf/1509.06113.pdf); [Zhang et al., 2016](http://arxiv.org/pdf/1507.01273.pdf); [Fu et al., 2015](http://arxiv.org/pdf/1509.06841.pdf)) are not currently implemented, but the former
two extensions are in progress.

**Why Caffe?**

Caffe provides a straight-forward interface in python that makes it easy to define, train, and deploy simple architectures.

We plan to add support for TensorFlow to enable training more flexible network architectures.

**How can I find more details on the algorithms implemented in this codebase?**

For the most complete, up-to-date reference, we recommend this paper:

Sergey Levine\*, Chelsea Finn\*, Trevor Darrell, Pieter Abbeel. *End-to-End Training of Deep Visuomotor Policies*. 2015. arxiv 1504.00702. [[pdf](http://arxiv.org/pdf/1504.00702.pdf)]

