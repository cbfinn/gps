SSRL
=====

This code in branch SSRL is an implementation of the paper Generalizing Skills with Semi-Supervised Reinforcement Learning (https://arxiv.org/abs/1612.00429). Here are the instructions to run our experiments shown in the paper.

Obstacle
-----

First we need to generate demonstrations using RL algorithm for S3G and cost regression. Run the following line of code in the gps directory:
```Python
python python/gps/gps_main.py mjc_pointmass_wall_example
```

After finish running the RL algorithm, we can run our S3G and cost regression algorithm using the following two lines of code respectively:
```Python
python python/gps/gps_main.py mjc_pointmass_wall_ioc_example
python python/gps/gps_main.py mjc_pointmass_wall_supervised
```
To get the oracle policy, we can change the hyperparameters of agent of mjc_pointmass_wall_example in gps/experiments/mjc_pointmass_wall_example/hyperparam.py to be the same as the hyperparameters of agent of mjc_pointmass_wall_ioc_example in gps/experiments/mjc_pointmass_wall_ioc_example/hyperparam.py and run
```Python
python python/gps/gps_main.py mjc_pointmass_wall_example
```
again to generate the oracle policy.
To visualize the comparison between the 4 policies, we can run the following line of code:
```Python
python python/gps/gps_main.py -m mjc_pointmass_wall_ioc_example
```
Note you can change the conditions in test_xx_range in gps/experiments/mjc_pointmass_wall_ioc_example/hyperparam.py to be the test conditions that you desire and change the hyparameters in compare_samples_curve in gps/python/utility/visualization.py accordingly (now the default is taking 20 test conditions). 

2-link reacher
-----

Similarly, generate demos:
```Python
python python/gps/gps_main.py reacher_mdgps_weight
```

Run S3G and cost regression:
```Python
python python/gps/gps_main.py reacher_mdgps_weight_ioc
python python/gps/gps_main.py reacher_mdgps_weight_supervised
```

Generate oracle policy: change hyperparams of agent in gps/experiments/reacher_mdgps_weight to be the same as gps/experiments/reacher_mdgps_weight_ioc and run:
```Python
python python/gps/gps_main.py reacher_mdgps_weight
```

Visualize comparison: change test conditions in gps/experiments/reacher_mdgps_weight_ioc to be desired test conditions and run:
```Python
python python/gps/gps_main.py -m reacher_mdgps_weight_ioc
```

2-link reacher with vision
-----

Similarly, generate demos:
```Python
python python/gps/gps_main.py reacher_images
```

Run S3G and cost regression:
```Python
python python/gps/gps_main.py reacher_ioc_feat_mdgps
python python/gps/gps_main.py reacher_iocsup_images
```

Generate oracle policy: change hyperparams of agent in gps/experiments/reacher_images to be the same as gps/experiments/reacher_ioc_feat_mdgps and run:
```Python
python python/gps/gps_main.py reacher_images
```

Visualize comparison: change test conditions in gps/experiments/reacher_ioc_feat_mdgps to be desired test conditions and run:
```Python
python python/gps/gps_main.py -m reacher_ioc_feat_mdgps
```


half-cheetah
-----

Similarly, generate demos:
```Python
python python/gps/gps_main.py cheetah_hop
```

Run S3G and cost regression:
```Python
python python/gps/gps_main.py cheetah_hop_ioc
python python/gps/gps_main.py cheetah_hop_supervised
```

Generate oracle policy: change hyperparams of agent in gps/experiments/cheetah_hop to be the same as gps/experiments/cheetah_hop_ioc and run:
```Python
python python/gps/gps_main.py cheetah_hop
```

Visualize comparison: change test conditions in gps/experiments/cheetah_hop_ioc to be desired test conditions and run:
```Python
python python/gps/gps_main.py -m cheetah_hop_ioc
```
Note: Due to the randomness of tensorflow, the result you get when you run S3G and cost regression might be different from the result we showed in the paper, but should still reveal the difference between these methods as we did in the paper.

GPS
======

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.
