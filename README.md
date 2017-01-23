aSSRL
=====

This code in branch SSRL is an implementation of the paper Generalizing Skills with Semi-Supervised Reinforcement Learning (https://arxiv.org/abs/1612.00429). Here are the instructions to run our experiments shown in the paper.

Obstacle
-----
![](https://ab9c5ab5-a-62cb3a1a-s-sites.googlegroups.com/site/semisupervisedrl/home/wallrl.crop.itr15.cond0.samp0.gif?attachauth=ANoY7copBCmgP3zMuUPoQ6PmRJXb8Wx9JvE6evoWTndy23CSLyZv29j9VNmXlrdVEOjjwi4yyw_OS5igWrXQHAgAyDdsVbiUo0XJ49wNcA0QvsAt1_FOcQWMDPhNvi1A3emOMI1G0YRnEGWC_D8KL_7Cn2JtW0-wOX2Ufp4X4HO66EDNzpuloAjK7lkH9v7npaGKhKSYT7M1uHMqQhT_RfbCRpgdL9HV5MAMikAfs7M07abeM2ab3ClMKM2GYNsQLnBQ1TTzIAmj&attredirects=0&height=138&width=200)

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
python python/gps/gps_main.py -m [number of samples per condition] mjc_pointmass_wall_ioc_example
```
Note you can change the conditions in test_xx_range in gps/experiments/mjc_pointmass_wall_ioc_example/hyperparam.py to be the test conditions that you desire and change the hyparameters in compare_samples_curve in gps/python/utility/visualization.py accordingly (now the default is taking 20 test conditions). 

After running the above line of code, you could get the visualization of comparison between 4 policies like this:
![](https://raw.githubusercontent.com/tianheyu927/gps/ssrl_work/images/obstacle_sample_conds_distr.png)

2-link reacher
-----
<img src="https://ab9c5ab5-a-62cb3a1a-s-sites.googlegroups.com/site/semisupervisedrl/home/crop.pol0_cond4.gif?attachauth=ANoY7crkivCCQGSnu9wD24dlVks1zNSnFcEo3vZZv_aMiDP3Y1DW9z6gfoYN_A3KKGp8ilkC0nluNeqRolqWGGHzyosMgjIOj_CSx3dS33sLnRY_wcwx8DzWMlcIO5IinyoLCrzOlA3CpV7pC1KX3CQ1ZLar_N_T7CB7hdxSnTlXyHu6Lt1Mk-ATWcCA0iAEcMQPa2zDWjezYXuefRU5uHiUDTWv0XsFPOGwlTRlSqf5CMCDH7knGBE%3D&attredirects=0&height=200&width=200" width="400" height="400">

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
python python/gps/gps_main.py -m [number of samples per condition] reacher_mdgps_weight_ioc
```
After running the above line of code, you could get the visualization of comparison between 4 policies like this:
![](https://raw.githubusercontent.com/tianheyu927/gps/ssrl_work/images/sample_conds_distr.png)

2-link reacher with vision
-----
<img src="https://ab9c5ab5-a-62cb3a1a-s-sites.googlegroups.com/site/semisupervisedrl/home/s3g8.gif?attachauth=ANoY7cplieTFzOcP90391rQD_arGGnbAllC6n0GrbcU6THoxV2pmlNDRgTdxZGerDsue02fb7J2KEYXGhopN4AbKYCOOW2kb3A8hXgw9uLRaoM99q3IMaYIC_8hpok7mTNpPJDTrGwEFUoelUuhshAn_peKGNur8SGbz_BLuysU5QPOr7EAYg-SRoLY2-ATTx0ZHriF0sCbNoWbPaKCOLfaiHkHmV8mrfQ%3D%3D&attredirects=0&width=200" width="400" height="400">

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
python python/gps/gps_main.py -m [number of samples per condition] reacher_ioc_feat_mdgps
```


half-cheetah
-----
![](https://ab9c5ab5-a-62cb3a1a-s-sites.googlegroups.com/site/semisupervisedrl/home/crop.itr30.cond0.samp0.gif?attachauth=ANoY7coxlUg1sMxgVpu5Xj9rogtPJwwQ0A_zdyTdQMW7P90uML0ZnjKo_vW3Bh9wE6XGWei6WBnCBVxZDXyziILl_rPb6__zocPGKSMxvjxc3nXA5jlhg0D2D4h9V_z87r5J8G3RvuJVhjrdADtfUpZi5gbN3mW1wYlwAo53hj8d3yG3gfA8Aea04OPqCAtPFi3Jp9V-3AL4pkt-TruPwhMe0HtjHck8L8_F8bUigEqfd3SF64fcbi-F5Jxe0VJeGJkdXrR74ukW&attredirects=0&width=210)

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
python python/gps/gps_main.py -m [number of samples per condition] cheetah_hop_ioc
```
Note: Due to the randomness of tensorflow, the result you get when you run S3G and cost regression might be different from the result we showed in the paper, but should still reveal the difference between these methods as we did in the paper.

After running the above line of code, you could get the visualization of comparison between 4 policies like this:
![](https://raw.githubusercontent.com/tianheyu927/gps/ssrl_work/images/cheetah_sample_conds_distr.png)

GPS
======

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.
