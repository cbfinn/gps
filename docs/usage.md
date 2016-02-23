Intended Usage
==============

The intended command line usage is through the gps_main.py script:
```sh
cd /path/to/gps
python python/gps/gps_main.py -h
usage: gps_main.py [-h] [-n] [-t] [-r N] experiment

Run the Guided Policy Search algorithm.

positional arguments:
  experiment         experiment name

optional arguments:
  -h, --help         show this help message and exit
  -n, --new          create new experiment
  -t, --targetsetup  run target setup
  -r N, --resume N   resume training from iter N
```
Usage:
* `python python/gps/gps_main.py <EXPERIMENT_NAME> -n`

  Creates a new experiment folder at  `experiments/<EXPERIMENT_NAME>` with an empty hyperparams file `hyperparams.py`. Copy and paste your old `hyperparams.py` from your previous experiment and make any modifications.

* `python python/gps/gps_main.py <EXPERIMENT_NAME> -t` (for ROS only)

  Opens the Target Setup GUI, for target setup when using ROS. See the [Target Setup GUI](gui.html#target-setup-gui-for-ros-only) section for details.

* `python python/gps/gps_main.py <EXPERIMENT_NAME>`

  Opens the GPS Training GUI and runs the guided policy search algorithm for your specific experiment hyperparams. See the [Training GUI](gui.html#gps-training-gui) section for details.

* `python python/gps/gps_main.py <EXPERIMENT_NAME> -r N`

  Resumes the guided policy search algorithm, loading the algorithm state from iteration N. (The file `experiments/<EXPERIMENT_NAME>/data_files/algorithm_itr_<N>.pkl` must exist.)


For your reference, your experiments folder contains the following:

  * `data_files/` - holds the data files.
    * `data_files/algorithm_itr_<N>.pkl` - the algorithm state at iteration N.
    * `data_files/traj_samples_itr_<N>.pkl` - the trajectory samples collected at iteration N.
    * `data_files/pol_samples_itr_<N>.pkl` - the policy samples collected at iteration N.
    * `data_files/figure_itr_<N>.png` - an image of the GPS Training GUI figure at iteration N.
  * `hyperparams.py` - the hyperparams used for this experiment. For more details, see [this page](hyperparams.html).
  * `log.txt` - the log text of output from the Target Setup GUI and the GPS Training GUI.
  * `targets.npz` - the initial and target state used for this experiment (ROS agent only -- set for other agents in hyperparams.py)

