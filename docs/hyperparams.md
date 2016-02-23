Configuration & Hyperparameters
===
All of the configuration settings are stored in a             config.py file in the relevant code directory. All             hyperparameters can be changed from the default value             in the hyperparams.py file for a particular  experiment.

This page contains all of the config settings that are exposed                via the experiment hyperparams file. See the                corresponding config files for more detailed comments                on each variable.
*****
#### Algorithm and Optimization

**Algorithm base class**
* initial_state_var
* sample_decrease_var
* kl_step
* init_traj_distr
* sample_increase_var
* cost
* inner_iterations
* dynamics
* min_step_mult
* min_eta
* max_step_mult
* traj_opt

**BADMM Algorithm**
* fixed_lg_step
* exp_step_decrease
* init_pol_wt
* max_policy_samples
* policy_dual_rate
* inner_iterations
* exp_step_lower
* exp_step_increase
* policy_sample_mode
* exp_step_upper
* lg_step_schedule
* policy_dual_rate_covar
* ent_reg_schedule

**LQR Traj Opt**
* del0
* eta_error_threshold
* min_eta

**Caffe Policy Optimization**
* gpu_id
* batch_size
* iterations
* weights_file_prefix
* weight_decay
* init_var
* use_gpu
* ent_reg
* lr
* network_model
* network_arch_params
* lr_policy
* momentum
* solver_type

**Policy Prior & GMM**
* strength
* keep_samples
* max_clusters
* strength
* min_samples_per_cluster
* max_samples
#### Dynamics

**Dynamics GMM Prior**
* max_clusters
* strength
* min_samples_per_cluster
* max_samples
#### Cost Function

**State cost**
* wp_final_multiplier
* ramp_option
* l2
* data_types
* l1
* alpha

**Forward kinematics cost**
* evalnorm
* wp_final_multiplier
* alpha
* target_end_effector
* l2
* ramp_option
* env_target
* wp
* l1

**Action cost**
* wu

**Sum of costs**
* costs
* weights
#### Initialization

**Initial Trajectory Distribution - PD initializer**
* init_action_offset
* pos_gains
* init_var
* vel_gains_mult

**Initial Trajectory Distribution - LQR initializer**
* init_acc
* stiffness
* init_gains
* init_var
* stiffness_vel
* final_weight
#### Agent Interfaces

**Agent base class**
* noisy_body_var
* x0var
* dH
* noisy_body_idx
* smooth_noise_renormalize
* smooth_noise
* smooth_noise_var
* pos_body_offset
* pos_body_idx

**Box2D agent**

**Mujoco agent**
* substeps

**ROS agent**
