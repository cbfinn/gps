
from gps.agent.config import AGENT, AGENT_BOX2D, AGENT_ROS, AGENT_MUJOCO
from gps.algorithm.config import ALG, ALG_BADMM
from gps.algorithm.traj_opt.config import TRAJ_OPT_LQR
from gps.algorithm.policy.config import INIT_LG_LQR, INIT_LG_PD, POLICY_PRIOR, POLICY_PRIOR_GMM
from gps.algorithm.cost.config import COST_FK, COST_STATE, COST_SUM, COST_ACTION
from gps.algorithm.dynamics.config import DYN_PRIOR_GMM
from gps.algorithm.policy_opt.config import POLICY_OPT_CAFFE

header = "Configuration & Hyperparameters"

overview = "All of the configuration settings are stored in a \
            config.py file in the relevant code directory. All \
            hyperparameters can be changed from the default value \
            in the hyperparams.py file for a particular  experiment.\n\n"

description = "This page contains all of the config settings that are exposed \
               via the experiment hyperparams file. See the \
               corresponding config files for more detailed comments \
               on each variable."

f = open('hyperparams.md','w')
f.write(header + '\n' + '===\n' + overview + description + '\n*****\n')

f.write('#### Algorithm and Optimization\n')

f.write('\n**Algorithm base class**\n')
for key in ALG.keys():
  f.write('* ' + key + '\n')

f.write('\n**BADMM Algorithm**\n')
for key in ALG_BADMM.keys():
  f.write('* ' + key + '\n')

f.write('\n**LQR Traj Opt**\n')
for key in TRAJ_OPT_LQR.keys():
  f.write('* ' + key + '\n')

f.write('\n**Caffe Policy Optimization**\n')
for key in POLICY_OPT_CAFFE.keys():
  f.write('* ' + key + '\n')

f.write('\n**Policy Prior & GMM**\n')
for key in POLICY_PRIOR.keys():
  f.write('* ' + key + '\n')
for key in POLICY_PRIOR_GMM.keys():
  f.write('* ' + key + '\n')

f.write('#### Dynamics\n')

f.write('\n**Dynamics GMM Prior**\n')
for key in DYN_PRIOR_GMM.keys():
  f.write('* ' + key + '\n')

f.write('#### Cost Function\n')

f.write('\n**State cost**\n')
for key in COST_STATE.keys():
  f.write('* ' + key + '\n')

f.write('\n**Forward kinematics cost**\n')
for key in COST_FK.keys():
  f.write('* ' + key + '\n')

f.write('\n**Action cost**\n')
for key in COST_ACTION.keys():
  f.write('* ' + key + '\n')

f.write('\n**Sum of costs**\n')
for key in COST_SUM.keys():
  f.write('* ' + key + '\n')

f.write('#### Initialization\n')

f.write('\n**Initial Trajectory Distribution - PD initializer**\n')
for key in INIT_LG_PD.keys():
  f.write('* ' + key + '\n')

f.write('\n**Initial Trajectory Distribution - LQR initializer**\n')
for key in INIT_LG_LQR.keys():
  f.write('* ' + key + '\n')

f.write('#### Agent Interfaces\n')

f.write('\n**Agent base class**\n')
for key in AGENT.keys():
  f.write('* ' + key + '\n')

f.write('\n**Box2D agent**\n')
for key in AGENT_BOX2D.keys():
  f.write('* ' + key + '\n')

f.write('\n**Mujoco agent**\n')
for key in AGENT_MUJOCO.keys():
  f.write('* ' + key + '\n')

f.write('\n**ROS agent**\n')
for key in AGENT_ROS.keys():
  f.write('* ' + key + '\n')

f.close()
