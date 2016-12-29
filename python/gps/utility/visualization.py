from gps.proto.gps_pb2 import RGB_IMAGE
from gps.utility.general_utils import compute_distance, mkdir_p
from gps.sample.sample_list import SampleList
import numpy as np
import random
import matplotlib.pyplot as plt


TRIALS = 20
THRESH = {'reacher': 0.05, 'pointmass': 0.1}

def compare_samples_curve(gps, N, agent_config, weight_varying=False, experiment='reacher'):
    """
    Compare samples between IOC and demo policies and visualize them in a plot.
    Args:
        gps: GPS object.
        N: number of samples taken from the policy for comparison
        config: Configuration of the agent to sample.
        weight_varying: whether the experiment is weight-varying or not.
        experiment: whether the experiment is reacher or pointmass.
    """
    pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
    alg_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    alg_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + \
                                            'data_files/algorithm_itr_%02d' % pol_iter + '.pkl')
    alg_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_controller_file'])
    alg_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + \
                                            'data_files_oracle/algorithm_itr_09.pkl')
    algorithms = [alg_ioc, alg_sup, alg_demo, alg_oracle]
    if not weight_varying:
        pos_body_offset = gps._hyperparams['agent']['pos_body_offset']
    M = agent_config['conditions']
    if experiment == 'reacher' and not weight_varying: #reset body offsets
        np.random.seed(101)
        for m in range(M):
            self.agent.reset_initial_body_offset(m)

    policies = [alg.policy_opt.policy for alg in algorithms]
    successes = {i: np.zeros((TRIALS, M)) for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    if not weight_varying:
        ioc_conditions = agent_config['pos_body_offset']
    else:
        ioc_conditions = [np.log10(agent_config['density_range'][i])-4.0 \
                            for i in xrange(M)]
    pos_body_offset = gps.agent._hyperparams['pos_body_offset'][i]
    target_position = np.array([.1,-.1,.01])+pos_body_offset

    for seed in xrange(TRIALS):
        random.seed(seed)
        np.random.seed(seed)
        for i in xrange(M):
            # Gather demos.
            for j in xrange(N):
                for k in xrange(len(policies)):
                    sample = agent.sample(
                        policies[k], i,
                        verbose=(i < gps._hyperparams['verbose_trials']), 
                        noisy=True)
                    dists_to_target = compute_distance(target_position, SampleList([sample]))[0]
                    if dists_to_target <= THRESH[experiment]:
                        successes[k][seed, i] = 1.0
    
    success_rates = [np.mean(success, axis=0) for success in successes]
    print 'ioc: ' + repr(success_rates[0].mean()) + ', ' + repr(success_rates[0])
    print 'sup: ' + repr(success_rates[1].mean()) + ', ' + repr(success_rates[1])
    print 'demo: ' + repr(success_rates[2].mean()) + ', ' + repr(success_rates[2])
    print 'oracle: ' + repr(success_rates[3].mean()) + ', ' + repr(success_rates[3])

    plt.close('all')
    fig = plt.figure(figsize=(8, 5))

    subplt = plt.subplot()
    subplt.plot(ioc_conditions, success_rates[0], '-rx')
    subplt.plot(ioc_conditions, success_rates[1], '-gx')
    subplt.plot(ioc_conditions, success_rates[2], '-bx')
    subplt.plot(ioc_conditions, success_rates[3], '-yx')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(['S3G', 'cost regr', 'RL policy', 'oracle'], loc='lower left')
    plt.ylabel('Success Rate', fontsize=22)
    plt.xlabel('Log Mass', fontsize=22, labelpad=-4)
    plt.title("Generalization for 2-link reacher", fontsize=25)
    plt.savefig(gps._data_files_dir + 'sample_conds_distr.png')
    plt.close('all')

def visualize_samples(gps, N, agent_config, experiment='reacher'):
    """
    Compare samples between IOC and demo policies and visualize them in a plot.
    Args:
        gps: GPS object.
        N: number of samples taken from the policy for comparison
        config: Configuration of the agent to sample.
        experiment: whether the experiment is peg, reacher or pointmass.
    """
    pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
    alg_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    alg_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + \
                                            'data_files/algorithm_itr_%02d' % pol_iter + '.pkl')
    alg_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_controller_file'])
    alg_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + \
                                            'data_files_oracle/algorithm_itr_09.pkl')
    algorithms = [alg_ioc, alg_sup, alg_demo, alg_oracle]
    M = agent_config['conditions']
    policies = [alg.policy_opt.policy for alg in algorithms]
    samples = {i: [] for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    ioc_conditions = [np.array([np.log10(agent_config['density_range'][i]), 0.])
                        for i in xrange(M)]
    print "Number of testing conditions: %d" % M

    if 'record_gif' in gps._hyperparams:
        gif_config = gps._hyperparams['record_gif']
        gif_fps = gif_config.get('fps', None)
        gif_dir = gif_config.get('gif_dir', gps._hyperparams['common']['data_files_dir'])
        mkdir_p(gif_dir)
    for i in xrange(M):
        # Gather demos.
        for j in xrange(N):
            for k in xrange(len(samples)):
                gif_name = os.path.join(gif_dir, 'pol%d_cond%d.gif' % (k, i))
                sample = agent.sample(
                    policies[k], i,
                    verbose=(i < gps._hyperparams['verbose_trials']), noisy=True, 
                    record_image=True, record_gif=gif_name, record_gif_fps=gif_fps
                    )
                samples[k].append(sample)
