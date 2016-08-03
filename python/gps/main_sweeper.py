""" This file is for performing hyperparameter sweeps of GPS, calling GPS main several times. """

import argparse
import copy
import imp
import itertools
import os

from gps_main import main as gps_main

def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the GPS algorithm with hyperparameter sweeps.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    args = parser.parse_args()
    args.resume = None
    args.policy = 0
    args.quit = True
    args.targetsetup = False

    exp_name = args.experiment
    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'
    config = imp.load_source('hyperparams', hyperparams_file).config

    all_configs = []

    # for now assuming that the sweep values dict only has one key in it
    if 'sweep_values' not in config:
        print('WARNING - no sweep values detected')
        all_configs.append(config)
        sweep_values = [""]
    else:
        sweep_info = config['sweep_values']
        sweep_values = list(itertools.product(*sweep_info.values()))

        for values in sweep_values:
            conf_copy = copy.deepcopy(config)
            for key, value in zip(sweep_info.keys(), values):
                # convert from X.Y.Z to dict['X']['Y']['Z']
                nested_keys = key.split('.')
                if len(nested_keys) == 1:
                    conf_copy[nested_keys[0]] = value
                elif len(nested_keys) == 2:
                    conf_copy[nested_keys[0]][nested_keys[1]] = value
                elif len(nested_keys) == 3:
                    conf_copy[nested_keys[0]][nested_keys[1]][nested_keys[2]] = value
                elif len(nested_keys) == 4:
                    conf_copy[nested_keys[0]][nested_keys[1]][nested_keys[2]][nested_keys[3]] = value
                else:
                    raise ValueError('too many nested keys - unsupported')

            all_configs.append(conf_copy)
            import pdb; pdb.set_trace()

    for i, config in zip(range(len(all_configs)), all_configs):
        data_dir = 'data_files_' + str(i) + '/'
        config['common']['no_sample_logging'] = True
        config['common']['data_files_dir'] = config['common']['experiment_dir'] + data_dir
        if not os.path.exists(config['common']['data_files_dir']):
            os.makedirs(config['common']['data_files_dir'])
        config['common']['log_filename'] = config['common']['data_files_dir'] + 'log.txt'
        # put info regarding which hyperparams we're running in log or somewhere else (str(sweep_values[i]))

        gps_main(args, config)
        # TODO - compile summary stats here.


if __name__ == "__main__":
    main()
