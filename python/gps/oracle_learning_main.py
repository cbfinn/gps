import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.optimizers import Adam
policies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..',
                                             'experiments/mjc_pointmass_example/data_files/policies/'))


def main():
    #policy_folders = os.listdir(policies_path)
    #policy_dict_path = policies_path + '/' + policy_folders[0] + '/_pol'
    #pol_dict = pickle.load(open(policy_dict_path, "rb"))
    #print pol_dict.keys()
    train_net()
    #train_net()
    #run_net()


def train_net():

    the_data = np.load(policies_path + '/the_data.npy')
    the_actions = np.load(policies_path + '/the_actions.npy')
    the_goals = np.load(policies_path + '/the_goals.npy')
    X_train_obs = the_data.reshape(the_data.shape[0]*the_data.shape[1], the_data.shape[2])
    X_train_goals = the_goals.reshape(the_goals.shape[0]*the_goals.shape[1], the_goals.shape[2])
    y_train = the_actions.reshape(the_actions.shape[0]*the_actions.shape[1], the_actions.shape[2])
    obs_dims = (4,)
    goal_dims = (2,)
    dim_action = 2

    goal_encoder = Sequential()
    goal_encoder.add(Dense(10, input_shape=goal_dims, activation='relu'))

    obs_encoder = Sequential()
    obs_encoder.add(Dense(30, input_shape=obs_dims, activation='relu'))
    obs_encoder.add(Dense(30, activation='relu'))

    decoder = Sequential()
    decoder.add(Merge([goal_encoder, obs_encoder], mode='concat'))
    decoder.add(Dense(32, activation='relu'))
    decoder.add(Dense(dim_action))

    decoder.compile(loss='mse', optimizer='adam')

    decoder.fit([X_train_goals, X_train_obs], y_train,
                nb_epoch=20, batch_size=64,
                show_accuracy=True, shuffle=True)
    #score = model.evaluate(X_test, y_test, batch_size=16)


def run_net():
    pass


class KerasPolicy:
    def __init__(self):
        pass

    def act(self):
        pass


if __name__ == '__main__':
    main()


