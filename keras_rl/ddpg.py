import universe # should be imported before tensorflow
import numpy as np
import gym
from keras.models import Model
from keras.layers import Input, merge, GaussianNoise
from keras.layers import Dense, Activation, Flatten, Convolution2D, Reshape
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from agents import MiniWorldDDPGAgent, MiniWorldProcessor
from keras import backend as K
K.set_learning_phase(1)


INPUT_SHAPE = (160, 160, 3)

ENV_NAME = 'wob.mini.ClickTest-v0'
gym.undo_logger_setup()
env = gym.make(ENV_NAME)
# automatically creates a local docker container
env.configure(remotes=1, fps=5,
              vnc_driver='go',
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0,
                          'fine_quality_level': 100, 'subsample_level': 0})

np.random.seed(123)
env.seed(123)
nb_actions = 2

actor_observation_input = Input(
    shape=(1,) + INPUT_SHAPE, name='actor_observation_input')
x = Reshape(INPUT_SHAPE)(actor_observation_input)
x = Convolution2D(32, 8, 8, subsample=(4, 4))(x)
x = Activation('relu')(x)
x = Convolution2D(64, 3, 3, subsample=(1, 1))(x)
x = Activation('relu')(x)
x = Flatten()(x)
x = Dense(256)(x)
x = Activation('tanh')(x)
x = Dense(nb_actions)(x)
x = Activation('sigmoid')(x)
# x = GaussianNoise(stddev=1)(x)
actor = Model(input=actor_observation_input, output=x)
print(actor.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(
    shape=(1,) + INPUT_SHAPE, name='observation_input')

x = Reshape(INPUT_SHAPE)(observation_input)
x = Convolution2D(32, 8, 8, subsample=(4, 4))(x)
x = Activation('relu')(x)
x = Convolution2D(64, 3, 3, subsample=(1, 1))(x)
x = Activation('relu')(x)
x = Flatten()(x)
x = Dense(32)(x)
x = Activation('relu')(x)

x = merge([action_input, x], mode='concat')
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

memory = SequentialMemory(limit=100000, window_length=1)
# random_process = OrnsteinUhlenbeckProcess(
#     size=nb_actions, theta=.15, mu=0., sigma=.05)

processor = MiniWorldProcessor()
agent = MiniWorldDDPGAgent(nb_actions=nb_actions,
                           actor=actor, critic=critic,
                           critic_action_input=action_input,
                           memory=memory, processor=processor,
                           nb_steps_warmup_critic=100,
                           nb_steps_warmup_actor=100,
                           random_process=None, gamma=.99,
                           target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=50000, visualize=True,
          verbose=2, nb_max_episode_steps=10)
