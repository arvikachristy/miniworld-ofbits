import universe
import numpy as np
import warnings

from rl.agents import DDPGAgent
from rl.core import Processor
from copy import deepcopy
from keras.callbacks import History
from rl.callbacks import Visualizer, CallbackList
from keras import backend as K
K.set_learning_phase(1)

INPUT_SHAPE = (160, 160, 3)


class MiniWorldDDPGAgent(DDPGAgent):
    def _convert_action(self, action):
        # add some noise manually
        action = np.array(action) + np.random.normal(0, 0.05, 2)
        action = [a * 160 for a in action]
        xcoord = int(min(max(0, action[0]), 160)) + 10
        ycoord = int(min(max(0, action[1]), 160)) + 75 + 50
        vnc_action = [[universe.spaces.PointerEvent(xcoord, ycoord, 0),
                       universe.spaces.PointerEvent(xcoord, ycoord, 1),
                       universe.spaces.PointerEvent(xcoord, ycoord, 0)]]
        print(vnc_action)
        return vnc_action

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None,
            verbose=1, visualize=False, nb_max_start_steps=0,
            start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        # if verbose == 1:
        #     callbacks += [TrainIntervalLogger(interval=log_interval)]
        # elif verbose > 1:
        #     callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    assert observation is not None
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 \
                        else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(
                            self._convert_action(action))
                        env.render()
                        while info.get('env_status.env_state') is None:
                            observation, r, done, info = env.step(
                                self._convert_action(action))
                            env.render()
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info = \
                                self.processor.process_step(
                                    observation, reward, done, info)
                            done = done[0]
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(
                                    observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                K.set_learning_phase(1)
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(
                        self._convert_action(action))
                    env.render()
                    while info.get('env_status.env_state') is None:
                        observation, r, done, info = env.step(
                            self._convert_action(action))
                        env.render()
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(
                            observation, r, done, info)
                        done = done[0]
                        print(r, done, info)
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                K.set_learning_phase(1)
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode
                }
                print(action, reward)
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    K.set_learning_phase(1)
                    self.forward(observation)
                    K.set_learning_phase(1)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history


class MiniWorldProcessor(Processor):
    def process_observation(self, observation):
        for ob in observation:
            if ob is not None:
                x = ob['vision']
                crop = x[125:125 + 160, 10:10 + 160, :]
            else:
                crop = np.zeros((160, 160, 3))
        return np.array(crop).astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        print(reward)
        return np.sum(reward)
