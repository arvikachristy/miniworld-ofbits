import sys
import os
import gym
import universe 
import itertools
import tensorflow as tf 
import numpy as np
import random

from collections import deque, namedtuple

# Global dictionary with set parameters for the Deep-Q Network
ri_dict = {
  "num_Episodes" : 1000,
  'num_Actions' : 5,
  'replay_memory_size' : 5000,
  'replay_memory_init_size' : 500,
  'update_target_estimator_every' : 1000,
  'discount_factor' : 0.90,
  'epsilon_start' : 1.0,
  'epsilon_end': 0.1,
  'epsilon_decay_steps': 5000,
  'batch_size' : 32,
  'startState' : 1
}

# Global dictionary that stores the x-coordinate and y-coordinate values # 
cord_dict = {
  "xCord" : tf.Variable(tf.zeros([1],dtype = tf.int32), name = "xCord"),
  "yCord" : tf.Variable(tf.zeros([1],dtype = tf.int32), name = "yCord")
  }

# List of discrete actions (will later attempt to look into continuous action spaces) # 
# Very imoortant that the list of discrete actions start indexing from 0 rather than 1 due to batch probability calculation purposes #
act_list = {
  0 : "click",
  1 : "up",
  2 : "down",
  3 : "left",
  4 : "right"
}

#The prepState class takes the vision object of a state (the pixels) and downsamples the image to an 80x80 grayscale image 

class prepState():
  #Downsample a provided state from [160,160,3] to [80,80,1] (Grayscale)
  def __init__(self):
    self.input_state = tf.placeholder(shape = [210,160,3], dtype = tf.uint8)
    self.output = tf.image.rgb_to_grayscale(self.input_state)
    self.output = tf.image.crop_to_bounding_box(self.output, 50, 0, 160, 160)
    self.output = tf.image.resize_images(self.output,[80,80], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    self.output = tf.squeeze(self.output)

  def process(self, sess, state):
    crop = state[75:75+210, 10:10+160, :]
    return sess.run(self.output, feed_dict = {self.input_state: crop})

#The Q Network class
class qNetwork():
  def __init__(self, scope = "estimator"):
    self.scope = scope
    with tf.variable_scope(scope):
      self._build_model()

  def _build_model(self):
    #Placeholders for the input
    #The input will be 4 80x80 Grayscale images
    self.X_pl = tf.placeholder(shape = [None,80,80,4], dtype = tf.uint8, name = "X")
    #The Temporal-Difference target value
    self.y_pl = tf.placeholder(shape = [None], dtype = tf.float32, name = "y")
    #Integer ID of which action was selected
    self.actions_pl = tf.placeholder(shape = [None], dtype = tf.int32, name = "actions")

    X = tf.to_float(self.X_pl) #Simply converts X_pl to float
    batch_size = tf.shape(self.X_pl)[0] #Gets the first index from the shape of X_pl (number of batches)

    #Three convolutional layers
    conv1 = tf.contrib.layers.convolution2d(X,32, 8, 4, activation_fn=tf.nn.relu)
    conv2 = tf.contrib.layers.convolution2d(conv1,64, 4, 2, activation_fn=tf.nn.relu)
    conv3 = tf.contrib.layers.convolution2d(conv2,64, 3, 1, activation_fn=tf.nn.relu)

    #Fully connected layers
    flattened = tf.contrib.layers.flatten(conv3)
    fcl = tf.contrib.layers.fully_connected(flattened, 512)
    self.predictions = tf.contrib.layers.fully_connected(fcl, ri_dict["num_Actions"])

    #Get the predictions for the chosen actions only
    gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
    self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

    #Calculate the loss
    self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
    self.loss = tf.reduce_mean(self.losses)

    #Optimizer parameters
    self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

  def predict(self, sess, s):
    # Predicts action values for a given state 's' #
    return sess.run(self.predictions, {self.X_pl : s})

  def update(self, sess, s, a, y):
    # Updates the network based upon the given targets #
    # Arguments : s (state input), a(chosen action for state [batch_size]), y (TD target of shape [batch_size])

    # We squeeze 'a' as it contains one single dimension entry that is not necesssary
    a = np.squeeze(a)
    # We remove the last dimension from 'y' as it is essentially a duplicate
    y = y[:, 0]
    feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
    global_step, _, loss = sess.run([tf.contrib.framework.get_global_step(), self.train_op, self.loss], feed_dict)
    return loss

def copy_model_parameters(sess, estimator1, estimator2):
  #Copies the model parameters of one qNet to another

  e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
  e1_params = sorted(e1_params, key = lambda v:v.name)
  e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
  e2_params = sorted(e2_params, key = lambda v:v.name)

  update_ops = []
  for e1_v, e2_v in zip(e1_params,e2_params):
    op = e2_v.assign(e1_v)
    update_ops.append(op)

  sess.run(update_ops)

def make_epsilon_greedy_policy(qnetwork, numA):
  #Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.#
  def policy_fn(sess, observation, epsilon):
    A = np.ones(numA, dtype = float) * epsilon / numA
    q_values = qnetwork.predict(sess, np.expand_dims(observation,0))[0]
    best_action = np.argmax(q_values)
    print "Best Action <---------------------------------------------------------------------------------------"
    print best_action
    A[best_action] += (1.0 - epsilon)
    return A
  return policy_fn

# newCord updates the x-Cordinates and y-Cordinates based on the numeric action parameter. Also contains bounds to prevent the calculated coordinates from overlapping the 
# action space

def newCord(act):
  if act == 1:
    x = tf.subtract(cord_dict['yCord'],20).eval()
    y = 125
    if tf.less_equal(x,y).eval() == 1:
      print "Position unchanged"
    else:
      sess.run(cord_dict["yCord"].assign_sub([20]))
  elif act == 2:
    x = tf.add(cord_dict['yCord'],20).eval()
    y = 285 
    if tf.greater_equal(x,y).eval() == 1:
      print "Position unchanged"
    else:
      sess.run(cord_dict["yCord"].assign_add([20]))
  elif act == 3:
    x = tf.add(cord_dict['xCord'],20).eval()
    y = 170
    if tf.greater_equal(x,y).eval() == 1:
      print "Position unchanged"
    else:
      sess.run(cord_dict["xCord"].assign_add([20]))
  elif act == 4:
    x = tf.subtract(cord_dict['xCord'],20).eval()
    y = 10
    if tf.less_equal(x,y).eval() == 1:
      print "Position unchanged"
    else:
      sess.run(cord_dict["xCord"].assign_sub([20]))
  elif act == 0:
    print "Click"

# randCord randomises the x-cordinate and y-cordinate values # 
def randCord():
  sess.run(cord_dict["xCord"].assign(tf.random_uniform([1],minval = 10, maxval = 170, dtype = tf.int32)))
  sess.run(cord_dict["yCord"].assign(tf.random_uniform([1],minval =125, maxval = 285, dtype = tf.int32)))

# chooseAction produces a random integer (representing the action to take) based on the action probabilities table
def chooseAction(action_prob):
  if ri_dict["startState"] == 1:
    ri_dict["startState"] = 0
    return -1
  elif ri_dict["startState"] == 0:
    act = np.random.choice(np.arange(0,5), p = action_prob)
    return act

#doAction returns the next action
def doAction(act_num):
  if act_num == -1:
    randCord()
      # 1. move to x,y with left button released, and click there (2. and 3.)
    action = [universe.spaces.PointerEvent(cord_dict["xCord"].eval(), cord_dict["yCord"].eval(),0)]
    return action
  elif act_num == 0:
    action = [universe.spaces.PointerEvent(cord_dict["xCord"].eval(), cord_dict["yCord"].eval(),1),
              universe.spaces.PointerEvent(cord_dict["xCord"].eval(), cord_dict["yCord"].eval(),0)]
    return action
  else:
    newCord(act_num)
    action = [universe.spaces.PointerEvent(cord_dict["xCord"].eval(), cord_dict["yCord"].eval(),0)]
    return action

def deep_q_learning(sess, env,experiment_dir,qnetwork, targetNetwork, state_proc):
  #Uses the namedTuple module to initialize Experience Replay tuple
  Transition = namedtuple("Transition", ["state","action","reward","next_state","done"])

  #The replay memory is initialized
  replay_memory = []

  #Create directories for checkpoints
  checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
  checkpoint_path = os.path.join(checkpoint_dir, "model")

  #This is not in use yet
  monitor_path = os.path.join(experiment_dir, "monitor") 

  if not os.path.exists(checkpoint_dir):
    print "Missing checkpoint_dir, creating one now <------------"
    os.makedirs(checkpoint_dir)
  if not os.path.exists(monitor_path):
    print "Missing monitor_path, creating one now <------------"
    os.makedirs(monitor_path)

  saver = tf.train.Saver()
  #Load a previous checkpoint if we find one
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  if latest_checkpoint:
    print("Loading model checkpoint {}...\n".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

  #Get the current time step
  total_t = sess.run(tf.contrib.framework.get_global_step())

  #The epsilon decay schedule
  epsilons = np.linspace(ri_dict["epsilon_start"],ri_dict["epsilon_end"], ri_dict["epsilon_decay_steps"])

  #The policy we're following
  policy = make_epsilon_greedy_policy(qnetwork, ri_dict["num_Actions"])

  #Populate the replay memory with initial experience
  print "Populating replay memory"
  state = env.reset()

  #There are instances were the states returned are [None]. This loop will iterate until a legitimate state is returned
  if state == [None]:
    while state == [None]:
      state, reward, done, _ = env.step([[]])
      env.render()

  state = state_proc.process(sess, (state[0])['vision'])
  
  # We stack to take 4 instances of states
  state = np.stack([state] * 4, axis = 2)
  
  env.render()

  #The following loop is to simply fill the replay memory with transition tuples. The number of iterations is stated in the initial parameters
  for i in range(ri_dict["replay_memory_init_size"]):
    print "Still in replay memory initialization <---------------------------------------------------------------------------------------"
    print i 
    print epsilons[min(total_t, ri_dict["epsilon_decay_steps"]-1)]

    #Update action probabilities table
    action_probs = policy(sess, state, epsilons[min(total_t, ri_dict["epsilon_decay_steps"]-1)])
    print action_probs
    act = chooseAction(action_probs)
    action = [doAction(act)]

    #Take a step in the environment from the predicted action
    next_state, reward, done, _ = env.step(action)
    if next_state == [None]:
      while next_state == [None]:
        next_state, reward, done, _ = env.step([[]])
        env.render()
    env.render()
    next_state = state_proc.process(sess, (next_state[0])['vision'])
    next_state = np.append(state[:,:,1:], np.expand_dims(next_state,2),axis = 2)

    #Add transition to the replay memory as long as it is not the starting transition
    if act != -1:
      replay_memory.append(Transition(state,act,reward,next_state,done))
    
    #Reset the environment if done
    if done:
      state = env.reset()
      if state == [None]:
        while state == [None]:
          state, reward, done, _ = env.step([[]])
      state = state_proc.process(sess, (state[0])['vision'])
      state = np.stack([state] * 4, axis = 2)
      env.render()
    else:
      state = next_state
      env.render()

  #Once replay memory initialization is completed, run the training process
  for i in range(ri_dict["num_Episodes"]):
    print "Running training steps <---------------------------------------------------------------------------------------"
    #Save the current checkpoint
    saver.save(tf.get_default_session(), checkpoint_path)

    #Reset the environment
    state = env.reset()
    if state == [None]:
      while state == [None]:
        state, reward, done, _ = env.step([[]])
        env.render()
    env.render()
    state = state_proc.process(sess, (state[0])['vision'])
    state = np.stack([state] * 4, axis = 2)
    loss = None

    #One step in the environment
    for t in itertools.count():
      #Epsilon for this time step
      epsilon = epsilons[min(total_t, ri_dict["epsilon_decay_steps"]-1)]

      #Update the target estimator
      if total_t % ri_dict["update_target_estimator_every"] == 0:
        copy_model_parameters(sess, q_net, target_net)
        print "\nCopied model parameters to target network",

      #Print current step we are on
      print "\rStep {} ({}) @ Episode {} / {}, loss {}".format(t,total_t,i+1,ri_dict["num_Episodes"],loss),
      sys.stdout.flush()

      #Take a step
      print epsilon
      action_probs = policy(sess, state, epsilon)
      print action_probs
      act = chooseAction(action_probs)
      action = [doAction(act)]
      next_state, reward, done, _ = env.step(action)
      if next_state == [None]:
        while next_state == [None]:
          next_state, reward, done, _ = env.step([[]])
          env.render()
      env.render()
      next_state = state_proc.process(sess, (next_state[0])['vision'])
      next_state = np.append(state[:,:,1:], np.expand_dims(next_state,2),axis = 2)

      #If our replay memory is full, pop the first element
      if len(replay_memory) == ri_dict["replay_memory_size"]:
        replay_memory.pop(0)

      #Save transition to replay memory
      if act != -1:
        replay_memory.append(Transition(state,act,reward,next_state,done))

      #Sample a minibatch from the replay memory
      samples = random.sample(replay_memory, ri_dict["batch_size"])
      states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

      #Calculate the q values and targets
      q_values_next = targetNetwork.predict(sess, next_states_batch)
      targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * ri_dict["discount_factor"] * np.amax(q_values_next, axis = 1)

      #Perform gradient descent update
      states_batch = np.array(states_batch)
      loss = qnetwork.update(sess, states_batch, action_batch, targets_batch)

      if done[0] == True:
        break

      state = next_state
      total_t += 1

env = gym.make('wob.mini.ClickTest-v0')
env.configure(remotes=1, fps=15,
              vnc_driver='go', 
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 
                          'fine_quality_level': 100, 'subsample_level': 0})

#Where we will save our checkpoints
experiment_dir = os.path.abspath("./")

#Create global step variabled
global_step = tf.Variable(0, name = 'global_step', trainable = False)

#Create networks
q_net = qNetwork(scope = "q")
target_net = qNetwork(scope = "target_q")

#State processor
state_processor = prepState()

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for t in deep_q_learning(sess, env,experiment_dir, q_net, target_net, state_processor):
      print "Next t"