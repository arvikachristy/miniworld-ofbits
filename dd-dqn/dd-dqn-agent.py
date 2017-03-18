import gym
import universe # register the universe environments
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import sys

from utils.Output import Output
from ExperienceBuffer import ExperienceBuffer

# DD-DQN implementation based on Arthur Juliani's tutorial "Simple Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond"

class QLearner():
  def __init__(self, h_size, num_actions, zoom_to_cursor):
    if zoom_to_cursor:
        self.imageIn = tf.placeholder(shape=[None, 84, 32, 3], dtype=tf.float32)

        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[5,5],stride=[3,2],padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[5,4],stride=[3,2],padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[2,1],padding='VALID', biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d( \
            inputs=self.conv3,num_outputs=512,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None)
    else:
        self.imageIn = tf.placeholder(shape=[None, 105, 80, 3], dtype=tf.float32)

        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.imageIn,
            num_outputs=32,
            kernel_size=[8,8],
            stride=[4,4],
            padding='VALID',
            biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1,
            num_outputs=64,
            kernel_size=[4,4],
            stride=[2,2],
            padding='VALID',
            biases_initializer=None)
        self.pool1 = tf.nn.max_pool( \
            value=self.conv2,
            ksize=[1,8,5,1],
            strides=[1,1,1,1],
            padding='VALID')
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.pool1,
            num_outputs=64,
            kernel_size=[3,3],
            stride=[1,1],
            padding='VALID',
            biases_initializer=None)
        '''self.pool3 = tf.nn.max_pool( \
            value=self.conv3,
            ksize=[1,8,5,1],
            strides=[1,1,1,1],
            padding='VALID')'''
        self.conv4 = tf.contrib.layers.convolution2d( \
            inputs=self.conv3,#pool3,
            num_outputs=1024,
            kernel_size=[2,2],
            stride=[2,2],
            padding='VALID',
            biases_initializer=None)

    # We take the output from the final convolutional layer and split it into separate
    # advantage and value streams.
    self.streamAC, self.streamVC = tf.split(self.conv4, num_or_size_splits=2, axis=3)
    self.streamA = tf.contrib.layers.flatten(self.streamAC)
    self.streamV = tf.contrib.layers.flatten(self.streamVC)
    self.AW = tf.Variable(tf.random_normal([h_size/2, num_actions]))
    self.VW = tf.Variable(tf.random_normal([h_size/2, 1]))
    self.Advantage = tf.matmul(self.streamA, self.AW)
    self.Value = tf.matmul(self.streamV, self.VW)

    # Then combine them together to get our final Q-values
    self.QOut = self.Value + tf.subtract(self.Advantage,
      tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
    self.predict = tf.argmax(self.QOut, 1)

    # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

    self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
    self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
    
    self.Q = tf.reduce_sum(tf.multiply(self.QOut, self.actions_onehot), reduction_indices=1)
    
    self.td_error = tf.square(self.targetQ - self.Q)
    self.loss = tf.reduce_mean(self.td_error)
    self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    self.updateModel = self.trainer.minimize(self.loss)

def processState(s, zoom_to_cursor, cursorY, cursorX):
    if s is not None and len(s) > 0 and s[0] is not None:
        if type(s[0]) == dict and 'vision' in s[0]:
            v = s[0]['vision']
            crop = v[75:75+210, 10:10+160, :]

            if not zoom_to_cursor:
                lowres = scipy.misc.imresize(crop, (105, 80, 3))
                return lowres

            # divide by 5
            lowres = scipy.misc.imresize(crop, (42, 32, 3))
            lowresT = tf.stack(lowres)

            windowX = 32
            windowY = 42
            center = focusAtCursor(crop, cursorY, cursorX, windowY, windowX)

            stacked = tf.stack([center, lowresT], axis=0)
            return tf.reshape(stacked, shape=[84, 32, 3]).eval()
    if not zoom_to_cursor:
        return np.zeros(shape=[105, 80, 3])
    return np.zeros(shape=[84, 32, 3])

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(sess, op_holder):
    for op in op_holder:
        sess.run(op)

def intToVNCAction(a, include_stay, x, y):
    small_step = 5
    minY = 125
    maxY = 285
    minX = 10
    maxX = 170
    if a == 0:
        return [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, 1),
            universe.spaces.PointerEvent(x, y, 0)], x, y
    elif a == 1:
        if y + small_step <= maxY:
            return [universe.spaces.PointerEvent(x, y + small_step, 0)], x, y + small_step
    elif a == 2:
        if x + small_step <= maxX:
            return [universe.spaces.PointerEvent(x + small_step, y, 0)], x + small_step, y
    elif a == 3:
        if y - small_step >= minY:
            return [universe.spaces.PointerEvent(x, y - small_step, 0)], x, y - small_step
    elif a == 4:
        if x - small_step >= minX:
            return [universe.spaces.PointerEvent(x - small_step, y, 0)], x - small_step, y
    elif a == 5 and include_stay:
        return [universe.spaces.PointerEvent(x, y, 0)], x, y
    return [], x, y

def getEpisodeNumber(info, prev):
    if info['n'][0]['env_status.episode_id'] is not None:
        return int(info['n'][0]['env_status.episode_id'])
    else:
        return prev

def focusAtCursor(imageIn, cursorY, cursorX, windowY, windowX):
    cursorX = cursorX - 10
    cursorY = cursorY - 75
    imageIn = tf.reshape(imageIn, shape=[-1, 210, 160, 3])
    padded = tf.pad(tf.reshape(imageIn[0,:,:,:], shape=[210, 160, 3]),
                    [[windowY/2, windowY/2],[windowX/2, windowX/2],[0, 0]],
                    "CONSTANT")
    result = padded[cursorY:cursorY+windowY, cursorX:cursorX+windowX, :]
    return result

def chooseActionFromSingleQOut(singleQOut):
    unique = np.unique(singleQOut)
    if len(unique) == 1:
        return np.random(0, len(QOut))
    else:
        return np.argmax(singleQOut, 0)

def chooseActionFromQOut(QOut):
    if QOut.ndim == 1:
        return chooseActionFromSingleQOut(QOut)
    else:
        return [chooseActionFromSingleQOut(q) for q in QOut]

def isValidObservation(s):
    return s is not None and len(s) > 0 and s[0] is not None and type(s[0]) == dict and 'vision' in s[0]

def getValidObservation(env, zoom_to_cursor, prevX, prevY):
    s = None
    while not isValidObservation(s):
        s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    s = processState(s, zoom_to_cursor, prevY, prevX)
    return s

def makeEnvironment():
    env = gym.make('wob.mini.ClickTest-v0')
    # automatically creates a local docker container
    env.configure(remotes=1, fps=5,
                  vnc_driver='go',
                  vnc_kwargs={'encoding': 'tight', 'compress_level': 0,
                              'fine_quality_level': 100, 'subsample_level': 0})
    return env

def initEnvironment(env, save_history):
    s = env.reset()
    prevY = 80+75+50
    prevX = 80+10
    s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    while not isValidObservation(s):
        s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    if not save_history:
        env.render()
    ep_num_offset = getEpisodeNumber(info, 0)
    print 'Offset:', ep_num_offset
    return s, info, prevY, prevX, ep_num_offset

def epNumIsConstant(info, ep_num, ep_num_offset):
    return getEpisodeNumber(info, ep_num + ep_num_offset) == ep_num + ep_num_offset

def addReward(r, ep_num, rewards, successes, fails, misses):
    if r == 0:
        misses += 1
    elif r > 0:
        successes += 1
        print 'Success'
    else:
        fails += 1
    rewards.append([ep_num, r])
    rewards = rewards[-100:]
    return rewards, successes, fails, misses

def discountEpsilon(epsilon, step_drop, end_epsilon):
    if epsilon > end_epsilon:
        epsilon -= step_drop
    return epsilon

def trainQNs(sess, mainQN, targetQN, trainBatch, batch_size, y):
    QOut1 = sess.run(mainQN.QOut,feed_dict={mainQN.imageIn:np.stack(trainBatch[:,3])})
    Q1 = chooseActionFromQOut(QOut1)
    Q2 = sess.run(targetQN.QOut,feed_dict={targetQN.imageIn:np.stack(trainBatch[:,3])})
    end_multiplier = -(trainBatch[:,4] - 1)
    doubleQ = Q2[range(batch_size),Q1]
    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
    # Update the network with our target values.
    _ = sess.run(mainQN.updateModel, \
        feed_dict={mainQN.imageIn:np.stack(trainBatch[:,0]),
            mainQN.targetQ:targetQ,
            mainQN.actions:trainBatch[:,1]})

def printSummary(stepList, rList, e, rewards, successes, fails, misses):
    print 'Actions taken', np.sum(stepList)
    print 'Average reward (last 100):', np.mean(rList[-100:]), '(last 10)', np.mean(rList[-10:])
    print 'Epsilon:', e
    print 'Successes:', successes, ':', float(successes)/len(rList), '\t', \
    'Fails:', fails, ':', float(fails)/len(rList), '\t', \
    'Misses:', misses, ':', float(misses)/len(rList), '\t' \
    'avg steps/episode (last 100):', np.mean(stepList[-100:]), '(last 10)', np.mean(stepList[-10:])
    print 'Rewards', rewards

def getOutputDirNames():
    checkpoint_path_suffix = "dqn-model"
    checkpoint_path = checkpoint_path_suffix # The path to save our model to.
    evaluation_path_suffix = "evaluation"
    evaluation_path = evaluation_path_suffix
    tboard_path_suffix = "tboard"
    tboard_path = tboard_path_suffix
    # Add ID to output directory names to distinguish between runs
    if len(sys.argv) > 1:
        agent_id = str(sys.argv[1])
        checkpoint_path = checkpoint_path_suffix + "-" + agent_id
        evaluation_path = evaluation_path_suffix + "-" + agent_id
        tboard_path = tboard_path_suffix + "-" + agent_id
    return checkpoint_path, evaluation_path, tboard_path

def loadModel(sess, saver, checkpoint_path):
    print 'Loading Model...'
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

def plotVision(s):
    plt.close()
    plt.imshow(s)
    plt.show(block=False)


batch_size = 32 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
start_epsilon = 1 # Starting chance of random action
end_epsilon = 0.1 # Final chance of random action
anneling_steps = 100000. # How many steps of training to reduce startE to endE.
num_episodes = 7000 # How many episodes of game environment to train network with.
pre_train_steps = 5000 # How many steps of random actions before training begins.
h_size = 512 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network
num_actions = 6

load_model = False # Whether to load a saved model.
checkpoint_path, evaluation_path, tboard_path = getOutputDirNames()
plot_vision = False # Plot the agent's view of the environment
include_stay = False
if not include_stay:
    num_actions = 5
zoom_to_cursor = False
if not zoom_to_cursor:
    h_size = 1024
save_history = True # If true, write results to file. If false, render environment.
tboard_summaries = True
summary_print_freq = 10 # How often (in episodes) to print a summary
summary_freq = 20 # How often (in episodes) to write a summary to a summary file
checkpoint_freq = 100 # How often (in episodes) to save a checkpoint of model parameters


env = makeEnvironment()

tf.reset_default_graph()
mainQN = QLearner(h_size, num_actions, zoom_to_cursor)
targetQN = QLearner(h_size, num_actions, zoom_to_cursor)

global_step = tf.Variable(0, name='global_step', trainable=False)
init = tf.global_variables_initializer()

# Periodically saves snapshots of the trained CNN weights
# Also responsible for restoring model from checkpoint file
saver = tf.train.Saver()

if save_history:
    # Records the agent's performance (e.g. avg reward)
    history_writer = Output(summaryFreq=summary_freq, outputDirName=evaluation_path)
    #Make a path for our model to be saved in.
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

if tboard_summaries:
    summary_writer = tf.summary.FileWriter(tboard_path)
    if not os.path.exists(tboard_path):
        os.makedirs(tboard_path)

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

myBuffer = ExperienceBuffer()
rewards = []

#Set the rate of random action decrease. 
epsilon = start_epsilon
step_drop = (start_epsilon - end_epsilon)/anneling_steps

#create lists to contain total rewards and steps per episode
stepList = []
rList = []
total_steps = 0

successes = 0
fails = 0
misses = 0

with tf.Session() as sess:
    if load_model:
        loadModel(sess, saver, checkpoint_path)
    sess.run(init)
    total_t = sess.run(tf.contrib.framework.get_global_step())
    updateTarget(sess, targetOps) #Set the target network to be equal to the primary network.
    ep_num = 0
    # Center cursor and wait until state observation s is valid
    s, info, prevY, prevX, ep_num_offset = initEnvironment(env, save_history)

    while ep_num < num_episodes:
        episodeBuffer = ExperienceBuffer()
        s = getValidObservation(env, zoom_to_cursor, prevX, prevY)
        s1 = s
        d = False
        rAll = 0
        step_num = 0
        ep_num = getEpisodeNumber(info, ep_num + ep_num_offset) - ep_num_offset
        print '\n------ Episode', ep_num
        print (prevX, prevY)
        #The Q-Network
        while epNumIsConstant(info, ep_num, ep_num_offset):
            step_num += 1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
                a_num = np.random.randint(0, num_actions)
            else:
                QOut = sess.run(mainQN.QOut,feed_dict={mainQN.imageIn:[s]})[0]
                print QOut
                a_num = chooseActionFromQOut(QOut)
                print step_num, 'Decided',
                if include_stay:
                    action_numbers = {0: 'CLICK', 1: 'UP', 2: 'RIGHT', 3: 'DOWN', 4: 'LEFT', 5: 'STAY'}
                else:
                    action_numbers = {0: 'CLICK', 1: 'UP', 2: 'RIGHT', 3: 'DOWN', 4: 'LEFT'}
                print action_numbers[a_num]
            a, prevX, prevY = intToVNCAction(a_num, include_stay, prevX, prevY)
            s1, r, d, info = env.step([a])
            if type(s1) == tf.Tensor:
                s1 = s1.eval()
            while not isValidObservation(s1):
                s1, r, d, info = env.step([a])
            if not save_history:
                env.render()
            s1 = processState(s1, zoom_to_cursor, prevY, prevX)

            if plot_vision:
                plotVision(s1)
            episodeBuffer.add(np.reshape(np.array([s,a_num,r[0],s1,d[0]]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                epsilon = discountEpsilon(epsilon, step_drop, end_epsilon)
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    trainQNs(sess, mainQN, targetQN, trainBatch, batch_size, y)
                    updateTarget(sess, targetOps) #Set the target network to be equal to the primary network.
            rAll += r[0]
            
            if d[0] == True:
                rewards, successes, fails, misses = addReward(r[0], ep_num, rewards, successes, fails, misses)
                if save_history:
                    history_writer.saveEpisode(ep_num, r[0])
                if tboard_summaries:
                    episode_summary = tf.Summary()
                    episode_summary.value.add(simple_value=r[0], tag="Reward")
                    episode_summary.value.add(simple_value=epsilon, tag="Epsilon")
                    episode_summary.value.add(simple_value=step_num, tag="Actions per episode")
                    summary_writer.add_summary(episode_summary, total_t)
                    summary_writer.flush()
                    total_t += 1
                    print 'Wrote to tboard', tboard_path
                total_steps += step_num
                print 'Steps taken this episode:', step_num
            s = s1
        
        #Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        stepList.append(step_num)
        rList.append(rAll)

        #Periodically save the model.
        if save_history and ep_num % checkpoint_freq == 0:
            saver.save(sess, checkpoint_path+'/model', global_step=ep_num)#+'.cptk')
            print "Saved Model"
        if ep_num % summary_print_freq == 0:
            printSummary(stepList, rList, epsilon, rewards, successes, fails, misses)
    if save_history:
        saver.save(sess, checkpoint_path+'/model-'+str(ep_num)+'.cptk')
print "Average reward: ", np.mean(rList)
