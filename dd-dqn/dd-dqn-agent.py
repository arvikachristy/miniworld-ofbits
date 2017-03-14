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
  def __init__(self, h_size):
    self.num_actions = 6
    self.imageIn = tf.placeholder(shape=[None, 84, 32, 3], dtype=tf.float32)

    self.conv1 = tf.contrib.layers.convolution2d( \
        inputs=self.imageIn,num_outputs=32,kernel_size=[5,5],stride=[3,2],padding='VALID', biases_initializer=None)
    self.conv2 = tf.contrib.layers.convolution2d( \
        inputs=self.conv1,num_outputs=64,kernel_size=[5,4],stride=[3,2],padding='VALID', biases_initializer=None)
    self.conv3 = tf.contrib.layers.convolution2d( \
        inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[2,1],padding='VALID', biases_initializer=None)
    self.conv4 = tf.contrib.layers.convolution2d( \
        inputs=self.conv3,num_outputs=512,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None)

    # We take the output from the final convolutional layer and split it into separate
    # advantage and value streams.
    self.streamAC, self.streamVC = tf.split(self.conv4, num_or_size_splits=2, axis=3)
    self.streamA = tf.contrib.layers.flatten(self.streamAC)
    self.streamV = tf.contrib.layers.flatten(self.streamVC)
    self.AW = tf.Variable(tf.random_normal([h_size/2, self.num_actions]))
    self.VW = tf.Variable(tf.random_normal([h_size/2, 1]))
    self.Advantage = tf.matmul(self.streamA, self.AW)
    self.Value = tf.matmul(self.streamV, self.VW)

    # Then combine them together to get our final Q-values
    self.QOut = self.Value + tf.subtract(self.Advantage,
      tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
    self.predict = tf.argmax(self.QOut, 1)

    # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)

    self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
    self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
    
    self.Q = tf.reduce_sum(tf.multiply(self.QOut, self.actions_onehot), reduction_indices=1)
    
    self.td_error = tf.square(self.targetQ - self.Q)
    self.loss = tf.reduce_mean(self.td_error)
    self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    self.updateModel = self.trainer.minimize(self.loss)

def processState(s, cursorY, cursorX):
    if s is not None and len(s) > 0 and s[0] is not None:
        if type(s[0]) == dict and 'vision' in s[0]:
            v = s[0]['vision']
            crop = v[75:75+210, 10:10+160, :]

            # divide by 5
            lowres = scipy.misc.imresize(crop, (42, 32, 3))
            lowresT = tf.stack(lowres)

            windowX = 32
            windowY = 42
            center = focusAtCursor(crop, cursorY, cursorX, windowY, windowX)

            stacked = tf.stack([center, lowresT], axis=0)
            return tf.reshape(stacked, shape=[84, 32, 3]).eval()
    return np.zeros(shape=[84, 32, 3])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def intToVNCAction(a, x, y):
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
        return [universe.spaces.PointerEvent(x, y, 0)], x, y
    elif a == 2:
        if y + small_step <= maxY:
            return [universe.spaces.PointerEvent(x, y + small_step, 0)], x, y + small_step
    elif a == 3:
        if x + small_step <= maxX:
            return [universe.spaces.PointerEvent(x + small_step, y, 0)], x + small_step, y
    elif a == 4:
        if y - small_step >= minY:
            return [universe.spaces.PointerEvent(x, y - small_step, 0)], x, y - small_step
    elif a == 5:
        if x - small_step >= minX:
            return [universe.spaces.PointerEvent(x - small_step, y, 0)], x - small_step, y
    return [], x, y

def episodeNumber(info, prev):
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


batch_size = 32 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
startE = 1 # Starting chance of random action
endE = 0.1 # Final chance of random action
anneling_steps = 30000. # How many steps of training to reduce startE to endE.
num_episodes = 7000 # How many episodes of game environment to train network with.
pre_train_steps = 500 # How many steps of random actions before training begins.
h_size = 512 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network

load_model = False # Whether to load a saved model.
checkpoint_path_suffix = "dqn-model"
checkpoint_path = checkpoint_path_suffix # The path to save our model to.
evaluation_path_suffix = "evaluation"
evaluation_path = evaluation_path_suffix
plot_vision = False # Plot the agent's view of the environment
save_history = True # If true, write results to file. If false, render environment.
summary_print_freq = 10 # How often (in episodes) to print a summary
summary_freq = 20 # How often (in episodes) to write a summary to a summary file
checkpoint_freq = 100 # How often (in episodes) to save a checkpoint of model parameters

# Add ID to output directory names to distinguish between runs
if len(sys.argv) > 1:
    agent_id = str(sys.argv[1])
    checkpoint_path = agent_id + "-" + checkpoint_path_suffix
    evaluation_path = agent_id + "-" + evaluation_path_suffix


env = gym.make('wob.mini.ClickTest-v0')
# automatically creates a local docker container
env.configure(remotes=1, fps=5,
              vnc_driver='go',
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0,
                          'fine_quality_level': 100, 'subsample_level': 0})

tf.reset_default_graph()
mainQN = QLearner(h_size)
targetQN = QLearner(h_size)

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

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

myBuffer = ExperienceBuffer()
rewards = []

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

successes = 0
fails = 0
misses = 0

with tf.Session() as sess:
    if load_model:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps, sess) #Set the target network to be equal to the primary network.
    i = 0
    s = env.reset()
    prevY = 80+75+50
    prevX = 80+10
    s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    while not isValidObservation(s):
        s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    if not save_history:
        env.render()
    episodeOffset = episodeNumber(info, i)
    print 'Offset:', episodeOffset
    while i < num_episodes:
        episodeBuffer = ExperienceBuffer()

        if type(s) == tf.Tensor:
            s = s.eval()
        while not isValidObservation(s):
            s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
        s = processState(s, prevY, prevX)
        s1 = s
        d = False
        rAll = 0
        j = 0
        i = episodeNumber(info, i + episodeOffset) - episodeOffset
        print '\n------ Episode', i
        print (prevX, prevY)
        #The Q-Network
        while episodeNumber(info, i + episodeOffset) == i + episodeOffset:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a_num = np.random.randint(0,6)
            else:
                QOut = sess.run(mainQN.QOut,feed_dict={mainQN.imageIn:[s]})[0]
                print QOut
                a_num = chooseActionFromQOut(QOut)
                print j, 'Decided',
                action_numbers = {0: 'CLICK', 1: 'STAY', 2: 'UP', 3: 'RIGHT', 4: 'DOWN', 5: 'LEFT'}
                print action_numbers[a_num]
            a, prevX, prevY = intToVNCAction(a_num, prevX, prevY)
            s1, r, d, info = env.step([a])
            if type(s1) == tf.Tensor:
                s1 = s1.eval()
            while not isValidObservation(s1):
                s1, r, d, info = env.step([a])
            if not save_history:
                env.render()
            s1 = processState(s1, prevY, prevX)

            if plot_vision:
                plt.close()
                plt.imshow(s1)
                plt.show(block=False)
            episodeBuffer.add(np.reshape(np.array([s,a_num,r[0],s1,d[0]]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    QOut1 = sess.run(mainQN.QOut,feed_dict={mainQN.imageIn:np.stack(trainBatch[:,3])})
                    Q1 = chooseActionFromQOut(QOut1)
                    Q2 = sess.run(targetQN.QOut,feed_dict={targetQN.imageIn:np.stack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.imageIn:np.stack(trainBatch[:,0]),
                            mainQN.targetQ:targetQ,
                            mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
            rAll += r[0]
            
            if d[0] == True:
                if r[0] == 0:
                    misses += 1
                elif r[0] > 0:
                    successes += 1
                    print 'Success'
                else:
                    fails += 1
                if save_history:
                    history_writer.saveEpisode(i,r[0])
                rewards.append([i, r[0]])
                rewards = rewards[-100:]
                total_steps += j
                print 'Steps taken this episode:', j
            s = s1
        
        #Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)

        #Periodically save the model.
        if save_history and i % checkpoint_freq == 0:
            saver.save(sess, checkpoint_path+'/model', global_step=i)#+'.cptk')
            print "Saved Model"
        if i % summary_print_freq == 0:
            print 'Actions taken', np.sum(jList)
            print 'Average reward (last 100):', np.mean(rList[-100:]), '(last 10)', np.mean(rList[-10:])
            print 'Epsilon:', e
            print 'Successes:', successes, ':', float(successes)/len(rList), '\t', \
            'Fails:', fails, ':', float(fails)/len(rList), '\t', \
            'Misses:', misses, ':', float(misses)/len(rList), '\t' \
            'avg steps/episode (last 100):', np.mean(jList[-100:]), '(last 10)', np.mean(jList[-10:])
            print 'Rewards', rewards
    if save_history:
        saver.save(sess, checkpoint_path+'/model-'+str(i)+'.cptk')
print "Average reward: ", np.mean(rList)
