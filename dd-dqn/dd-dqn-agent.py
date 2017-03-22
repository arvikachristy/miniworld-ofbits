import gym
import universe # register the universe environments
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.misc
import os
import sys

from utils.Output import Output
from ExperienceBuffer import ExperienceBuffer

# DD-DQN implementation based on Arthur Juliani's tutorial "Simple Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond"

class QLearner():
  def __init__(self, h_size, num_actions, zoom_to_cursor, include_rgb, include_prompt):
    z = 1
    if include_rgb:
        z = 3
    if zoom_to_cursor:
        height = 64
        if include_prompt:
            height = 84
        self.imageIn = tf.placeholder(shape=[None, height, 32, z], dtype=tf.float32)

        # if include_prompt, 1 is added to the dimensions below to obtain desired output size
        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[5,5],stride=[2+include_prompt,2],padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[2,1],padding='VALID', biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d( \
            inputs=self.conv3,num_outputs=256,kernel_size=[3,3],stride=[2,2],padding='VALID', biases_initializer=None)

    else:
        height = 80
        pool1_ksize_y = 5
        if include_prompt:
            height = 105
            pool1_ksize_y = 8
        self.imageIn = tf.placeholder(shape=[None, height, 80, z], dtype=tf.float32)

        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None)
        self.pool1 = tf.nn.max_pool( \
            value=self.conv2,ksize=[1,pool1_ksize_y,5,1],strides=[1,1,1,1],padding='VALID')
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.pool1,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID',biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d( \
            inputs=self.conv3,num_outputs=1024,kernel_size=[2,2],stride=[2,2],padding='VALID',biases_initializer=None)

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

def rgbToGrayscale(img):
    h, w, _ = img.shape
    gray = np.zeros(shape=[h, w, 1], dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            px = img[y][x]
            gray[y, x] = int(0.299*px[0] + 0.587*px[1] + 0.114*px[2])
    return gray

def processState(sess, s, zoom_to_cursor, include_rgb, include_prompt, cursorY, cursorX):
    if s is not None and len(s) > 0 and s[0] is not None:
        if type(s[0]) == dict and 'vision' in s[0]:
            v = s[0]['vision']
            crop = v[75:75+210, 10:10+160, :]
            width = 160
            if include_prompt:
                height = 210
            else:
                height = 160
                crop = crop[50:, :, :]

            if not zoom_to_cursor:
                lowres = scipy.misc.imresize(crop, (height/2, width/2, 3))
                if not include_rgb:
                    # Convert to grayscale using weighted average
                    lowres = rgbToGrayscale(lowres)
                return lowres #/ 25.5

            # divide by 5
            fifth_height = height/5
            fifth_width = width/5
            lowres = scipy.misc.imresize(crop, (fifth_height, fifth_width, 3))
            crop = scipy.misc.imresize(crop, (height/2, width/2, 3))
            if not include_rgb:
                # Convert to grayscale using weighted average
                crop = rgbToGrayscale(crop)
                lowres = rgbToGrayscale(lowres)
            lowresT = tf.stack(lowres)

            center = focusAtCursor(sess, crop, 1/2.0, cursorY, cursorX, fifth_height, fifth_width, include_rgb, include_prompt)

            stacked = tf.stack([center, lowresT], axis=0)
            imgOut = tf.reshape(stacked, shape=[2*fifth_height, fifth_width, -1]).eval()
            return imgOut #/ 25.5
    # Return a zero ndarray with the appropriate dimensions
    z = 1
    if include_rgb:
        z = 3
    if zoom_to_cursor:
        y = 32
        x = 32
        if include_prompt:
            y = 42
    else:
        y = 80
        x = 80
        if include_prompt:
            y = 105
    return np.zeros(shape=[y, x, z])

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
    small_step = 10
    minY = 125
    maxY = 285
    minX = 10
    maxX = 170
    if a == 0:
        return [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, 1),
            universe.spaces.PointerEvent(x, y, 0)], x, y
    elif a == 1:
        if y - small_step >= minY:
            return [universe.spaces.PointerEvent(x, y - small_step, 0)], x, y - small_step
    elif a == 2:
        if y + small_step <= maxY:
            return [universe.spaces.PointerEvent(x, y + small_step, 0)], x, y + small_step
    elif a == 3:
        if x - small_step >= minX:
            return [universe.spaces.PointerEvent(x - small_step, y, 0)], x - small_step, y
    elif a == 4:
        if x + small_step <= maxX:
            return [universe.spaces.PointerEvent(x + small_step, y, 0)], x + small_step, y
    elif a == 5 and include_stay:
        return [universe.spaces.PointerEvent(x, y, 0)], x, y
    return [], x, y

def getEpisodeNumber(info, prev):
    if info['n'][0]['env_status.episode_id'] is not None:
        return int(info['n'][0]['env_status.episode_id'])
    else:
        return prev

def focusAtCursor(sess, imageIn, scale_mult, cursorY, cursorX, windowY, windowX, include_rgb, include_prompt):
    orig_cursorY = cursorY
    cursorX = int(round((cursorX - 10) * scale_mult))
    cursorY = int(round((cursorY - 75 - 50) * scale_mult))
    #print 'cursorY, cursorX after scaling', cursorY, cursorX
    #print 'windowY, windowX', windowY, windowX
    height = int(round(160 * scale_mult))
    width = int(round(160 * scale_mult))
    if include_prompt:
        height = int(round(210 * scale_mult))
        cursorY = int(round((orig_cursorY - 75) * scale_mult))
    z = 1
    if include_rgb:
        z = 3
    imageIn = tf.reshape(imageIn, shape=[-1, height, width, z])
    padded = tf.pad(tf.reshape(imageIn[0,:,:,:], shape=[height, width, z]),
                    [[windowY/2, windowY/2],[windowX/2, windowX/2],[0, 0]],
                    "CONSTANT")

    result = padded[cursorY:cursorY+windowY, cursorX:cursorX+windowX, :]

    debug = False
    if debug:
        plotVision(padded.eval(), include_rgb)
        plt.gca().add_patch(Rectangle((cursorX,cursorY), windowX, windowY, fill=False))
    return result

def softMax(xs):
    e_xs = np.exp(xs - np.max(xs))
    return e_xs / e_xs.sum()

def chooseActionFromSingleQOut(singleQOut, use_probs, print_softmax):
    unique = np.unique(singleQOut)
    if len(unique) == 1:
        return np.random(0, len(singleQOut))
    elif use_probs:
        if print_softmax:
            print 'softmax', softMax(singleQOut), '\n'
        return np.random.choice(range(len(singleQOut)), p=softMax(singleQOut))
    else:
        return np.argmax(singleQOut, 0)

def chooseActionFromQOut(QOut, use_probs, print_softmax):
    if QOut.ndim == 1:
        return chooseActionFromSingleQOut(QOut, use_probs)
    else:
        return [chooseActionFromSingleQOut(q, use_probs) for q in QOut]

def isValidObservation(s):
    return s is not None and len(s) > 0 and s[0] is not None and type(s[0]) == dict and 'vision' in s[0]

def getValidObservation(sess, env, zoom_to_cursor, include_rgb, include_prompt, prevX, prevY):
    s = None
    while not isValidObservation(s):
        s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    s = processState(sess, s, zoom_to_cursor, include_rgb, include_prompt, prevY, prevX)
    return s

def makeEnvironment(env_name):
    env = gym.make(env_name)
    # automatically creates a local docker container
    env.configure(remotes=1, fps=15,
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
    return s, info, prevX, prevY, ep_num_offset

def epNumIsConstant(info, ep_num, ep_num_offset):
    return getEpisodeNumber(info, ep_num + ep_num_offset) == ep_num + ep_num_offset

def addReward(r, ep_num, rewards, successes, fails, misses):
    if r == 0:
        misses += 1
        print 'Miss'
    elif r > 0:
        successes += 1
        print 'Success'
    else:
        fails += 1
        print 'Fail'
    rewards.append([ep_num, r])
    return rewards, successes, fails, misses

def discountEpsilon(epsilon, step_drop, end_epsilon):
    if epsilon > end_epsilon:
        epsilon -= step_drop
    return epsilon

def trainQNs(sess, mainQN, targetQN, stochastic_policy, trainBatch, batch_size, y):
    QOut1 = sess.run(mainQN.QOut,feed_dict={mainQN.imageIn:np.stack(trainBatch[:,3])})
    Q1 = chooseActionFromQOut(QOut1, stochastic_policy, print_softmax=False)
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
    print 'Rewards', rewards[-100:]

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

def plotVision(s, include_rgb):
    plt.close()
    if include_rgb:
        plt.imshow(s)
    else:
        # Drop last dimension (of length 1)
        s = s.reshape(s.shape[:-1])
        plt.imshow(s, cmap='gray')
    plt.show(block=False)


env_name = 'wob.mini.ClickTest-v0'
batch_size = 32 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
start_epsilon = 1 # Starting chance of random action
end_epsilon = 0.1 # Final chance of random action
anneling_steps = 300000. # How many steps of training to reduce startE to endE.
num_episodes = 7000 # How many episodes of game environment to train network with.
pre_train_steps = 5000 # How many steps of random actions before training begins.
pre_anneling_steps = 50000 # How many steps of training before decaying epsilon
h_size = 512 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network
num_actions = 6
pos_reward_mult = 100

load_model = False # Whether to load a saved model.
checkpoint_path, evaluation_path, tboard_path = getOutputDirNames()
plot_vision = False # Plot the agent's view of the environment
include_rgb = False # If true, use an RGB view as input. If false, convert to grayscale.
include_prompt = False # If true, include yellow prompt in input.
stochastic_policy = True # If true, Q-function defines a probability distribution
include_stay = False # Include STAY as an action
include_horizontal_moves = True # Include LEFT and RIGHT as actions
if not include_stay:
    num_actions -= 1
if not include_horizontal_moves:
    num_actions -= 2
zoom_to_cursor = False # Process the input to includ a zoomed-in view around the cursor
if not zoom_to_cursor:
    h_size = 1024
save_history = True # If true, write results to file. If false, render environment.
tboard_summaries = True # If true, write summaries to file that can be shown in TensorBoard
summary_print_freq = 10 # How often (in episodes) to print a summary
summary_freq = 20 # How often (in episodes) to write a summary to a summary file
checkpoint_freq = 100 # How often (in episodes) to save a checkpoint of model parameters


env = makeEnvironment(env_name)

tf.reset_default_graph()
mainQN = QLearner(h_size, num_actions, zoom_to_cursor, include_rgb, include_prompt)
targetQN = QLearner(h_size, num_actions, zoom_to_cursor, include_rgb, include_prompt)

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
num_training_steps = 0

successes = 0
fails = 0
misses = 0

#max_q = 0

with tf.Session() as sess:
    if load_model:
        loadModel(sess, saver, checkpoint_path)
    sess.run(init)
    total_t = sess.run(tf.contrib.framework.get_global_step())
    updateTarget(sess, targetOps) #Set the target network to be equal to the primary network.
    ep_num = 0
    # Center cursor and wait until state observation s is valid
    s, info, currentX, currentY, ep_num_offset = initEnvironment(env, save_history)

    while ep_num < num_episodes:
        episodeBuffer = ExperienceBuffer()
        s = getValidObservation(sess, env, zoom_to_cursor, include_rgb, include_prompt, currentX, currentY)
        prevX, prevY = currentX, currentY
        s1 = s
        d = False
        rAll = 0
        step_num = 0
        ep_num = getEpisodeNumber(info, ep_num + ep_num_offset) - ep_num_offset
        print '\n------ Episode', ep_num
        print (currentX, currentY)
        #The Q-Network
        while epNumIsConstant(info, ep_num, ep_num_offset):
            step_num += 1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
                a_num = np.random.randint(0, num_actions)
            else:
                QOut = sess.run(mainQN.QOut,feed_dict={mainQN.imageIn:[s]})[0]
                Value = sess.run(mainQN.Value,feed_dict={mainQN.imageIn:[s]})[0]
                Advantage = sess.run(mainQN.Advantage, feed_dict={mainQN.imageIn:[s]})[0]
                print QOut, 'V', Value, 'A', Advantage
                a_num = chooseActionFromQOut(QOut, stochastic_policy, print_softmax=True)
                print step_num, 'Decided',
                action_numbers = {0: 'CLICK', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'STAY'}
                print action_numbers[a_num]
            a, currentX, currentY = intToVNCAction(a_num, include_stay, prevX, prevY)
            s1, r, d, info = env.step([a])
            if type(s1) == tf.Tensor:
                s1 = s1.eval()
            while not isValidObservation(s1):
                s1, r, d, info = env.step([a])
            s1 = processState(sess, s1, zoom_to_cursor, include_rgb, include_prompt, prevY, prevX)
            if r[0] > 0:
                r_scaled = r[0]#*pos_reward_mult
            else:
                r_scaled = r[0]#-1#
            episodeBuffer.add(np.reshape(np.array([s,a_num,r_scaled,s1,d[0]]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                if total_steps > pre_train_steps + pre_anneling_steps:
                        epsilon = discountEpsilon(epsilon, step_drop, end_epsilon)
                if (total_steps + step_num) % (update_freq) == 0:
                    num_training_steps += 1
                    #print 'Training step'
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    trainQNs(sess, mainQN, targetQN, stochastic_policy, trainBatch, batch_size, y)
                    updateTarget(sess, targetOps) #Set the target network to be equal to the primary network.
            rAll += r[0]
            
            if d[0] == True:
                total_steps += step_num
                print 'Steps taken this episode:', step_num
                if r[0] == 0 or abs(r[0]) > 1:
                    ep_num_offset += 1
                else:
                    rewards, successes, fails, misses = addReward(r[0], ep_num, rewards, successes, fails, misses)
                    if save_history:
                        history_writer.saveEpisode(ep_num, r[0])
                    if tboard_summaries:
                        episode_summary = tf.Summary()
                        episode_summary.value.add(simple_value=r[0], tag="Reward")
                        episode_summary.value.add(simple_value=epsilon, tag="Epsilon")
                        episode_summary.value.add(simple_value=total_steps, tag="Total steps")
                        episode_summary.value.add(simple_value=num_training_steps, tag="Total training steps")
                        episode_summary.value.add(simple_value=step_num, tag="Actions per episode")
                        episode_summary.value.add(simple_value=float(successes)/len(rewards), tag="Success-%")
                        episode_summary.value.add(simple_value=float(fails)/len(rewards), tag="Fail-%")
                        episode_summary.value.add(simple_value=float(misses)/len(rewards), tag="Miss-%")
                        summary_writer.add_summary(episode_summary, total_t)
                        summary_writer.flush()
                        total_t += 1
                        print 'Wrote to', tboard_path
            s = s1
            prevX, prevY = currentX, currentY
            if not save_history:
                env.render()
            if plot_vision:
                plotVision(s1, include_rgb)
        
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
