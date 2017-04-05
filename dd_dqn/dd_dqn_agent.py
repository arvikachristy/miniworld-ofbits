import gym
import universe # register the universe environments
import numpy as np
import random
import math
import tensorflow as tf
import scipy.misc
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from types import ModuleType

from utils.Output import Output
from ExperienceBuffer import ExperienceBuffer

# DD-DQN implementation based on Arthur Juliani's tutorial "Simple Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond"

class QLearner():
  def __init__(self, h_size, num_actions, zoom_to_cursor, four_convs, include_rgb, include_prompt):
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

        if four_convs:
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
            self.outLayer = self.conv4
        else:
            # Network architecture from DeepMind:
            self.conv1 = tf.contrib.layers.convolution2d( \
                inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None)
            self.conv2 = tf.contrib.layers.convolution2d( \
                inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None)
            self.conv3 = tf.contrib.layers.convolution2d( \
                inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID',biases_initializer=None)

            shape = self.conv3.get_shape().as_list()
            self.conv3_flat = tf.reshape(self.conv3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            self.fc1 = tf.contrib.layers.fully_connected( \
                inputs=self.conv3_flat,num_outputs=512,activation_fn=tf.nn.relu)
            self.outLayer = tf.reshape(self.fc1, shape=[-1, 1, 1, 512])

    # We take the output from the final convolutional layer and split it into separate
    # advantage and value streams.
    self.streamAC, self.streamVC = tf.split(self.outLayer, num_or_size_splits=2, axis=3)
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
    #self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
    self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    self.updateModel = self.trainer.minimize(self.loss)
    #self.updateModel = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

def rgbToGrayscale(img):
    h, w, _ = img.shape
    gray = np.zeros(shape=[h, w, 1], dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            px = img[y][x]
            gray[y, x] = int(0.299*px[0] + 0.587*px[1] + 0.114*px[2])
    return gray

def processState(s, zoom_to_cursor, include_rgb, include_prompt, cursorY, cursorX):
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

            center = focusAtCursor(crop, 1/2.0, cursorY, cursorX, fifth_height, fifth_width, include_rgb, include_prompt)

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

def intToVNCAction(a, x, y, step_size):
    minY = 125
    maxY = 285
    minX = 10
    maxX = 170
    if a == 0:
        return [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, 1),
            universe.spaces.PointerEvent(x, y, 0)], x, y
    elif a == 1:
        if y - step_size >= minY:
            return [universe.spaces.PointerEvent(x, y - step_size, 0)], x, y - step_size
    elif a == 2:
        if y + step_size <= maxY:
            return [universe.spaces.PointerEvent(x, y + step_size, 0)], x, y + step_size
    elif a == 3:
        if x - step_size >= minX:
            return [universe.spaces.PointerEvent(x - step_size, y, 0)], x - step_size, y
    elif a == 4:
        if x + step_size <= maxX:
            return [universe.spaces.PointerEvent(x + step_size, y, 0)], x + step_size, y
    elif a == 5:
        return [universe.spaces.PointerEvent(x, y, 0)], x, y
    action_numbers = {0: 'CLICK', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'STAY'}
    #print 'Cannot take action (', action_numbers[a], ')'
    return [], x, y

def getEpisodeNumber(info, prev):
    if info['n'][0]['env_status.episode_id'] is not None:
        return int(info['n'][0]['env_status.episode_id'])
    else:
        return prev

def focusAtCursor(imageIn, scale_mult, cursorY, cursorX, windowY, windowX, include_rgb, include_prompt):
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

def linearDistribution(xs):
    if np.min(xs) < 0:
        xs += 1.01 * abs(np.min(xs))
    #print xs, '\t', xs / xs.sum()
    return xs / xs.sum()

def chooseActionFromSingleQOut(singleQOut, use_probs, use_softmax, print_probs):
    unique = np.unique(singleQOut)
    if len(unique) == 1:
        return np.random.randint(0, len(singleQOut))
    elif use_probs:
        if use_softmax:
            probs = softMax(singleQOut)
        else:
            probs = linearDistribution(singleQOut) 
        if print_probs:
            print 'probs', probs, '\n'
        return np.random.choice(range(len(singleQOut)), p=probs)
    else:
        return np.argmax(singleQOut, 0)

def chooseActionFromQOut(QOut, use_probs, use_softmax, print_probs):
    if QOut.ndim == 1:
        return chooseActionFromSingleQOut(QOut, use_probs, use_softmax, print_probs)
    else:
        return [chooseActionFromSingleQOut(q, use_probs, use_softmax, print_probs) for q in QOut]

def isValidObservation(s):
    return s is not None and len(s) > 0 and s[0] is not None and type(s[0]) == dict and 'vision' in s[0]

def getValidObservation(env, prevX, prevY):
    s = None
    while not isValidObservation(s):
        s, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
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

def getCenterOfButton(s, prevX, prevY, cursorH, cursorW):
    # TODO: check for RGB image
    coords = corner_peaks(corner_harris(s), min_distance=2, threshold_rel=0.01)
    coords_subpix = corner_subpix(s, coords, window_size=13)

    #print 'coords_subpix'
    for item in coords_subpix:
        #print '  ', item[0],item[1],
        if abs(item[0]+125 - prevY) < cursorH and abs(item[1]+10 - prevX) < cursorW:
            #print 'CURSOR',
            item[0] = np.nan
            item[1] = np.nan
        #print ''
    #print '--'

    newX = coords_subpix[:,1]
    newY = coords_subpix[:,0]
    newX = newX[np.logical_not(np.isnan(newX))]
    newY = newY[np.logical_not(np.isnan(newY))]

    goalX = np.mean(newX) + 10
    goalY = np.mean(newY) + 125
    return goalX, goalY, newX, newY

def generateSupervisedAction(s, prevX, prevY, cursorH, cursorW, step_size, include_prompt, num_actions, step_num):
    '''Only implemented for the ClickTest-v0 environment'''
    # TODO: other environments
    goalX, goalY, newX, newY = getCenterOfButton(s, prevX, prevY, cursorH, cursorW)

    y_offset = 125
    if include_prompt:
        y_offset -= 50
    plt.scatter(newX, newY, s=5, c='red', marker='o')
    plt.scatter(goalX - 10, goalY - y_offset, s=6, c='blue', marker='o')
    plt.gca().add_patch(Rectangle((prevX - 10, prevY - y_offset), cursorW, cursorH, fill=False))
    plt.show(block=False)

    if math.isnan(goalX) or math.isnan(goalY):
        print '---- Found no corners!'
        return np.random.randint(0, num_actions)
    xDiff = goalX - prevX
    yDiff = goalY - prevY
    #print 'diffs', goalX, '-', prevX, '=', xDiff, '\t', goalY, '-', prevY, '=', yDiff
    print step_num, 'Supervised',
    if abs(xDiff) < step_size and abs(yDiff) < step_size:
        print 'CLICK'
        return 0 # CLICK
    elif abs(xDiff) > abs(yDiff):
        if xDiff > 0:
            print 'RIGHT'
            return 4 # RIGHT
        else:
            print 'LEFT'
            return 3 # LEFT
    else:
        if yDiff > 0:
            print 'DOWN'
            return 2 # DOWN
        else:
            print 'UP'
            return 1 # UP

def sampleAction(sess, raw_s, s, prevX, prevY, mainQN, num_actions, epsilon, step_num, total_steps, pre_train_steps, \
                 supervised_episode, stochastic_policy, use_softmax, include_prompt, cursorH, cursorW, step_size=15):
    # TODO:
    # check if s needs to be unprocessed
    
    if supervised_episode:
        # Use full size image for corner detection
        full_size_s = raw_s[0]['vision']
        full_size_s = full_size_s[75+50:75+210, 10:10+160, :]
        full_size_s = rgbToGrayscale(full_size_s)
        plotVision(full_size_s, include_rgb=False)
        #print 'Supervised episode'
        a_num = generateSupervisedAction(full_size_s, prevX, prevY, cursorH, cursorW, step_size, include_prompt, num_actions, step_num)
        #print a_num
        if a_num != -1:
            return a_num

    QOut = sess.run(mainQN.QOut, feed_dict={mainQN.imageIn:[s]})[0]
    Value = sess.run(mainQN.Value, feed_dict={mainQN.imageIn:[s]})[0]
    Advantage = sess.run(mainQN.Advantage, feed_dict={mainQN.imageIn:[s]})[0]
    print QOut, 'V', Value, 'A', Advantage
    if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
        a_num = np.random.randint(0, num_actions)
    else:
        #QOut, Value, Advantage = sess.run(mainQN.QOut, mainQN.Value, mainQN.Advantage,feed_dict={mainQN.imageIn:[s]})[0]
        #print 'V, A before QOut calculation:', Value, Advantage
        #QOut = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, reduction_indices=1, keep_dims=True))
        a_num = chooseActionFromQOut(QOut, stochastic_policy, use_softmax, print_probs=True)
        print step_num, 'Decided',
        action_numbers = {0: 'CLICK', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'STAY'}
        print action_numbers[a_num]
    return a_num, QOut, Value, Advantage

def addReward(r, rewards, successes, fails, misses):
    if r == 0:
        misses += 1
        print 'Miss'
    elif r > 0:
        successes += 1
        print 'Success'
    else:
        fails += 1
        print 'Fail'
    rewards.append(r)
    return rewards, successes, fails, misses

def discountHyperParameter(param, step_drop, end_value):
    if param > end_value:
        param -= step_drop
    return param

def trainQNs(sess, mainQN, targetQN, learning_rate, stochastic_policy, use_softmax, trainBatch, batch_size, y):
    QOut1 = sess.run(mainQN.QOut,feed_dict={mainQN.imageIn:np.stack(trainBatch[:,3])})
    Q1 = chooseActionFromQOut(QOut1, stochastic_policy, use_softmax, print_probs=False)
    Q2 = sess.run(targetQN.QOut,feed_dict={targetQN.imageIn:np.stack(trainBatch[:,3])})
    end_multiplier = -(trainBatch[:,4] - 1)
    doubleQ = Q2[range(batch_size),Q1]
    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
    # Update the network with our target values.
    _ = sess.run(mainQN.updateModel, \
        feed_dict={mainQN.imageIn:np.stack(trainBatch[:,0]),
            mainQN.targetQ:targetQ,
            mainQN.actions:trainBatch[:,1]})
            #mainQN.learning_rate:learning_rate})

def printSummary(stepList, rList, e, learning_rate, rewards, successes, fails, misses):
    print 'Actions taken', np.sum(stepList)
    print 'Average reward (last 100):', np.mean(rList[-100:]), '(last 10)', np.mean(rList[-10:])
    print 'Epsilon:', e, '\tLearning rate', learning_rate
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

def translateQsToGrayscale(qs):
    if len(qs) == 0:
        return qs
    elif len(qs) == 1:
        return [0.5]
    visual_min = 0.0
    visual_max = 1.0
    visual_span = visual_max - visual_min
    q_span = max(qs) - min(qs)
    qs_scaled = [visual_min + ((float(q) - min(qs))/q_span) * visual_span for q in qs]
    return qs_scaled

def translateQsToRGB(qs):
    q_pos = filter(lambda q: q > 0, qs)#[q in q for qs if q > 0]
    q_neg = filter(lambda q: q < 0, qs)#[q in q for qs if q < 0]
    q_pos_scaled = translateQsToGrayscale(q_pos)
    q_neg_scaled = [q * -1 for q in translateQsToGrayscale(q_neg)]
    q_neg_scaled = [q + 1 for q in q_neg_scaled]
    qs_scaled = []
    i_pos = 0
    i_neg = 0
    # Reconstruct original order
    for q in qs:
        if q > 0:
            qs_scaled.append(q_pos_scaled[i_pos])
            i_pos += 1
        elif q < 0:
            qs_scaled.append(q_neg_scaled[i_neg])
            i_neg += 1
        else:
            qs_scaled.append(0)
    return qs_scaled
    #sq_scaled = [(1-i,i,0) for i in your_floats]

def plotVision(s, include_rgb, show_heatmap=False, prevX=0, prevY=0, step_size=0, Q=0, unicolor=False):
    plt.close()
    if include_rgb:
        plt.imshow(s)
    else:
        # Drop last dimension (of length 1)
        s = s.reshape(s.shape[:-1])
        plt.imshow(s, cmap='gray')
        if show_heatmap and len(Q) > 0:
            if unicolor:
                Q_visual = translateQsToGrayscale(Q)
            else:
                Q_visual = translateQsToRGB(Q)
            for action in range(0, 5):
                _, x, y = intToVNCAction(action, prevX, prevY, step_size)
                visual_x = (x - 10)/2
                visual_y = (y - 50 - 75)/2
                if not unicolor and Q[action] > 0:
                    colour = 'green'
                elif not unicolor and Q[action] == 0:
                    colour = 'yellow'
                else:
                    colour = 'red'
                plt.scatter(visual_x, visual_y, s=15, c=colour, alpha=max(min(Q_visual[action], 1), 0.03))
    plt.show(block=False)


def dd_dqn_main():
    env_name = 'wob.mini.ClickTest-v0'
    batch_size = 32 # How many experiences to use for each training step.
    update_freq = 4 # How often to perform a training step.
    y = .99 # Discount factor on the target Q-values
    start_learning_rate = 0.0001#0.001#0.1
    end_learning_rate = 0.0001
    learning_anneling_steps = 1000
    start_epsilon = 1 # Starting chance of random action
    end_epsilon = 0.1 # Final chance of random action
    anneling_steps = 900000. # How many steps of training to reduce startE to endE.
    num_episodes = 15000 # How many episodes of game environment to train network with.
    pre_train_steps = 500#0 # How many steps of random actions before training begins.
    pre_anneling_steps = 500#00 # How many steps of training before decaying epsilon
    num_supervised_episodes = 0 # How many episodes to train on supervised actions.
    experience_buffer_size = 15000 # How many past steps are stored in the buffer at any one time.
    h_size = 512 # The size of the final layer before splitting it into Advantage and Value streams.
    tau = 0.001 # Rate to update target network toward primary network
    num_actions = 6
    step_size = 15 # How many pixels to move in one action
    cursor_height = 15
    cursor_width = 7
    pos_reward_mult = 1

    load_model = False # Whether to load a saved model.
    checkpoint_path, evaluation_path, tboard_path = getOutputDirNames()
    plot_vision = True # Plot the agent's view of the environment
    show_heatmap = True
    include_rgb = False # If true, use an RGB view as input. If false, convert to grayscale.
    include_prompt = False # If true, include yellow prompt in input.
    stochastic_policy = True # If true, Q-function defines a probability distribution
    use_softmax = True # Whether to use softmax in deriving probability distribution form Q-values
    include_stay = False # Include STAY as an action
    include_horizontal_moves = True # Include LEFT and RIGHT as actions
    if not include_stay:
        num_actions -= 1
    if not include_horizontal_moves:
        num_actions -= 2
    zoom_to_cursor = False # Process the input to includ a zoomed-in view around the cursor
    four_convs = False # Use a 4-layer convolutional network
    if not zoom_to_cursor and four_convs:
        h_size = 1024
    save_history = True # If true, write results to file. If false, render environment.
    tboard_summaries = True # If true, write summaries to file that can be shown in TensorBoard
    summary_print_freq = 10 # How often (in episodes) to print a summary
    summary_freq = 20 # How often (in episodes) to write a summary to a summary file
    checkpoint_freq = 100 # How often (in episodes) to save a checkpoint of model parameters


    env = makeEnvironment(env_name)

    tf.reset_default_graph()
    mainQN = QLearner(h_size, num_actions, zoom_to_cursor, four_convs, include_rgb, include_prompt)
    targetQN = QLearner(h_size, num_actions, zoom_to_cursor, four_convs, include_rgb, include_prompt)

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

    myBuffer = ExperienceBuffer(experience_buffer_size)
    rewards = []

    #Set the rate of random action decrease. 
    epsilon = start_epsilon
    epsilon_step_drop = (start_epsilon - end_epsilon)/anneling_steps

    learning_rate = start_learning_rate
    learning_step_drop = (start_learning_rate - end_learning_rate)/learning_anneling_steps

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
            episodeBuffer = ExperienceBuffer(experience_buffer_size)
            raw_s = getValidObservation(env, currentX, currentY)
            prevX, prevY = currentX, currentY
            s = processState(raw_s, zoom_to_cursor, include_rgb, include_prompt, prevY, prevX)
            s1 = s
            d = False
            rAll = 0
            step_num = 0
            current_ep_num = ep_num
            #ep_num = getEpisodeNumber(info, ep_num + ep_num_offset) - ep_num_offset
            print '\n------ Episode', ep_num
            print (currentX, currentY)

            while ep_num == current_ep_num:#epNumIsConstant(info, ep_num, ep_num_offset):
                step_num += 1
                # Sample an action given state s using mainQN and an epsilon-greedy policy
                supervised_episode = ep_num < num_supervised_episodes
                a_num, Q, V, A = sampleAction(sess, raw_s, s, prevX, prevY, mainQN, num_actions, epsilon, step_num, total_steps, pre_train_steps, \
                    supervised_episode, stochastic_policy, use_softmax, include_prompt, cursor_height, cursor_width, step_size)
                prevX, prevY = currentX, currentY

                a, currentX, currentY = intToVNCAction(a_num, prevX, prevY, step_size)
                raw_s1, r, d, info = env.step([a])
                #if (not s1 is None) and len(s1) > 0 and len(s1[0]) > 1:
                #    print s1[0]['text']
                if type(raw_s1) == tf.Tensor:
                    raw_s1 = raw_s1.eval()
                while not isValidObservation(raw_s1):
                    raw_s1, r, d, info = env.step([a])
                s1 = processState(raw_s1, zoom_to_cursor, include_rgb, include_prompt, prevY, prevX)
                if r[0] > 0:
                    r_scaled = r[0]*pos_reward_mult
                else:
                    r_scaled = r[0]#-1#
                episodeBuffer.add(np.reshape(np.array([s,a_num,r_scaled,s1,d[0]]),[1,5])) #Save the experience to our episode buffer.

                if total_steps > pre_train_steps:
                    if total_steps > pre_train_steps + pre_anneling_steps:
                            epsilon = discountHyperParameter(epsilon, epsilon_step_drop, end_epsilon)
                    if (total_steps + step_num) % (update_freq) == 0:
                        num_training_steps += 1
                        #print 'Training step'
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        trainQNs(sess, mainQN, targetQN, learning_rate, stochastic_policy, use_softmax, trainBatch, batch_size, y)
                        updateTarget(sess, targetOps) #Set the target network to be equal to the primary network.
                        learning_rate = discountHyperParameter(learning_rate, learning_step_drop, end_learning_rate)
                rAll += r[0]
                
                if d[0] == True:
                    print '.....', ep_num, r
                    total_steps += step_num
                    print 'Steps taken this episode:', step_num
                    if r[0] != 0 and abs(r[0]) <= 1:
                        ep_num += 1
                        rewards, successes, fails, misses = addReward(r[0], rewards, successes, fails, misses)
                        if save_history:
                            history_writer.saveEpisode(r[0])
                        if tboard_summaries:
                            episode_summary = tf.Summary()
                            episode_summary.value.add(simple_value=r[0], tag="Reward")
                            episode_summary.value.add(simple_value=epsilon, tag="Epsilon")
                            episode_summary.value.add(simple_value=learning_rate, tag="Learning Rate")
                            episode_summary.value.add(simple_value=total_steps, tag="Total steps")
                            episode_summary.value.add(simple_value=num_training_steps, tag="Total training steps")
                            episode_summary.value.add(simple_value=step_num, tag="Actions per episode")
                            episode_summary.value.add(simple_value=float(successes)/len(rewards), tag="Success-%")
                            episode_summary.value.add(simple_value=float(fails)/len(rewards), tag="Fail-%")
                            episode_summary.value.add(simple_value=float(misses)/len(rewards), tag="Miss-%")
                            if len(Q) > 0:
                                for i in range(len(Q)):
                                    episode_summary.value.add(simple_value=Q[i], tag="Q-"+str(i))
                                for i in range(len(V)):
                                    episode_summary.value.add(simple_value=V[i], tag="V-"+str(i))
                                for i in range(len(A)):
                                    episode_summary.value.add(simple_value=A[i], tag="A-"+str(i))
                            summary_writer.add_summary(episode_summary, total_t)
                            summary_writer.flush()
                            total_t += 1
                            print 'Wrote to', tboard_path
                s = s1
                raw_s = raw_s1
                if not save_history:
                    env.render()
                if plot_vision:
                    plotVision(s1, include_rgb, show_heatmap, prevX, prevY, step_size, Q)
            
            #Get all experiences from this episode and discount their rewards.
            myBuffer.add(episodeBuffer.buffer)
            stepList.append(step_num)
            rList.append(rAll)

            #Periodically save the model.
            if save_history and ep_num % checkpoint_freq == 0:
                saver.save(sess, checkpoint_path+'/model', global_step=ep_num)#+'.cptk')
                print "Saved Model"
            if ep_num % summary_print_freq == 0:
                printSummary(stepList, rList, epsilon, learning_rate, rewards, successes, fails, misses)
        if save_history:
            saver.save(sess, checkpoint_path+'/model-'+str(ep_num)+'.cptk')
    print "Average reward: ", np.mean(rList)

if __name__ == "__main__":
    dd_dqn_main()
