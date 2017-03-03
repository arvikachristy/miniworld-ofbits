import gym
import universe # register the universe environments
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os

from ExperienceBuffer import ExperienceBuffer

# DD-DQN implementation based on Arthur Juliani's tutorial "Simple Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond"

class QLearner():
  def __init__(self, h_size):
    self.num_actions = 6
    # 8064 = 42 * 64 * 3
    #self.scalarInput = tf.placeholder(shape=[-1, 8064], dtype = tf.float32)
    self.imageIn = tf.placeholder(shape=[None, 84, 32, 3], dtype=tf.float32)#tf.reshape(self.scalarInput, shape=[-1, 84, 32, 3])

    weights_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    self.conv1 = tf.contrib.layers.convolution2d( \
        inputs=self.imageIn,num_outputs=32,kernel_size=[5,5],stride=[3,2],padding='VALID', biases_initializer=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    self.conv2 = tf.contrib.layers.convolution2d( \
        inputs=self.conv1,num_outputs=64,kernel_size=[5,4],stride=[2,2],padding='VALID', biases_initializer=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    self.conv3 = tf.contrib.layers.convolution2d( \
        inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[2,1],padding='VALID', biases_initializer=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
    self.conv4 = tf.contrib.layers.convolution2d( \
        inputs=self.conv3,num_outputs=512,kernel_size=[5,4],stride=[1,1],padding='VALID', biases_initializer=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))

    # We take the output from the final convolutional layer and split it into separate
    # advantage and value streams.
    self.streamAC, self.streamVC = tf.split(3, 2, self.conv4)
    self.streamA = tf.contrib.layers.flatten(self.streamAC)
    self.streamV = tf.contrib.layers.flatten(self.streamVC)
    self.AW = tf.Variable(tf.random_normal([h_size/2, self.num_actions]), name='AW')
    self.VW = tf.Variable(tf.random_normal([h_size/2, 1]), name='VW')
    self.Advantage = tf.matmul(self.streamA, self.AW)
    self.Value = tf.matmul(self.streamV, self.VW)

    # Then combine them together to get our final Q-values
    self.QOut = self.Value + tf.sub(self.Advantage,
      tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
    self.predict = tf.argmax(self.QOut, 1)

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)

    self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
    self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
    
    self.Q = tf.reduce_sum(tf.mul(self.QOut, self.actions_onehot), reduction_indices=1)
    
    self.td_error = tf.square(self.targetQ - self.Q)
    self.loss = tf.reduce_mean(self.td_error)
    self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    self.updateModel = self.trainer.minimize(self.loss)

def cropState(s):
    if s is not None and len(s) > 0 and s[0] is not None:
        if type(s[0]) == dict and 'vision' in s[0]:
            v = s[0]['vision']
            print('Printing shape:', v.shape)
            crop = v[75:75+210, 10:10+160, :]
            return np.reshape(crop, [210*160*3])
    return np.zeros(210*160*3)

def processState(s, cursorY, cursorX):
    if s is not None and len(s) > 0 and s[0] is not None:
        if type(s[0]) == dict and 'vision' in s[0]:
            v = s[0]['vision']
            crop = v[75:75+210, 10:10+160, :]

            #divide by 5
            lowres = scipy.misc.imresize(crop, (42, 32, 3))
            lowresT = tf.pack(lowres)

            windowX = 32
            windowY = 42
            center = focusAtCursor(crop, cursorY, cursorX, windowY, windowX)

            stacked = tf.stack([center, lowresT], axis=0)
            return tf.reshape(stacked, shape=[84, 32, 3])
    return np.zeros(42*64*3)

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

batch_size = 12 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 1 #Final chance of random action
anneling_steps = 50000. #How many steps of training to reduce startE to endE.
num_episodes = 7000 #How many episodes of game environment to train network with.
pre_train_steps = 100#2000#0 #How many steps of random actions before training begins.
#max_epLength = 5000 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn-model" #The path to save our model to.
h_size = 512#64#1024 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
plot_vision = False

env = gym.make('wob.mini.ClickTest-v0')
# automatically creates a local docker container
env.configure(remotes=1, fps=5,
              vnc_driver='go', 
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 
                          'fine_quality_level': 100, 'subsample_level': 0})

tf.reset_default_graph()

init = tf.global_variables_initializer()

#saver = tf.train.Saver()

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

#Make a path for our model to be saved in.
#if not os.path.exists(path):
#    os.makedirs(path)

with tf.Session() as sess:
    mainQN = QLearner(h_size)
    targetQN = QLearner(h_size)
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    sess.run(init)
    sess.run(tf.variables_initializer([mainQN.VW, targetQN.VW]))
    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
    i = 0
    s_raw = env.reset()
    prevY = 80+75+50
    prevX = 80+10
    s_raw, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    while s_raw is None: #or (len(s) > 0 and s[0] is None):
        s_raw, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
    env.render()
    episodeOffset = episodeNumber(info, i)
    print 'Offset:', episodeOffset
    while i < num_episodes:
        episodeBuffer = ExperienceBuffer()

        if type(s_raw) == tf.Tensor:
            s_raw = s_raw.eval()
        while s_raw is None or (len(s_raw) > 0 and s_raw[0] is None):
            s_raw, r, d, info = env.step([[universe.spaces.PointerEvent(prevX, prevY, 0)]])
        s = processState(s_raw, prevY, prevX)
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
                a_num = sess.run(mainQN.predict,feed_dict={mainQN.imageIn:[s], mainQn.cursorX:prevX, mainQn.cursorY:prevY})[0]
                print j, 'Decided',
                action_numbers = {0: 'CLICK', 1: 'STAY', 2: 'UP', 3: 'RIGHT', 4: 'DOWN', 5: 'LEFT'}
                print action_numbers[a_num]
            a, prevX, prevY = intToVNCAction(a_num, prevX, prevY)
            s1_raw, r, d, info = env.step([a])
            if type(s1_raw) == tf.Tensor:
                s1_raw = s1_raw.eval()
            while s1_raw is None or (len(s1_raw) > 0 and s1_raw[0] is None):
                s1_raw, r, d, info = env.step([a])
            env.render()
            s1 = processState(s1_raw, prevY, prevX)
            if type(s1) == tf.Tensor:
                s1 = s1.eval()
            if plot_vision:
                plt.close()
                plt.imshow(s1)
                plt.show(block=False)
            total_steps += 1
            if total_steps % 10000 == 0:
                print 'Total steps', total_steps
            episodeBuffer.add(np.reshape(np.array([s,a_num,r[0],s1,d[0]]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.imageIn:np.stack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.QOut,feed_dict={targetQN.imageIn:np.stack(trainBatch[:,3])})
                    #end_multiplier = -(trainBatch[:,4] - 1)
                    end_multiplier = -(trainBatch[:,2] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.imageIn:np.stack(trainBatch[:,0]),
                            mainQN.targetQ:targetQ,
                            mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
                    
                    #q_out = sess.run(mainQN.QOut, feed_dict={mainQN.imageIn:[s]})#[0]
                    #print q_out

            rAll += r[0]
            
            if d[0] == True:
                rewards.append([i, r[0]])
                if r[0] == 0:
                    misses += 1
                elif r[0] > 0:
                    successes += 1
                    print 'Success'
                else:
                    fails += 1
            s = s1
        
        #Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        #TODO: Periodically save the model.
        #if i % 500 == 0:
        #    saver.save(sess,path+'/model', global_step=i)#+'.cptk')
        #    print "Saved Model"
        if len(rList) % 10 == 0:
            print total_steps,np.mean(rList[-100:]),np.mean(rList[-10:]), e
            print successes, ':', float(successes)/len(rList), '\t', \
            fails, ':', float(fails)/len(rList), '\t', \
            misses, ':', float(misses)/len(rList), '\t' \
            'avg steps/episode:', float(total_steps)/len(rList)
            #print 'Actions', mainQN.actions
            #print 'Actions 1hot', mainQN.actions_onehot
            #print 'Q', mainQN.Q
            print 'Rewards', rewards
    #saver.save(sess,path+'/model-'+str(i)+'.cptk')
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"