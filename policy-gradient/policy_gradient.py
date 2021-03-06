import gym
import universe
import math as math
import numpy as np
import tensorflow as tf
import scipy.misc
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import sys, os
from skimage.feature import corner_harris, corner_subpix, corner_peaks, peak_local_max

env = gym.make('wob.mini.FocusText-v0')
env.configure(remotes=1, fps=15,
              vnc_driver='go', 
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 
                          'fine_quality_level': 100, 'subsample_level': 0})
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    set_reward = r[r.size-1]
    if abs(set_reward[0]) < 0.00001:
        set_reward[0]=-1    
    for t in reversed(xrange(0, r.size)):
        get_reward = r[t]
        running_add = running_add * gamma + get_reward[0]
        print "get rew", get_reward ,"and run add", running_add
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        print "Agent speaking:", a_size
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)
        self.debug = tf.shape(self.output)[0]
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph() #Clear the Tensorflow graph.
               
def getOutputGraph():
    tboard_path_suffix = "tboard"
    tboard_path = tboard_path_suffix
    # Add ID to output directory names to distinguish between runs
    if len(sys.argv) > 1:
        agent_id = str(sys.argv[1])
        tboard_path = tboard_path_suffix + "-" + agent_id
    return tboard_path    


def validObserv(s):
    return s is not None and len(s) > 0 and s[0] is not None and type(s[0]) == dict and 'vision' in s[0]

def doAction(a, x, y):
    small_step = 18
    minY = 125
    maxY = 285
    minX = 10
    maxX = 170
    if a == 0:
        return [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, 1),
            universe.spaces.PointerEvent(x, y, 0)], x, y
    elif a == 3:
        return [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, 1),
            universe.spaces.PointerEvent(x, y, 0)], x, y
    elif a == 2:
        return [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, 1),
            universe.spaces.PointerEvent(x, y, 0)], x, y            
    elif a == 4:
        if y + small_step <= maxY:
            return [universe.spaces.PointerEvent(x, y + small_step, 0)], x, y + small_step
    elif a == 1:
        if y - small_step >= minY:
            return [universe.spaces.PointerEvent(x, y - small_step, 0)], x, y - small_step
    return [], x, y

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray    

def manipulateState(s, coordX,coordY):
    if s is not None and len(s) > 0 and s[0] is not None:
        if type(s[0]) == dict and 'vision' in s[0]:
            vi = s[0]['vision']
            mid = vi[75:75+210, 10:10+160, :]
            square = mid[75+50:75+50+160, 10:10+160, :] 

            grey = rgb2gray(square)
            coords = corner_peaks(corner_harris(grey), num_peaks=20, min_distance=5)
            coords_subpix = corner_subpix(grey, coords)
            num_coords = coords.shape[0]
            coords_array = np.zeros((1344,2))
            coords_array[:num_coords, :]=coords
            return tf.reshape(coords_array, shape=[1, -1]).eval()
    return np.zeros(shape=[1,84*32])


myAgent = agent(lr=1e-2,s_size=84*32,a_size=5,h_size=1036) #Load the agent.
#s_sze is expecting one input coz its a flat vector (array), sp
total_episodes = 1000 #Set total number of episodes to train agent on.
update_frequency = 1
success_rate = 0
tboard_summaries = True
tboard_path = getOutputGraph()

global_step= tf.Variable(0, name='global_step', trainable=False)
init = tf.global_variables_initializer()

if tboard_summaries:
    summary_writer = tf.summary.FileWriter(tboard_path)
    if not os.path.exists(tboard_path):
        os.makedirs(tboard_path)    

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    total_t = sess.run(tf.contrib.framework.get_global_step())
    i = 0
    total_reward = []
    total_lenght = []
    print_cur_reward = []
    print_tot_reward = []
    cur_episode = 0
    
    #starting cursor in the middle of the frame
    coordX = np.random.randint(0, 80)+10
    coordY = np.random.randint(0, 80)+125
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    end_rewards =0
    while i < total_episodes:
        coordX = np.random.randint(0, 80)+10
        coordY = np.random.randint(0, 80)+125
        s1,r,d,_ = env.step([[universe.spaces.PointerEvent(coordX, coordY)]])
        s = env.reset()
        cur_episode +=1
        running_reward = 0
        d = False
        completed_click =0
        #TODO SET S into something until s is not
        ep_history = []
        
        #loop through each clicks
        while d != True:
            completed_click+=1
            s = manipulateState(s, coordX, coordY)
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:np.array(s)})
            bug1 = sess.run(myAgent.state_in,feed_dict={myAgent.state_in:np.array(s)})
            bug = sess.run(myAgent.output,feed_dict={myAgent.state_in:np.array(s)})

            print myAgent.chosen_action, "CHOSEN ACTio"
            a_dist = 0.75 * (a_dist + np.ones((1, 5)) / 5) #average uniform probability
            a_dist /= a_dist.sum()
            a = np.random.choice(np.arange(0,5),p=a_dist[0])
            
            actionset= doAction(a, coordX, coordY)
            
            s1,r,d,_ = env.step([actionset[0]]) #Get our reward for taking an action given a bandit.
            print "Am I done?", d, "current episode:", cur_episode
            print "Chosen action:", a, "and its reward:", r[0]
            ep_history.append([s,a,r,s1])
            s = s1

            if d[0] == True:
                print_cur_reward.append(r[0])
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                if(r[0]>0):
                    success_rate += 1
                if tboard_summaries:
                    episode_summary = tf.Summary()
                    episode_summary.value.add(simple_value=r[0], tag="Reward")
                    episode_summary.value.add(simple_value=success_rate, tag="Success Rate")
                    summary_writer.add_summary(episode_summary, total_t)
                    summary_writer.flush()
                    total_t+=1                        
                running_reward += r[0]
                total_reward.append(running_reward)
                d=True
                break
            env.render()
        end_rewards=running_reward/completed_click
        cumulative_r = float(end_rewards/total_episodes)
        print "current CUMULATIVE", cumulative_r
        d=False

        if i % 100 == 0:
            print np.mean(total_reward[-100:])
        i += 1
print "Print Average per 10 click:", print_tot_reward       
print "last, Reward Each click:", print_cur_reward        
cumulative_r = float(end_rewards/total_episodes)
print "CUMULATIVE", cumulative_r

