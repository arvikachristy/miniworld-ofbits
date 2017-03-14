import gym
import universe
import numpy as np
import tensorflow as tf
import scipy.misc
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from skimage.feature import corner_harris, corner_subpix, corner_peaks, peak_local_max

env = gym.make('wob.mini.ClickTest-v0')
env.configure(remotes=1, fps=2,
              vnc_driver='go', 
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 
                          'fine_quality_level': 100, 'subsample_level': 0})
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        # print "discount", r, "ddfg", r[t]
        get_reward = r[t]
        running_add = running_add * gamma + get_reward[0]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        print "Agent speaking:", lr, s_size,a_size,h_size
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
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
                               

def validObserv(s):
    return s is not None and len(s) > 0 and s[0] is not None and type(s[0]) == dict and 'vision' in s[0]

def doAction(a, x, y):
    small_step = 20
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

def manipulateState(s, coordX,coordY):
    if s is not None and s[0] is not None:
        if type(s[0]) == dict and 'vision' in s[0]:
            vi = s[0]['vision']
            crop = vi[75:75+210, 10:10+160, :]

            # divide by 5
            # lowres = scipy.misc.imresize(crop, (42, 32, 3))
            # lowresT = tf.pack(lowres)

            #making it from RGB to grayscale
            grey= np.mean(crop, axis=2)

            coords = corner_peaks(corner_harris(grey), num_peaks=20, min_distance=5)

            print coords, "SUBPIX SIZE"
            coords_subpix = corner_subpix(grey, coords)
            num_coords = coords.shape[0]
            coords_array = np.zeros((20,2))

            # center= np.mean(center, axis=1)
            
            # coords_subpix = np.reshape(coords_subpix, (1,22))

            # stacked = tf.stack([coords_subpix], axis=0)
            # print tf.reshape(coords_subpix, shape=[1, -1]).eval(), "HOLO"
            coords_array[:num_coords, :]=coords
            return tf.reshape(coords_array, shape=[1, -1]).eval()
    return np.zeros(shape=[1, 40])

myAgent = agent(lr=1e-2,s_size=40,a_size=6,h_size=8) #Load the agent.
#s_sze is expecting one input coz its a flat vector (array), sp
total_episodes = 10#Set total number of episodes to train agent on.
update_frequency = 1

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []
    print_cur_reward = []
    print_tot_reward = []
    cur_episode = 0
    coordX = 80+10 
    coordY = 80+75+50
    s = env.reset()
    #starting cursor in the middle of the frame
    
    s1,r,d,_ = env.step([[universe.spaces.PointerEvent(coordX, coordY)]])
    # while not validObserv(s):
    #     s1,r,d,_ = env.step([[universe.spaces.PointerEvent(coordX, coordY)]])
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    end_rewards =0
    while i < total_episodes:
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
            print "new state", s
            # while not validObserv(s):
            #     s1,r,d,_ = env.step([[universe.spaces.PointerEvent(coordX, coordY)]])            
            #Choose either a random action or one from our network.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:np.array(s)})
            #a_dist = [[0.15,0.15,0.15,0.15,0.15,0.25]]
            print "Agent speaking", myAgent.state_in
            print a_dist.sum(), "SUM IS HERE"
            a_dist = 0.5 * (a_dist + np.ones((1, 6)) / 6) #average uniform probability
            a_dist /= a_dist.sum()
            print "adist!" ,a_dist
            #pick from 0-6 choices and get the probability
            a = np.random.choice(np.arange(0,6),p=a_dist[0])
            
            #a = np.argmax(a_dist == a) #takes the highest value from the array, but why a_dist==a?
            actionset= doAction(a, coordX, coordY)
            print "GDJHSGD", actionset
            s1,r,d,_ = env.step([actionset[0]]) #Get our reward for taking an action given a bandit.
            #print "GDJHSGD", actionset
            print "DONE?", d, "I'm in episode:", cur_episode
            
            print "Chosen action:", a, "and HEREEE", r[0]
            ep_history.append([s,a,r,s1])
            s = s1
            
            running_reward += r[0]
            print "Total Cumulative Reward now:", running_reward, "R is here",r

            print "gfhgj", len(r)
            if d[0] == True:
                print_cur_reward.append(r[0])
                print "Reward each click:", print_cur_reward
                print "HErEEEEEEEEEE TRUEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
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

                total_reward.append(running_reward)
                # total_lenght.append(j)
                d=True
                break
            env.render()
        end_rewards=running_reward/completed_click
        d=False
        print (running_reward/float(10)), "heyo", completed_click
        print_tot_reward.append(running_reward/float(10))
            #Update our running tally of scores.
        if i % 100 == 0:
            print np.mean(total_reward[-100:])
        i += 1
print "Print Average per 10 click:", print_tot_reward       
print "last, Reward Each click:", print_cur_reward        
cumulative_r = float(end_rewards/total_episodes)
print "CUMULATIVE", cumulative_r

