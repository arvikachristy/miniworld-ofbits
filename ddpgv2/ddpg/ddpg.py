""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import gym 
from gym import wrappers
import universe
import tensorflow as tf
import numpy as np
import tflearn

from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'wob.mini.ClickTest-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self): 
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.multiply(out, self.action_bound) # Scale output to -action_bound to action_bound
        return inputs, out, scaled_out 

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch 
        # w.r.t. that action (i.e., sum of dy/dx over all ys). We then divide
        # through by the minibatch size to scale the gradients down correctly.
        self.action_grads = tf.div(tf.gradients(self.out, self.action), tf.constant(MINIBATCH_SIZE, dtype=tf.float32))

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# =========
# Actions
# =========
max_dist = 10 #maximum distance per action

#bounds for verification
min_x = 10
max_x = min_x+160
min_y = 125
max_y = min_y+160

def scale_angle(magnitude): #choose a random angle between 0 and 359 degrees based on probabilities, returns in RADIANS
    angle = magnitude*360
    angle = angle * (np.pi/180)
    return angle

def scale_distance(magnitude): #choose a random distance between 0 and max distance based on probabilities
    distance = magnitude*max_dist
    return distance

def move(xcoord, ycoord, distance, angle): #move (if within bounds) and return updated coords 
    xdist = distance * np.sin(angle)
    ydist = distance * np.cos(angle)
    print "Distance: ", distance
    print "Angle (in radians): ", angle
    print "Angle (in degrees): ", angle*(180/np.pi)
    print "Current X: ", xcoord
    print "Current Y: ", ycoord
    print "Xdist: ", xdist
    print "Ydist: ", ydist
    #verification
    if(xdist>0):
        if(ydist>0):
            #both +ve
            if(xcoord+xdist<max_x and ycoord+ydist<max_y):
                xcoord += xdist
                ycoord += ydist
        else:
            #xdist +ve, ydist -ve
            if(xcoord+xdist<max_x and ycoord+ydist>min_y):
                xcoord += xdist
                ycoord += ydist
    else:
        if(ydist>0):
            #xdist -ve, ydist +ve
            if(xcoord+xdist>min_x and ycoord+ydist<max_y):
                xcoord += xdist
                ycoord += ydist
        else:
            #both -ve
            if(xcoord+xdist>min_x and ycoord+ydist>min_y):
                xcoord += xdist
                ycoord += ydist
    return xcoord,ycoord

def click(xcoord,ycoord):
    #click at x,y
    action = [[universe.spaces.PointerEvent(xcoord, ycoord, 0),
            universe.spaces.PointerEvent(xcoord, ycoord, 1),
            universe.spaces.PointerEvent(xcoord, ycoord, 0)]]
    return action

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    for i in xrange(MAX_EPISODES):
        observation = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0

        #initialize randomly for each episode
        xcoord = np.random.randint(0, 160) + 10  
        ycoord = np.random.randint(0, 160) + 75 + 50 
        prev_am = [0]#previous action magnitude - angle only at the moment
        for j in xrange(MAX_EP_STEPS):
            env.render()
            #print "Overall observation: ", observation
            for ob in observation:
                if ob is not None:
                    x = ob['vision']
                    crop = x[75:75+210, 10:10+160, :]
                else:
                    crop = [0,0]
                print "Previous observation: ", crop

                # Added exploration noise
                noise = (1. / (1. + i + j))
                current_am = actor.predict(np.reshape(prev_am, (1,1))) + noise
                print "current_am", current_am
                prev_am=current_am

                #current_dm = current_am[0]
                current_anm = current_am[0]
                xcoord,ycoord=move(xcoord,ycoord,max_dist,scale_angle(current_anm)) #get new x and ycoord, currently only moves fixed distance
                a=click(xcoord,ycoord)
                
                new_observation, r, terminal, info = env.step(a)
                print "chosen action: ", a
                env.render()

                replay_buffer.add(np.reshape(observation, (actor.s_dim,)), np.reshape(current_am, (actor.a_dim,)), r, \
                    terminal, np.reshape(new_observation, (actor.s_dim,)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                # if replay_buffer.size() > MINIBATCH_SIZE: 
                if 0:    
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # Calculate targets
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                    y_i = []
                    for k in xrange(MINIBATCH_SIZE):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + GAMMA * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
                
                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)                
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

                observation = new_observation
                ep_reward += r[0]

                if terminal:

                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j+1)
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                    print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                        '| Qmax: %.4f' % (ep_ave_max_q / float(j+1))

                    break

with tf.Session() as sess:
    
    env = gym.make(ENV_NAME)
    env.configure(remotes=1, fps=5,vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 90, 'fine_quality_level': 100, 'subsample_level': 0})
    # np.random.seed(RANDOM_SEED)
    # tf.set_random_seed(RANDOM_SEED)
    # env.seed(RANDOM_SEED)

    state_dim = 1 #env.observation_space.shape[0]
    print "state_dim: ", state_dim
    action_dim = 1 #env.action_space.shape[0]
    print "action_dim: ", action_dim
    action_bound = np.array([1.])
    print "action_bound type: ", type(action_bound)
    #print env.action_space.low
    # Ensure action bound is symmetric
    #assert (env.action_space.high == -env.action_space.low)

    actor = ActorNetwork(sess, state_dim, action_dim, action_bound, \
        ACTOR_LEARNING_RATE, TAU)

    critic = CriticNetwork(sess, state_dim, action_dim, \
        CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

    # if GYM_MONITOR_EN:
    #     if not RENDER_ENV:
    #         env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=True)
    #     else:
    #         env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)

    train(sess, env, actor, critic)

    #if GYM_MONITOR_EN:
        #env.monitor.close() #deprecated
