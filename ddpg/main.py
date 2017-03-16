#Implementation of Deep Deterministic Gradient with Tensor Flow"
#TODO: Adapt probabilities to update according to agent's rewards via actor-critic

import gym
from gym.spaces import Box, Discrete
import universe
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
#specify parameters here:
episodes=10000
is_batch_norm = False #batch normalization switch
max_dist = 100 #maximum distance per action

#bounds for verification
min_x = 10
max_x = min_x+160
min_y = 125
max_y = min_y+160

#initialize angles probabilities array
angles = []
for i in range(360):
    angles.append(np.random.randint(1,10)) 

norm_a = [float(i)/sum(angles) for i in angles] #normalize the angle array

#initialize distance probabilities array
distances = []
for i in range(max_dist):
    distances.append(np.random.randint(1,10))

norm_d = [float(i)/sum(distances) for i in distances] #normalize the distance array
#print(norm)

def choose_angle(): #choose a random angle between 0 and 359 degrees based on probabilities, returns in RADIANS
    angle = np.random.choice(np.arange(0,360),p=norm_a)
    angle = angle * (np.pi/180)
    return angle

def choose_distance(): #choose a random distance between 0 and max distance based on probabilities
    distance = np.random.choice(np.arange(0, max_dist),p=norm_d)
    return distance

def move(xcoord, ycoord, distance, angle): #move coords (if within bounds) and click
    xdist = distance * np.sin(angle)
    ydist = distance * np.cos(angle)
    print "Distance: ", distance
    print "Angle (in radians): ", angle
    print "Angle (in degrees): ", angle*(180/np.pi)
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


    #click after moving, we can change this later to randomly change PointerEvents (with more probabilities, thus modelling clicking continuously)
    action = click(xcoord,ycoord)
    return action

def click(xcoord,ycoord):
    #click at x,y
    action = [[universe.spaces.PointerEvent(xcoord, ycoord, 0),
            universe.spaces.PointerEvent(xcoord, ycoord, 1),
            universe.spaces.PointerEvent(xcoord, ycoord, 0)]]
    return action

def main():
    experiment= 'wob.mini.ClickTest-v0' #specify environments here
    env= gym.make(experiment)
    print "Observation space for ", experiment, ": ", env.observation_space
    print "Action space for ", experiment, ": ", env.action_space
    steps= env.spec.timestep_limit #steps per episode   
    env.configure(remotes=1, fps=5,vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 90, 'fine_quality_level': 100, 'subsample_level': 0})
    #assert isinstance(env.observation_space, Box), "observation space must be continuous"
    #assert isinstance(env.action_space, Box), "action space must be continuous"
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
    num_states = 1 #env.observation_space.shape[0]
    num_actions = 3 #env.action_space.shape[0] 
    exploration_noise = OUNoise(num_actions)
    counter=0
    reward_per_episode = 0    
    total_reward=0
    print "Number of States:", num_states
    print "Number of Actions:", num_actions
    print "Number of Steps per episode:", steps
    #saving reward:
    reward_st = np.array([0])
    
    for i in xrange(episodes):
        print "==== Starting episode no:",i,"====","\n"
        reward_per_episode = 0
        observation = env.reset()
        print "OBSERVATION: ", observation
        #initialize xcoord and ycoord randomly for each episode
        xcoord = np.random.randint(0, 160) + 10  
        ycoord = np.random.randint(0, 160) + 75 + 50 
        for t in xrange(steps):
            #rendering environment            
            env.render()
            for ob in observation:
                if ob is not None:
                    x = ob['vision']
                    crop = x[75:75+210, 10:10+160, :]
                    print "Previous observation: ", crop
                    print "Shape? ", crop.shape
                else:
                    crop=None

            ##Original code for action
            # action = agent.evaluate_actor(np.reshape(prevobv,[1,num_states])) #currently returning [ nan  nan  nan  nan  nan  nan]
            # noise = exploration_noise.noise()
            # action = action[0] + noise #Select action according to current policy and exploration noise
            # print "Noise: ", noise
  

            action = move(xcoord, ycoord, choose_distance(),choose_angle())

            print "Action at step", t ," :",action,"\n"
            
            observation,reward,done,info=env.step(action)
            env.render()
            print "Done?", done

            #add previous observation,current observation,action and reward to agent's experience memory
            agent.add_experience(crop,observation,action,reward,done)

            #train critic and actor network
            if counter > 64: #why 64? Perhaps to account for initialiisation? 
                agent.train()

            reward_per_episode+=reward[0]
            counter+=1
            #check if episode ends:
            if (done[0]==True or (t == steps-1)):
                print 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode
                print "Printing reward to file"
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print '\n\n'
                break
    total_reward+=reward_per_episode            
    print "Average reward per episode {}".format(total_reward / episodes)    


if __name__ == '__main__':
    main()    