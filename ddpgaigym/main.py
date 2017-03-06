#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Adaptation

import gym
from gym.spaces import Box, Discrete
import universe
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
#specify parameters here:
episodes=10000
is_batch_norm = False #batch normalization switch

def main():
    experiment= 'wob.mini.ClickTest-v0' #specify environments here
    env= gym.make(experiment)
    print "Observation space for ", experiment, ": ", env.observation_space
    print "Action space for ", experiment, ": ", env.action_space
    steps= env.spec.timestep_limit #steps per episode   
    env.configure(remotes=1, fps=5,
              vnc_driver='go', 
              vnc_kwargs={'encoding': 'tight', 'compress_level': 90, 
                          'fine_quality_level': 100, 'subsample_level': 0})
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
        for t in xrange(steps):
            #rendering environment            
            env.render()
            prevobv = observation
            print "Previous observation: ", prevobv

            ##Original code (needs to be adapted)
            # action = agent.evaluate_actor(np.reshape(x,[1,num_states])) #currently returning [ nan  nan  nan  nan  nan  nan]
            # noise = exploration_noise.noise()
            # action = action[0] + noise #Select action according to current policy and exploration noise
            # print "Noise: ", noise

            #temp placeholder code for actions
            xcoord = np.random.randint(0, 160) + 10  
            ycoord = np.random.randint(0, 160) + 75 + 50   

            action = [[universe.spaces.PointerEvent(xcoord, ycoord, 0),
            universe.spaces.PointerEvent(xcoord, ycoord, 1),
            universe.spaces.PointerEvent(xcoord, ycoord, 0)]]

            print "Action at step", t ," :",action,"\n"
            
            observation,reward,done,info=env.step(action)
            env.render()
            print "Done?", done
            print "Current observation", observation[0]

            #add previous observation,current observation,action and reward to agent's experience memory
            agent.add_experience(prevobv,observation,action,reward,done)

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