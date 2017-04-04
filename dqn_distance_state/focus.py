import gym

import universe # register the universe environments

import numpy as np
from matplotlib import pyplot as plt
import math
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse

from RL_brain import DeepQNetwork
resetX,resetY = 50,140
cur_x,cur_y =resetX,resetY
#get vector a->b
def step_reward(last_state,this_state):
    if abs(last_state[0])+abs(last_state[1])>abs(this_state[0])+abs(this_state[1]):
        return 1
    return -1
def avoid_escape(coords):
    minY = 125
    maxY = 125+160
    minX = 10
    maxX = 10+160
    if coords[0]<maxY-15 or coords[1]<maxX-15:
       return 1
    
    return 0
def penalty(coords):
    r= distance(coords,[0,0])*(-0.02)
    return r
def distance(pointA,pointB):
    return math.sqrt((pointA[0]-pointB[0])**2+(pointA[1]-pointB[1])**2)
def intToVNCAction(a, x, y):
    small_step = 6
    minY = 125
    maxY = 125+160
    minX = 10
    maxX = 10+160
    if a == 0:
        return [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, 1),
            universe.spaces.PointerEvent(x, y, 0)], x, y
    #elif a == 1:
        #return [universe.spaces.PointerEvent(x, y, 0)], x, y
    elif a == 1:
        if y + small_step <= maxY:
            return [universe.spaces.PointerEvent(x, y + small_step, 0)], x, y + small_step

    elif a == 2:
        if y - small_step >= minY:
            return [universe.spaces.PointerEvent(x, y - small_step, 0)], x, y - small_step

    return [], x, y
def centre_button(ob):

  if ob is None: 
    return -1,-1
  x = ob['vision']
  crop = x[75:75+50+160, 10:10+160, :]               # miniwob coordinates crop
  square = x[75+50:75+50+160, 10:10+160, :]  
  gray =rgb2gray(square)
  coords = corner_peaks(corner_harris(gray), min_distance=5)
  coords_subpix = corner_subpix(gray, coords, window_size=13)
  newy = coords_subpix[:,0]
  newx = coords_subpix[:,1]
  newy = newy[np.logical_not(np.isnan(newy))]
  newx = newx[np.logical_not(np.isnan(newx))]

  goal_y,goal_x = np.mean(newy)+125,np.mean(newx)+10
  if math.isnan(goal_y) or math.isnan(goal_x) or goal_y ==None:
    return -1,-1
  
  return goal_y,goal_x

def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

  return gray
def forward(ob):
  """ 
  Takes raw (768,1024,3) uint8 screen and returns list of VNC events.
  The browser window indents the origin of MiniWob 
  by 75 pixels from top and
  10 pixels from the left. 
  The first 50 pixels along height are the query.
  """
  if ob is None: return []

  x = ob['vision']
  crop = x[75:75+50+160, 10:10+160, :]               # miniwob coordinates crop
  square = x[75+50:75+50+160, 10:10+160, :]  
  gray =rgb2gray(square)
  print gray
  coords = corner_peaks(corner_harris(gray), min_distance=5)
  coords_subpix = corner_subpix(gray, coords, window_size=13)
  for item in coords_subpix:
    pass
    #print item[0]+75+50,item[1]+10
  newy = coords_subpix[:,0]
  newx = coords_subpix[:,1]
  newy = newy[np.logical_not(np.isnan(newy))]
  newx = newx[np.logical_not(np.isnan(newx))]
  #if newx == None or newy == None:
    #return []

  
  goal_y,goal_x = np.mean(newy)+125,np.mean(newx)+10
  if math.isnan(goal_y) or math.isnan(goal_x):
    return []
  
  print goal_y,goal_x
  #xcoord = np.random.randint(0, 160) + 10         # todo: something more clever here
  #ycoord = np.random.randint(0, 160) + 75 + 50    # todo: something more clever here
  #print ycoord,xcoord
  # 1. move to x,y with left button released, and click there (2. and 3.)
  action = [universe.spaces.PointerEvent(goal_x, goal_y, 0),
            universe.spaces.PointerEvent(goal_x, goal_y, 1),
            universe.spaces.PointerEvent(goal_x, goal_y, 0)]

  return action

env = gym.make('wob.mini.FocusText-v0')
# automatically creates a local docker container
env.configure(remotes=1, fps=5,
              vnc_driver='go', 
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 
                          'fine_quality_level': 100, 'subsample_level': 0})
observation_n = env.reset()
#
step = 0
n_act =3
n_features =2
cur_y,cur_x=resetY,resetX
RL = DeepQNetwork(n_act, n_features,
              learning_rate=0.11,
              reward_decay=0.9,
              e_greedy=0.9,
              hidden_layers=[10, 10],
              replace_target_iter=200,
              memory_size=4000,
              # output_graph=True
              )
#

while True:
  goal_y,goal_x = centre_button(observation_n[0])
  if goal_y  ==-1:
    observation_n, reward_n, done_n, info = env.step([[universe.spaces.PointerEvent(resetX, resetY, 0)]])
    cur_x,cur_y=resetX,resetY
    env.render()
    continue
  state=[goal_y-cur_y,goal_x-cur_x]
  #state = [int(round(n)) for n in state] 
  state = np.array(state)
  action_index = RL.choose_action(state)
  int_action = ["click","down","up"]
  print 'action is ',int_action[action_index]
  operation,x,y = intToVNCAction(action_index, cur_x, cur_y)

  action = [universe.spaces.PointerEvent(goal_x, goal_y, 0),
            universe.spaces.PointerEvent(goal_x, goal_y, 1),
            universe.spaces.PointerEvent(goal_x, goal_y, 0)]
  action_n = [operation] # your agent here
  
  observation_n, reward_n, done_n, info = env.step(action_n)
  cur_y,cur_x= y,x;

  click_y,click_x = centre_button(observation_n[0])
  next_state=[click_y-cur_y,click_x-cur_x]
  #next_state = [int(round(n)) for n in next_state] 
  next_state = np.array(next_state)
  if reward_n[0]<0:
    reward = 0
    observation_n, reward_n, done_n, info = env.step([[universe.spaces.PointerEvent(resetX, resetY, 0)]])
    cur_x,cur_y=resetX,resetY
  elif reward_n[0]>0:
    next_state=np.array([0,0])
    reward = 15
    observation_n, reward_n, done_n, info = env.step([[universe.spaces.PointerEvent(resetX, resetY, 0)]])
    cur_x,cur_y=resetX,resetY
  else :
    reward = 0
  r1 = step_reward(state,next_state)
  if action_index==0:
    r1=0
  p=0
  if action_index==0:
    p=penalty(state)
  if not reward_n[0]<0 :
  
    RL.store_transition(state, action_index, reward+r1+p, next_state)


  print state,next_state,step,reward,r1,p,RL.epsilon
  if (step > 200) and (step % 5 == 0):
      #pass
      RL.learn()
  step+=1
  env.render()
