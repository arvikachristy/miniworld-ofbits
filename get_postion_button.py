import gym

import universe # register the universe environments
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse
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

env = gym.make('wob.mini.ClickTest-v0')
# automatically creates a local docker container
env.configure(remotes=1, fps=5,
              vnc_driver='go', 
              vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 
                          'fine_quality_level': 100, 'subsample_level': 0})
observation_n = env.reset()

while True:
  action_n = [forward(ob) for ob in observation_n] # your agent here
  
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()
