import gym
import universe # register the universe environments
import numpy as np

def forward(ob):
  """ 
  Takes raw (768,1024,3) uint8 screen and returns list of VNC events.
  The browser window indents the origin of MiniWob by 75 pixels from top and
  10 pixels from the left. The first 50 pixels along height are the query.
  """
  if ob is None: return []

  x = ob['vision']
  crop = x[75:75+210, 10:10+160, :]               # miniwob coordinates crop
  xcoord = np.random.randint(0, 160) + 10         # todo: something more clever here
  ycoord = np.random.randint(0, 160) + 75 + 50    # todo: something more clever here

  # 1. move to x,y with left button released, and click there (2. and 3.)
  action = [universe.spaces.PointerEvent(xcoord, ycoord, 0),
            universe.spaces.PointerEvent(xcoord, ycoord, 1),
            universe.spaces.PointerEvent(xcoord, ycoord, 0)]

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