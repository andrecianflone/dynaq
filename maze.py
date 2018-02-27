# Author: Andre Cianflone
# Some code Based on:
# https://github.com/deepmind/pycolab/blob/master/pycolab/examples/classics/four_rooms.py
# Added wrapper so behaves more like gym

import curses
import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites
# from gym import spaces
import numpy as np


BLOCKING_MAZE1 = ['############',
                  '#          #',
                  '#          #',
                  '#          #',
                  '########## #',
                  '#          #',
                  '#   P      #',
                  '############']

def make(map):
  return Gymify(map)

class Discrete():
  def __init__(self, n):
    self.n = n # number of actions

  def sample(self):
    return np.random.randint(0,self.n)

class Box():
  def __init__(self,low,high,shape,dtype):
    self.low = low
    self.high = high
    self.shape = shape
    self.dtype = dtype
    self.n = self.shape # number of states

class Gymify():
  """ Create Gym like environment """

  def __init__(self, map):
    self.map = map
    self.engine = self.new_engine()
    self.last_obs = None

    # These should come from gym.spaces, but to speed up colab, they are internal
    self.action_space = Discrete(4)
    self.observation_space = Box(
        low=0,
        high=1,
        shape=self.engine.rows*self.engine.cols,
        dtype='uint8'
        )

  def new_engine(self):
    return ascii_art.ascii_art_to_game(self.map, what_lies_beneath=' ',
                                                  sprites={'P': PlayerSprite})
  def reset(self):
    """ Reset game engine """
    self.engine = self.new_engine()
    obs, reward, gamma = self.engine.its_showtime();
    obs = self.observation_to_state(obs)
    return obs

  def step(self, move):
    """ Returns gym style step """
    # Get data from game engine
    obs, reward, gamma = self.engine.play(move)
    # Save the last observation in case of rendering
    self.last_obs = obs
    done = self.engine.game_over
    # Format other vars
    if reward == None: reward = 0
    info = ""
    # Format observation into proper state for RL
    obs = self.observation_to_state(obs)
    return obs, reward, done, info

  def observation_to_state(self, obs):
    """
    Given a Pycolab Observation object, encode to a state representation,
    which is basically the agent's current position in the board
    """
    arr = np.array(obs.layers['P'], dtype=np.float).flatten()
    return np.argmax(arr)

  def render(self):
    """ Render the last observation """
    for row in self.last_obs.board: print(row.tostring().decode('ascii'))

  # def __getattr__(self, name):
    # """ For unknown attr, try engine"""
    # self.engine.__dict__[name]

class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player.

  This `Sprite` ties actions to going in the four cardinal directions. If we
  reach a magical location (in this example, (4, 3)), the agent receives a
  reward of 1 and the epsiode terminates.
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del layers, backdrop, things   # Unused.

    # Apply motion commands.
    if actions == 0:    # walk upward?
      self._north(board, the_plot)
    elif actions == 1:  # walk downward?
      self._south(board, the_plot)
    elif actions == 2:  # walk leftward?
      self._west(board, the_plot)
    elif actions == 3:  # walk rightward?
      self._east(board, the_plot)

    # See if we've found the mystery spot.
    if self.position == (1, 10):
      the_plot.add_reward(1.0)
      the_plot.terminate_episode()

def main(argv=()):
  del argv  # Unused.

  # Build a four-rooms game.
  game = make_game()

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4},
      delay=200)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)
