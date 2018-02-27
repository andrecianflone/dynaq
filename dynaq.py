import maze
import numpy as np
import matplotlib.pyplot as plt

BLOCKING_MAZE1 = ['############',
                  '#          #',
                  '#          #',
                  '#          #',
                  '########## #',
                  '#          #',
                  '#   P      #',
                  '############']
BLOCKING_MAZE2 = ['############',
                  '#          #',
                  '#          #',
                  '#          #',
                  '# ##########',
                  '#          #',
                  '#   P      #',
                  '############']
class DynaQ():
  def __init__(self, game, n,alpha,gamma, epsilon, max_steps):
    self.game = game
    self.env = game.make(BLOCKING_MAZE1)
    self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    self.epsilon = epsilon
    self.model =  Model(self.env.observation_space.n, self.env.action_space.n)
    self.n = n

  def learn(self):
    """ Perform DynaQ learning, return cumulative return """
    s = self.env.reset() # initialize first state
    cum_reward = [0] # cumulative reward

    # Loop forever!
    for step in range(max_steps):
      # Epsilon greedy action
      if np.random.uniform() < self.epsilon:
        a = self.env.action_space.sample()
      else:
        a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])

      # Take action, observe outcome
      s_prime, r, done, info = self.env.step(a)

      # Q-Learning
      self.q[s,a] += alpha*(r + gamma*np.max(self.q[s_prime]) - self.q[s,a])

      # Learn model
      self.model.add(s,a,s_prime,r)

      # Planning for n steps
      self.planning()

      # Set state for next loop
      s = s_prime

      # Reset game if at the end
      if done:
        s = self.env.reset()

      # Check if time to switch board
      if step == 1000:
        self.env = self.game.make(BLOCKING_MAZE2)
        s = self.env.reset()

      # Add reward to count
      cum_reward.append(cum_reward[-1] + r)

    return np.array(cum_reward[1:])

  def planning(self):
    for i in range(self.n):
      s, a =  self.model.sample()
      s_prime, r = self.model.step(s,a)
      self.q[s,a] += alpha*(r + gamma*np.max(self.q[s_prime]) - self.q[s,a])

class Model():
  def __init__(self, n_states, n_actions):
    self.transitions = np.zeros((n_states,n_actions), dtype=np.uint8)
    self.rewards = np.zeros((n_states, n_actions))

  def add(self,s,a,s_prime,r):
    self.transitions[s,a] = s_prime
    self.rewards[s,a] = r

  def sample(self):
    """ Return random state, action"""
    # Random visited state
    s = np.random.choice(np.where(np.sum(self.transitions, axis=1) > 0)[0])
    # Random action in that state
    a = np.random.choice(np.where(self.transitions[s] > 0)[0])

    return s,a

  def step(self, s,a):
    """ Return state_prime and reward for state-action pair"""
    s_prime = self.transitions[s,a]
    r = self.rewards[s,a]
    return s_prime, r


def plot_data(y):
  """ y is a 1D vector """
  x = np.arange(y.size)
  _ = plt.plot(x, y, '-')
  plt.show()

def multi_plot_data(data, names):
  """ data, names are lists of vectors """
  x = np.arange(data[0].size)
  for i, y in enumerate(data):
    plt.plot(x, y, '-', markersize=2, label=names[i])
  plt.legend(loc='lower right', prop={'size': 16}, numpoints=5)
  plt.show()

if __name__ == '__main__':
  # Hyperparams

  alpha = 0.1 # learning rate
  gamma = 0.95 # discount
  epsilon = 0.3
  max_steps = 3000
  trials = 1

  dynaq_5_r = np.zeros((trials, max_steps))
  dynaq_50_r = np.zeros((trials, max_steps))
  qlearning_r = np.zeros((trials, max_steps))
  for t in range(trials):
    # DynaQ 5
    n = 5
    agent = DynaQ(maze, n, alpha, gamma, epsilon, max_steps)
    dynaq_5_r[t] = agent.learn()

    # DynaQ 50
    n = 50
    agent = DynaQ(maze, n, alpha, gamma, epsilon, max_steps)
    dynaq_50_r[t] = agent.learn()

    # Q-Learning
    n = 0
    agent = DynaQ(maze, n, alpha, gamma, epsilon, max_steps)
    qlearning_r[t] = agent.learn()

  # Average across trials
  dynaq_5_r = np.mean(dynaq_5_r, axis=0)
  dynaq_50_r = np.mean(dynaq_50_r, axis=0)
  qlearning_r = np.mean(qlearning_r, axis=0)

  data=[dynaq_5_r, dynaq_50_r, qlearning_r]
  names=["DynaQ, n=5", "DynaQ, n=50", "Q-Learning"]
  multi_plot_data(data,names)



