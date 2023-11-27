#powergrid is an environment for managing the power generation and supply in a colony.
#battery capacity for different building will be different. But all of them will be in between 5 to 15 kwh.
import gym
from gym import spaces
import numpy as np
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace 
import random
import logging
logger = logging.getLogger(__name__)

lowest_generation_capacity = 10**-4   #per second
highest_generation_capacity = 25*10**(-4) # per second
battery_capacity_range = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
battery_capacity_range_bias = [0.02, 0.02, 0.02, 0.1, 0.2, 0.3, 0.2, 0.1, 0.02, 0.02, 0.02]

class PowerGrid(gym.Env):
  #action space mapping
  #0->After generating, charge battery and load from battery
  #1->After generating, charge battery and load from micro grid
  #2->After generating, charge battery and load from grid
  #3->After generating, give it to micro grid and load from battery
  #4->After generating, give it to micro grid and load from micro grid
  #5->After generating, give it to micro grid and load from grid
  #6->After generating, give it to grid and load from battery
  #7->After generating, give it to grid and load from micro grid
  #8->After generating, give it to grid and load from grid

  def __init__(self, n_max = 5, buy_from_grid_cost = 15, sell_to_grid_cost = 8, buy_from_microgrid_cost = 10, sell_to_microgrid_cost = 10, max_steps: int = 10000) -> None:
    assert 4 <= n_max <= 8, "n_max should be range in [4,8]"
    self.n_agents = n_max
    self.max_steps = max_steps
    self.step_count = 0
    self.buy_from_grid_cost = buy_from_grid_cost
    self.sell_to_grid_cost = sell_to_grid_cost
    self.buy_from_microgrid_cost = buy_from_microgrid_cost
    self.sell_to_microgrid_cost = sell_to_microgrid_cost
    self.microgrid_extra_power = 0
    self.battery_capacities = self.battery_capacity_seleciton()
    self.battery_current = [0 for _ in range(self.n_agents)]
    self.highest_generation_capacities = self.highest_generate_capacity_selection(lowest_generation_capacity, highest_generation_capacity)
    self.current_weather = random.uniform(1,8)
    self._total_episode_reward = 0
    self.viewer = None   
    self.action_space = MultiAgentActionSpace([spaces.Discrete(9) for _ in range(self.n_agents)])
    self.observation_space = MultiAgentObservationSpace([spaces.Box(low=np.array([0.0002, 1.00]), high=np.array([0.0005, 8.00]), dtype=np.float32)
                                                             for _ in range(self.n_agents)])

  #select the battery capacities for each building
  def battery_capacity_seleciton(self):
    battery_capacity_lst = []
    for i in range(5):
      a = random.choices(battery_capacity_range, battery_capacity_range_bias)
      battery_capacity_lst.append(a)
    return battery_capacity_lst
  
  #select the generation capacities for each building
  def highest_generate_capacity_selection(self, low, high):
    highest_generate_capacity_lst = []
    for i in range(5):
      a = random.uniform(low, high)
      highest_generate_capacity_lst.append(a)
    return highest_generate_capacity_lst
  
  #step: maybe the most important function. Look out for bugs
  def step(self, agents_action):
    assert len(agents_action) == self.n_agents, \
            "Invalid action! It was expected to be list of {}" \
            " dimension but was found to be of {}".format(self.n_agents, len(agents_action))
    self.step_count += 1

    #the observation part. we do only need to sample the load at every time step. we will change the weather at every 1800th step
    if(self.step_count % 1800 == 0):
      self.current_weather = random.uniform(1, 8)
    observations = self.observation_space.sample()
    for x in observations:
      x[1] = self.current_weather

    #rewards
    rewards = [0 for _ in range(self.n_agents)]

    for agent_i, action in enumerate(agents_action):

      #charge battery and load from battery
      if(action == 0):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        self.battery_current[agent_i] += generated_power
        rewards[agent_i] += 100 * (generated_power - observations[agent_i][0])
        self.battery_current[agent_i] -= generated_power

      #charge battery and load from microgrid
      elif(action == 1):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        self.battery_current[agent_i] += generated_power
        load = observations[agent_i][0]
        if(load <= self.microgrid_extra_power):
          self.microgrid_extra_power -= load
          rewards[agent_i] += 100 * 0 + ((-1) * load * self.buy_from_microgrid_cost) 
        else:
          rewards[agent_i] += 100 * ((-1) * load)  # not sure about the logic

      #charge battery and load from grid
      elif(action == 2):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        self.battery_current[agent_i] += generated_power
        load = observations[agent_i][0]
        rewards[agent_i] = 100 * 0 + ((-1) * load * self.buy_from_grid_cost) # i guess, eikhane takar hishab ashbe
      
      #give to micro grid and load from battery
      elif(action == 3):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        load = observations[agent_i][0]
        self.microgrid_extra_power += generated_power
        if(self.battery_current[agent_i] >= load):
          self.battery_current[agent_i] -= load
          rewards[agent_i] += (100 * 0) + generated_power * self.sell_to_microgrid_cost
        else:
          rewards[agent_i] += (100 * (-1) * load) + generated_power * self.sell_to_microgrid_cost
      
      #give to micro grid and load from micro grid
      elif(action == 4):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        load = observations[agent_i][0]
        self.microgrid_extra_power += generated_power
        if(self.microgrid_extra_power >= load):
          self.microgrid_extra_power -= load
          rewards[agent_i] += (100 * 0) + self.sell_to_microgrid_cost
        else:
          rewards[agent_i] + (100 * (-1)* load) + self.sell_to_microgrid_cost
       
      #give to micro grid and load from grid 
      elif(action == 5):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        load = observations[agent_i][0]
        self.microgrid_extra_power += generated_power
        rewards[agent_i] = (100 * 0) + (generated_power * self.sell_to_microgrid_cost) + ((-1) * load * self.buy_from_grid_cost)
      
      #give it to grid and load from battery
      elif(action == 6):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        load = observations[agent_i][0]
        if(self.battery_current[agent_i] >= load):
          self.battery_current[agent_i] -= load
          rewards[agent_i] += (100 * 0) + (generated_power * self.sell_to_grid_cost)
        else:
          rewards[agent_i] += (100 * (-1) * load) + (generated_power *self.sell_to_grid_cost )
      
      #give it to grid and load from micro grid
      elif(action == 7):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        load = observations[agent_i][0]
        if(self.microgrid_extra_power >= load):
          self.microgrid_extra_power -= load
          rewards[agent_i] += (100 * 0) + (generated_power * self.sell_to_grid_cost)
        else:
          rewards[agent_i] += (100 * (-1) * load) + (generated_power * self.buy_from_microgrid_cost)
      
      #give it to grid and load from grid
      elif(action == 8):
        generated_power = self.highest_generation_capacities[agent_i] - ((self.highest_generation_capacities[agent_i] * observations[agent_i][1]) / 8)
        load = observations[agent_i][0]
        rewards[agent_i] += (100 * 0) + (generated_power * (self.sell_to_grid_cost)) + (generated_power * (-1) * self.buy_from_grid_cost) 
        self._total_episode_reward[agent_i] += rewards[agent_i]
    return observations, rewards, {}, {}
    
  def reset(self):
    self.step_count = None
    self.microgrid_extra_power = 0
    self.battery_capacities = self.battery_capacity_seleciton()
    self.battery_current = [0 for _ in range(self.n_agents)]
    self.highest_generation_capacities = self.highest_generate_capacity_selection(lowest_generation_capacity, highest_generation_capacity)
    self.current_weather = random.uniform(1,8)
    self._total_episode_reward = None
    self.viewer = None   
    return self.observation_space.sample()
  
  def close(self):
      if self.viewer is not None:
          self.viewer.close()
          self.viewer = None


  

