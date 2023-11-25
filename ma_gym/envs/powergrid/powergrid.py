#powergrid is an environment for managing the power generation and supply in a colony.
#battery capacity for different building will be different. But all of them will be in between 5 to 15 kwh.
import gym
from gym import spaces
import numpy as np
from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace 
import random

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

  def __init__(self, n_max = 5, buy_from_grid_cost = 15, sell_to_grid_cost = 8, buy_from_microgrid_cost = 10, sell_to_microgrid_cost = 10, max_steps: int = 1000) -> None:
    assert 4 <= n_max <= 8, "n_max should be range in [4,8]"
    self.n_agents = n_max
    self.max_steps = max_steps
    self.step_count = None
    self.buy_from_grid_cost = buy_from_grid_cost
    self.sell_to_grid_cost = sell_to_grid_cost
    self.buy_from_microgrid_cost = buy_from_microgrid_cost
    self.sell_to_microgrid_cost = sell_to_microgrid_cost
    self.battery_capacities = self.battery_capacity_seleciton()
    self.highest_generation_capacities = self.highest_generate_capacity_selection(lowest_generation_capacity, highest_generation_capacity)
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
  
  
# print(highest_generate_capacity_selection(lowest_generation_capacity, highest_generation_capacity))
# print(battery_capacity_seleciton())
