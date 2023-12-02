
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from ma_gym.envs.power_grid import PowerGrid
from ma_gym.envs.checkers import Checkers
from ma_gym.envs.switch import Switch
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
import time

# register all scenario with env class
REGISTRY = {}
REGISTRY["PowerGrid"] = PowerGrid
REGISTRY["Checkers"] = Checkers
REGISTRY["Switch2"] = Switch

# provide detailed information of each scenario
# mostly for policy sharing
policy_mapping_dict = {
    "Checkers": {
        "description": "two team cooperate",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "Switch2": {
        "description": "two team cooperate",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "PowerGrid": {
        "description": "five team cooperate",
        "team_prefix": ("1", "2", "3", "4", "5"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

# must inherited from MultiAgentEnv class
class RLlibMAGym(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)

        self.env = REGISTRY[map](**env_config)
        # assume all agent same action/obs space
        self.action_space = self.env.action_space[0]
        self.observation_space = GymDict({"obs": Box(low=np.array([0.0002, 1.00]), high=np.array([0.0005, 8.00]),
            shape=(self.env.observation_space[0].shape[0],),
            dtype=np.dtype("float32"))})
        self.agents = ["1", "2", "3", "4", "5"]
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": np.array(original_obs[i])}
        return obs

    def step(self, action_dict):
        action_ls = [action_dict[key] for key in action_dict.keys()]
        o, r, d, info = self.env.step(action_ls)
        rewards = {}
        obs = {}
        for i, key in enumerate(action_dict.keys()):
            rewards[key] = r[i]
            obs[key] = {
                "obs": np.array(o[i])
            }
        dones = {"__all__": True if sum(d) == self.num_agents else False}
        return obs, rewards, dones, {}

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 100,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["magym"] = RLlibMAGym
    # initialize env
    env = marl.make_env(environment_name="magym", map_name="PowerGrid", abs_path="../../examples/config/env_config/magym.yaml")
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
    # start learning
    mappo.fit(env, model, stop={'timesteps_total': 10000000}, local_mode=True, num_gpus=0,
              num_workers=5, share_policy = "all", checkpoint_freq=50)