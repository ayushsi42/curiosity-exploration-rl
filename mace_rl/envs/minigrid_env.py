import gym
from gym_minigrid.wrappers import *

class MiniGridWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Flatten the observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.observation_space['image'].shape[0] * self.env.observation_space['image'].shape[1] * 3,),
            dtype='uint8'
        )

    def observation(self, obs):
        return obs['image'].flatten()

def make_minigrid_env(env_name):
    """
    Creates and wraps a MiniGrid environment.
    """
    env = gym.make(env_name)
    env = MiniGridWrapper(env)
    return env

if __name__ == '__main__':
    env = make_minigrid_env('MiniGrid-Empty-5x5-v0')
    obs = env.reset()
    print("Observation shape:", obs.shape)
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Next observation shape:", obs.shape)
    env.close()

