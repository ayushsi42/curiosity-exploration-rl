import gym
from gym import spaces
import numpy as np
from minigrid.wrappers import *
from mace_rl.utils.logger import get_logger

logger = get_logger('MiniGridEnv')

class MiniGridWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        logger.debug(f"Initializing MiniGridWrapper for environment: {env}")
        
        # Keep the original image shape for the CNN to process
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.env.observation_space['image'].shape[0],
                self.env.observation_space['image'].shape[1],
                3  # RGB channels
            ),
            dtype=np.uint8
        )
        logger.debug(f"Created observation space: {self.observation_space}")

    def observation(self, obs):
        try:
            # Return the image part of the observation
            image = obs['image']
            logger.debug(f"Processing observation with shape: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error processing observation: {e}")
            raise

def make_minigrid_env(env_name):
    """
    Creates and wraps a MiniGrid environment.
    """
    logger.info(f"Creating MiniGrid environment: {env_name}")
    try:
        env = gym.make(env_name)
        logger.debug(f"Base environment created with observation space: {env.observation_space}")
        
        env = MiniGridWrapper(env)
        logger.debug(f"Environment wrapped with observation space: {env.observation_space}")
        
        return env
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        raise

if __name__ == '__main__':
    env = make_minigrid_env('MiniGrid-Empty-5x5-v0')
    obs = env.reset()
    print("Observation shape:", obs.shape)
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Next observation shape:", obs.shape)
    env.close()

