import gymnasium as gym
import pybullet_envs_gymnasium  # instead of pybullet_envs

def make_pybullet_env(env_name):
    """
    Creates a PyBullet environment.
    """
    env = gym.make(env_name)
    return env

if __name__ == '__main__':
    env = make_pybullet_env('MinitaurBulletEnv-v0')
    
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("Next observation shape:", obs.shape)
    
    env.close()