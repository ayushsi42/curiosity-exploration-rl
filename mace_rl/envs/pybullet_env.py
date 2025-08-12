import gym
import pybullet_envs

def make_pybullet_env(env_name):
    """
    Creates a PyBullet environment.
    """
    env = gym.make(env_name)
    return env

if __name__ == '__main__':
    env = make_pybullet_env('MinitaurBulletEnv-v0')
    obs = env.reset()
    print("Observation shape:", obs.shape)
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Next observation shape:", obs.shape)
    env.close()

