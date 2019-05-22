import gym
from gym_wrappers import ResizeWrapper, TorchWrapper


def create_environment(name='SpaceInvaders-v0'):
    env = gym.make(name)
    env = ResizeWrapper(env)
    env = TorchWrapper(env)
    env = gym.wrappers.Monitor(env, 'logs', force=True, write_upon_reset=True)

    return env
