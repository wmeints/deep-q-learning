import numpy as np
import gym
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ResizeWrapper, self).__init__(env)
    
    def observation(self, observation):
        # Crop and resize the image
        img = observation[1:176:2, ::2]

        # Convert the image to greyscale
        img = img.mean(axis=2)

        # Next we normalize the image from -1 to +1
        img = (img - 128) / 128 - 1

        return img.reshape(88,80,1)

class TorchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TorchWrapper, self).__init__(env)

    def observation(self, observation):
        output = np.rollaxis(observation, 2, 0)
        output = output.reshape(-1, output.shape[0], output.shape[1], output.shape[2]).astype(np.float32)

        return torch.from_numpy(output).to(device)