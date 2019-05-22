import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """ 
    This is the policy network used by the agent to take a decision on its next action.
    """
    def __init__(self, h, w, channels, outputs):
        """
        Initializes a new instance of DQN

        Parameters:
            h (int): The height of the input screen
            w (int): The width of the input screen
            channels (int): The number of channels in the input screen
            outputs (int): The number of actions to choose from
        """
        super(DQN, self).__init__()

        def conv_output_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Each conv2D layer resizes the input as calculated in the formula above.
        # Since we're stacking layers, we need to stack the output size calculation as well.
        features_w = conv_output_size(conv_output_size(conv_output_size(w, 8, 4), 4, 2), 3, 1)
        features_h = conv_output_size(conv_output_size(conv_output_size(h, 8, 4), 4, 2), 3, 1)

        self.conv1 = nn.Conv2d(channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(features_w*features_h*64, 512)
        self.output = nn.Linear(512, outputs)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        """
        Performs the forward pass through the neural network

        Parameters:
            x (tensor): The input tensor for the policy network
        """
        y = self.relu1(self.conv1(x))
        y = self.relu2(self.conv2(y))
        y = self.relu3(self.conv3(y))
        
        # Flatten the output of the convolutional layer.
        # This is required to match the shape with the linear layer.
        y = y.view(y.size(0), -1)

        y = self.relu4(self.fc(y))
        y = self.output(y)

        return y
