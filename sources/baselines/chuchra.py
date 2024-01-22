import sys

import torch
from torch import nn

sys.path.insert(0, "")


class Chuchra(nn.Module):
    def __init__(self, alphabet_size=41, device="cpu"):
        super(Chuchra, self).__init__()
        self.device = device

        # build ff module
        def init_weights(l):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)
                l.bias.data.fill_(0.01)

        # Convolutional Stem
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Dropout(0.1),
        )
        self.conv_layers.apply(init_weights)  # init cnn layers

        # Feed Forward Module
        self.feed_forward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 17, 32),
            nn.ReLU(),
        )
        self.feed_forward.apply(init_weights).to(self.device)
        self.out_pos_list = nn.ModuleList(
            [nn.Linear(32, 2).apply(init_weights).to(self.device) for i in range(alphabet_size)])

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.feed_forward(x)
        x = [torch.unsqueeze(l(x), 0) for l in self.out_pos_list]
        x_out = torch.cat(*[x], axis=0)
        return x_out
