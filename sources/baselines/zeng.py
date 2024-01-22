import torchvision.models as models
from torch import nn
import torch

import sys
sys.path.insert(0, "")

class Zeng(nn.Module):
    def __init__(self, channels=1, alphabet_size=41, pretrained=True, device="cpu"):
        super(Zeng, self).__init__()
        self.device = device
        if pretrained:
            resnet = models.resnet18(pretrained=True, progress=True)
            print("INFO -- Loading Imagenet Weights")
        else:
            resnet = models.resnet18(pretrained=False)
            if channels == 1:
                resnet.conv1.in_channels = 1
                resnet.conv1.kernel_size = (7, 7)
                resnet.conv1.weight = nn.Parameter((nn.init.xavier_normal_(torch.empty((64, 1, 7, 7)))))
            self.basenet = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                         resnet.layer2, resnet.layer3, resnet.layer4,
                                         resnet.avgpool)  # remove out layer

        # build ff module
        def init_weights(l):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)
                l.bias.data.fill_(0.01)

        self.feed_forward = nn.Sequential(
            nn.Flatten(),
        )

        self.feed_forward.apply(init_weights)
        # build classification head
        self.out_positions = nn.ModuleList(
            [nn.Linear(512, 2).apply(init_weights).to(self.device) for i in range(alphabet_size)])

    def forward(self, x):
        x = self.basenet(x)
        x = self.feed_forward(x)
        x = [torch.unsqueeze(l(x), 0) for l in self.out_positions]
        x_out = torch.cat(*[x], axis=0)
        return x_out
