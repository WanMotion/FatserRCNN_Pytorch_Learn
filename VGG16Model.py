from torch import nn


class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 64, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, stride=2),
                                   nn.Conv2d(64, 128, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, stride=2),
                                   nn.Conv2d(128, 256, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, stride=2),
                                   nn.Conv2d(256, 512, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, stride=2),
                                   nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
                                   nn.ReLU()
                                   )

    def forward(self, x):
        return self.model(x)
