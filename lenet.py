from collections import OrderedDict

import torch
import torch.nn as nn
from custompooling import CustomPooling


class LeNet(nn.Module):

    def __init__(self, num_classes=10, num_channels=3, pooling_function='max'):
        super(LeNet, self).__init__()
        self.pooling_layer = CustomPooling(pooling_function=pooling_function)
        self.layers = OrderedDict([
            ('conv1', nn.Conv2d(in_channels=num_channels, out_channels=6, kernel_size=5, stride=1, padding=0)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pooling1', self.pooling_layer),
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pooling2', self.pooling_layer),
            ('conv3', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)),
            ('relu3', nn.ReLU(inplace=True)),
        ])
        self.features = nn.Sequential(self.layers)

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def all_layers_output(self, input):
        outputs = OrderedDict()
        input = input.to('cpu')
        if torch.cuda.is_available():
            input = input.to('cuda')
        last_output = input.reshape(-1, input.size(0), input.size(1), input.size(2))
        outputs.update({'initial': self.pooling_layer(input)})
        for layer_name, layer in self.layers.items():
            last_output = layer(last_output)
            if 'conv' in layer_name or 'pooling' in layer_name:
                maxlayer = last_output.sum(dim=2).argmax(dim=1)
                visualized = last_output[:, maxlayer[0, 0], :, :].unsqueeze(dim=1)
                print(visualized.shape)
                outputs.update({layer_name: visualized})
        return outputs

    def visTensor(self, tensor, ch=0, allkernels=False, nrow=8, padding=1):
        n, c, w, h = tensor.shape

        if allkernels:
            tensor = tensor.view(n * c, -1, w, h)
        elif c != 3:
            tensor = tensor[:, ch, :, :].unsqueeze(dim=1)
        return tensor

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
