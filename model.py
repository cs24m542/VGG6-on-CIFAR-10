import torch
import torch.nn as nn
import torch.nn.functional as F
# -------------------------------
# VGG-6 Model with configurable activation
# -------------------------------
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M',256,"M"]
class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
        )
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def make_layers(cfg, activation=nn.ReLU, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            

            if activation==nn.ReLU:
                act_layer = activation(inplace=True) if hasattr(activation, "__call__") else activation()
            else:
                act_layer = activation()
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act_layer]
            else:
                layers += [conv2d, act_layer]
            in_channels = v
    return nn.Sequential(*layers)

