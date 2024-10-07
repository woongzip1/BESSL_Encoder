import torch
import torch.nn as nn
from model_encoder import ResNet18
from model_decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super(AutoEncoder, self).__init__()
        
        self.encoder = ResNet18(in_channels=in_channels)
        self.decoder = Decoder(bottleneck_shape=in_channels * 8)

        self.initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def initialize_weights(self):
        # Iterate through all layers and apply Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    