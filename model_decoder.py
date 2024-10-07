import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torchinfo import summary
from model_encoder import ResNet18

## Conv-ReLU-Conv with Residual Connection
class ResBlock(nn.Module):
    def __init__(self, n_ch):
        super(ResBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = weight_norm(nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1))

    def forward(self, x, final=False):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += identity
        if final:
            out = x
        else:
            out = self.relu(x)
        return out

class Decoder(nn.Module):
    def __init__(self, bottleneck_shape=16):
        super(Decoder, self).__init__()
        self.bottleneck_shape = bottleneck_shape

        # Reducing the padding and dilation to control the output size
        self.c1 = nn.ConvTranspose2d(self.bottleneck_shape, self.bottleneck_shape//2, kernel_size=(4,3), stride=(2,1), padding=(1,1), )
        self.c2 = nn.ConvTranspose2d(self.bottleneck_shape//2, self.bottleneck_shape//4, kernel_size=(4,3), stride=(2,1), padding=(1,1), )
        self.c3 = nn.ConvTranspose2d(self.bottleneck_shape//4, self.bottleneck_shape//8, kernel_size=(4,3), stride=(2,1), padding=(1,1), )
        self.c4 = nn.ConvTranspose2d(self.bottleneck_shape//8, self.bottleneck_shape//16, kernel_size=(4,3), stride=(2,1), padding=(1,1), )
        self.c5 = nn.ConvTranspose2d(self.bottleneck_shape//16, 1, kernel_size=(4,3), stride=(2,1), padding=(1,1), )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        return x

if __name__ == "__main__":
# Example input tensor with shape B x 512 x 10 x 80
    encoder = ResNet18()
    input = torch.rand(8,1,32*26, 40)
    out = encoder(input)

    model = Decoder()
    print(summary(model, input_data=out))
