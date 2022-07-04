from jittor import nn
from spade.architecture import SpadeResnetBlock
from spade.base_network import BaseNetwork

class SpadeGenerator(BaseNetwork):
    def __init__(self, opt):
        super(SpadeGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf
        
        self.sw = opt.crop_size // (2 ** 5)
        self.sh = round(self.sw / opt.aspect_ratio)
        
        self.fc = nn.Conv(opt.semantic_nc, nf << 4, kernel_size=3, padding=1)
        self.head_0 = SpadeResnetBlock(nf << 4, nf << 4, opt)
        
        self.G_mid_0 = SpadeResnetBlock(nf << 4, nf << 4, opt)
        self.G_mid_1 = SpadeResnetBlock(nf << 4, nf << 4, opt)
        
        self.up_0 = SpadeResnetBlock(nf << 4, nf << 3, opt)
        self.up_1 = SpadeResnetBlock(nf << 3, nf << 2, opt)
        self.up_2 = SpadeResnetBlock(nf << 2, nf << 1, opt)
        self.up_3 = SpadeResnetBlock(nf << 1, nf, opt)
        final_nc = nf
        
        self.conv_img = nn.Conv(in_channels=final_nc, out_channels=3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        
        
    def execute(self, segmap):
        # downsample segmap and run convolution
        x = nn.interpolate(segmap, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, segmap)
        x = self.up(x)
        x = self.G_mid_0(x, segmap)
        x = self.G_mid_1(x, segmap)
        x = self.up(x)
        x = self.up_0(x, segmap)
        x = self.up(x)
        x = self.up_1(x, segmap)
        x = self.up(x)
        x = self.up_2(x, segmap)
        x = self.up(x)
        x = self.up_3(x, segmap)
        x = self.conv_img(nn.leaky_relu(x, scale=2e-1))
        x = nn.Tanh()(x)
        
        return x