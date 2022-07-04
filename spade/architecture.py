from jittor import nn
from spade.norm import Spade
import torch.nn.utils.spectral_norm as spectral_norm
from jittor import models
import jittor as jt

class SpadeResnetBlock(nn.Module):

	def __init__(self, in_channels, out_channels, opt):
		super().__init__()
		
		# Attributes
		self.learned_shortcut = (in_channels != out_channels)
		mid_channels = min(in_channels, out_channels)
		
		# create conv layers
		self.conv_0 = nn.Conv(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
		self.conv_1 = nn.Conv(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1)
		if self.learned_shortcut:
			self.conv_s = nn.Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
					
		# define normalization layers
		self.norm_0 = Spade(in_channels, opt.semantic_nc)
		self.norm_1 = Spade(mid_channels, opt.semantic_nc)
		if self.learned_shortcut:
			self.norm_s = Spade(in_channels, opt.semantic_nc)

	def execute(self, x, segmap):
		x_shortcut = self.shortcut(x, segmap)

		dx = self.norm_0(x, segmap)
		dx = nn.leaky_relu(dx, scale=2e-1)
		dx = self.conv_0(dx)

		dx = self.norm_1(dx, segmap)
		dx = nn.leaky_relu(dx, scale=2e-1)
		dx = self.conv_1(dx)
  
		return dx + x_shortcut
  
	def shortcut(self, x, segmap):
		if self.learned_shortcut:
			return self.conv_s(self.norm_s(x, segmap))
		else:
			return x


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(jt.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = jt.nn.Sequential()
        self.slice2 = jt.nn.Sequential()
        self.slice3 = jt.nn.Sequential()
        self.slice4 = jt.nn.Sequential()
        self.slice5 = jt.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
