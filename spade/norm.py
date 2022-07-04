from jittor import nn
import torch.nn.utils.spectral_norm as spectral_norm


def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)
    
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)
    
    return add_norm_layer


class Spade(nn.Module):
	def __init__(self, norm_nc, label_nc):
		super().__init__()

		self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
		ks = 3 # kernel_size: 3x3
		pad = 1 # padding
		num_hidden = 128

		self.shared_net = nn.Sequential(
			nn.Conv(in_channels=label_nc, out_channels=num_hidden, kernel_size=ks, padding=pad),
			nn.ReLU()
		)
		self.gamma_net = nn.Conv(in_channels=num_hidden, out_channels=norm_nc, kernel_size=ks, padding=pad)
		self.beta_net = nn.Conv(in_channels=num_hidden, out_channels=norm_nc, kernel_size=ks, padding=pad)
  
	def execute(self, x, segmap):
		# Part 1. generate parameter-free normalized activations
		normalized = self.param_free_norm(x)

		# Part 2. produce scaling and bias conditioned on semantic map
		segmap = nn.interpolate(segmap, size=x.size()[2:], mode="nearest")
		activated = self.shared_net(segmap)
		gamma = self.gamma_net(activated)
		beta = self.beta_net(activated)

		# apply scale and bias
		return normalized * (1 + gamma) + beta
  