from jittor import nn
import jittor as jt
import numpy as np
from spade.base_network import BaseNetwork
import importlib
import jittor.pool as pool
from spade.norm import get_nonspade_norm_layer

def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.num_D = 2
        opt.netD_subarch = 'n_layer'
        self.opt = opt
        
        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)
        
    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return pool.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=1,
                            count_include_pad=False)

    def execute(self,input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for D in self.children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
            
        return result
    
    def add_module(self, name, mod):
        self.__dict__[name] = mod


class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.n_layers_D = 4
        self.opt = opt
        
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)
        
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv(input_nc, nf, kernel_size=kw, stride=2, padding=padw, bias=False),
                     nn.LeakyReLU(0.2)]]
        
        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw, bias=False)),
                          nn.LeakyReLU(0.2)
                          ]]

        sequence += [[nn.Conv(nf, 1, kernel_size=kw, stride=1, padding=padw, bias=False)]]
        
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))
    
    def add_module(self, name, mod):
        self.__dict__[name] = mod
    
    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc
    
    def execute(self,input):
        results = [input]
        
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
            
        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]