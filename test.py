import jittor as jt
import jittor.transform as transform
import os
from tqdm import tqdm
from datasets import *
from options.test_options import TestOptions
from pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from collections import OrderedDict

jt.flags.use_cuda = 1

test_opt = TestOptions().parse()
model = Pix2PixModel(test_opt)
model.eval()

visualizer = Visualizer(test_opt)

# Configure dataloaders
transforms_label = [
    transform.Resize(size=(384, 512), mode=Image.NEAREST),
    transform.ToTensor(),
]

transforms_img = [
    transform.Resize(size=(384, 512), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

dataloader = ImageDataset(test_opt.data_path, mode="val", transforms_label=transforms_label, transforms_img = transforms_img).set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=1,
)

# create a webpage that summarizes the all results
web_dir = os.path.join(test_opt.results_dir, test_opt.name,
                       '%s_%s' % (test_opt.phase, test_opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (test_opt.name, test_opt.phase, test_opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    generated = model(data_i, mode='inference')
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()