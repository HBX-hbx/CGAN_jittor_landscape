import jittor as jt
import jittor.transform as transform
import os
import numpy as np
import time
import time
from tqdm import tqdm
from datasets import *
from pix2pix_trainer import Pix2PixTrainer
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from collections import OrderedDict

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

opt = TrainOptions().parse()
print(opt)

os.makedirs(os.path.join(opt.output_path, "saved_models"), exist_ok=True)
writer = SummaryWriter(opt.output_path)

trainer = Pix2PixTrainer(opt)
visualizer = Visualizer(opt)

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

dataloader = ImageDataset(opt.data_path, mode="train", transforms_label=transforms_label, transforms_img = transforms_img).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

prev_time = time.time()
trainer.save('latest')
trainer.save(-1)
for epoch in range(opt.epoch, opt.n_epochs):
    epoch_step = len(dataloader)
    for i, data_i in tqdm(enumerate(dataloader)):
        # change the training to SPADE
        # train generator
        trainer.run_generator_one_step(data_i)
        # train discriminator
        trainer.run_discriminator_one_step(data_i)
        
        # Visualizations
        if i % 1000 == 0:
            visuals = OrderedDict([('input_label', data_i['label']),
                                    ('synthesized_image', trainer.get_latest_generated()),
                                    ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, i)
        
    trainer.update_learning_rate(epoch)
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d' %
              (epoch))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
 



