import math
import numbers
import os
import time
import numpy as np
import torch
import random
import argparse

from collections import deque
from imagen_pytorch import Unet, Imagen, ImagenTrainer, ElucidatedImagen
from imagen_pytorch.configs import ElucidatedImagenConfig
from imagen_pytorch.imagen_pytorch import BaseUnet64, SRUnet256, SRUnet1024
from imagen_pytorch.data import get_images_dataloader
from data_generator import ImageLabelDataset
from gan_utils import get_images, get_vocab
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--texts', type=str, default='상표유형은 심볼마크 이다.', help="sample_texts")
parser.add_argument('--scales', type=int, default=256, help="image_scales")
parser.add_argument('--batch', type=int, default=2, help="batch_size")
parser.add_argument('--max_batch', type=int, default=4, help="max_batch_size")
parser.add_argument('--source', type=str, default='/workspace/data99-1/train/01.symbol/', help="image source")
parser.add_argument('--network', type=str, default='/workspace/data99-1/models/train_unet.pth', help="model_file_path")
parser.add_argument('--iter', type=int, default=200000, help="iter_size")


scales = [scales]
batch_size = batch
max_batch_size = max_batch
unet_to_train = 1
shuffle = True
drop_tags=0.75
image_size = scales[unet_to_train-1]

# unets for unconditional imagen
source = source
network = network
imgs = get_images(source, verify=False)
txts = get_images(source, exts=".txt")

def txt_xforms(txt):  
    # print(f"txt: {txt}")
    txt = txt.split(", ")
    if False:
        np.random.shuffle(txt)

    txt = ", ".join(txt)

    return txt

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class PadImage(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return VTF.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)

unet = Unet(
    dim = 256,
    cond_dim = 512,
    dim_mults = (1, 2, 3, 4),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True),
    cond_on_text = True,
    memory_efficient = False
)

unets = unet

imagen = ElucidatedImagen(
    unets = unets,  
    image_sizes = scales,
    cond_drop_prob = 0.1,
    num_sample_steps = 64,
    sigma_min = 0.002,
    sigma_max = 80,
    sigma_data = 0.5,
    rho = 7,
    P_mean = -1.2,
    P_std = 1.2,
    S_churn = 80,
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
    auto_normalize_img = True,
).cuda()

print('Trainer')

#trainer = ImagenTrainer(imagen, fp16=True, dl_tuple_output_keywords_names=('images', 'texts'), split_valid_from_train=True).cuda()
trainer = ImagenTrainer(
    imagen,
    dl_tuple_output_keywords_names = ('images', 'texts'),
    split_valid_from_train = True,
    use_ema = True,
    #only_train_unet_number = unet_to_train
).cuda()

tforms = transforms.Compose([
        PadImage(),
        transforms.Resize((image_size, image_size)),
        transforms.Rsample_textsandomHorizontalFlip(),
        transforms.ToTensor()])
print('Data')

data = ImageLabelDataset(imgs, txts, None,
                        #poses=None,
                        dim=(image_size, image_size),
                        transform=tforms,
                        tag_transform=txt_xforms,
                        channels_first=True,
                        return_raw_txt=True,
                        no_preload=False)
dl = torch.utils.data.DataLoader(data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True)

trainer.add_train_dataloader(dl)
print('Network')
trainer.load(network, noop_if_not_exist = True)

sample_texts=[texts]


rate = deque([1], maxlen=5)
# working training loop
print('Scale: {} | Unet: {}'.format(image_size, unet_to_train))
for i in range(iter): #200000
    t1 = time.monotonic()
    loss = trainer.train_step(unet_number = unet_to_train, max_batch_size = max_batch_size)
    t2 = time.monotonic()
    rate.append(round(1.0 / (t2 - t1), 2))
    #wandb.log({"loss": loss, "step": i})
    print(f'loss: {loss} | Step: {i} | Rate: {round(np.mean(rate), 2)}')

    if not (i % 50) and False:
        valid_loss = trainer.valid_step(unet_number = unet_to_train, max_batch_size = max_batch_size)
        #wandb.log({"Validated loss": valid_loss, "step": i})
        print(f'valid loss: {valid_loss}')

    if not (i % 500) and trainer.is_main: # is_main makes sure this can run in distributed
        rng_state = torch.get_rng_state()
        torch.manual_seed(1)
        images = trainer.sample(batch_size = 4, return_pil_images = False, texts=sample_texts, stop_at_unet_number=unet_to_train, return_all_unet_outputs=True) # returns List[Image]
        torch.set_rng_state(rng_state)
        sample_images0 = transforms.Resize(image_size)(images[0])
        sample_images1 = transforms.Resize(image_size)(images[-1])
        sample_images = torch.cat([sample_images0, sample_images1])
        grid = make_grid(sample_images, nrow=4, normalize=False, range=(-1, 1))
        VTF.to_pil_image(grid).save(f'/workspace/data99-1/unet_outputs/unet-{i // 100}.png')
        print('SAVING NETWORK')
        trainer.save(network)
        print('Network Saved')
        images = wandb.Image(VTF.to_pil_image(grid), caption="Top: Unet1, Bottom: Unet{}".format(unet_to_train))
        #wandb.log({ "step": i, "outputs": images})
