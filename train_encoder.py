import sys
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import math
import os
from glob import glob
import random
from tqdm.auto import tqdm

import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
from torchvision import transforms, utils
import lpips

from model import Generator, Discriminator, Encoder
from custom.utils import mkdir, mkdirs, data_sampler, to256
from custom.torch_utils import load_state_dict
from custom.dataset import RealDataset

from distributed import (
    get_rank,
    synchronize,
)

# python3 -m torch.distributed.launch --nproc_per_node=<n_gpus> --master_port=8888 train_encoder.py --data_path=<data_path>

device = "cuda" if torch.cuda.is_available() else "cpu"

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        inputs=real_img,
        outputs=real_pred.sum(),  
        create_graph=True,
        retain_graph=True,
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()
    
def set_encoder(args):
    encoder = Encoder(w_plus = args.w_plus)
    args.resize = 512
    return encoder, args

def random_rotate(img, args, p = 0.5):
    # img: tensor value in [-1, 1]
    # angle: rotate angle value in degrees, counter-clockwise
    if random.random() > p:
        return img
    else:
        angle = args.max_angle * 2 * (random.random() - 0.5)
        return torchvision.transforms.functional.rotate(img + 1.0, angle) - 1.0

def train(args, g_ema, encoder, optim_E, optim_D):      
    "set transforms"
    my_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.RandomHorizontalFlip(),

            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
            ]
        )
    }

    random_erasing = transforms.Compose([transforms.RandomErasing(value=-1, p=0.25)])

    if args.is_distributed:
        g_ema_module = g_ema.module
        encoder_module = encoder.module
        discriminator_module = discriminator.module
    else:
        g_ema_module = g_ema
        encoder_module = encoder
        discriminator_module = discriminator

    percept = lpips.PerceptualLoss(model='net-lin', 
                                    net='vgg',
                                    gpu_ids = [torch.cuda.current_device()], 
                                    use_gpu=True if torch.cuda.is_available() else False)

    "Set Directories"
    save_dir = os.path.join("checkpoint/E/train_encoder")
    if get_rank() == 0:
        mkdirs(save_dir)
        print("-" * 30)
        print("INFO. save logs at:", save_dir)
        print("-" * 30)

    start_epoch = ckpt.get("epoch", 0)
    for epoch in range(start_epoch, args.epochs + 1):
        "load dataset"
        phases = ["train", "valid"]
        real_dataset = {phase:RealDataset(data_path = args.data_path,
                                          split     = phase, 
                                          transform = my_transforms[phase], 
                                          ) for phase in phases}                             

        real_dataloader = {phase: DataLoader(real_dataset[phase],
                                            batch_size  = args.batch_size,
                                            sampler     = data_sampler(real_dataset[phase], 
                                                                       shuffle = (phase == "train"), 
                                                                       distributed=args.is_distributed),
                                            num_workers = args.num_workers,
                                            drop_last   = True,
                                            pin_memory  = True) for phase in phases} 

        if get_rank() == 0:
            epoch_dir = os.path.join(save_dir,str(epoch))
            mkdir(epoch_dir)
            mkdir(os.path.join(epoch_dir, 'image'))

        for phase in phases:
            epoch_p_loss        = 0
            epoch_adv_loss      = 0
            epoch_d_loss        = 0
            epoch_real_pred     = 0
            epoch_fake_pred     = 0
            epoch_in_domain_loss = 0

            p_loss      = torch.tensor([0.], device = device)
            adv_loss    = torch.tensor([0.], device = device)
            d_loss      = torch.tensor([0.], device = device)
            real_pred   = torch.tensor([0.], device = device)
            fake_pred   = torch.tensor([0.], device = device)
            in_domain_loss = torch.tensor([0.], device = device)

            n_sample     = 0
            total_sample = len(real_dataset[phase])

            if phase == "train":
                discriminator.train()
                g_ema.eval()
                encoder.train()

            if phase == "valid":
                discriminator.eval()
                g_ema.eval()
                encoder.eval()

            iterator = iter(real_dataloader[phase])
            pbar = range(len(iterator))
            if get_rank() == 0: pbar = tqdm(range(len(iterator)), dynamic_ncols=True, smoothing=0.01)
            for step in pbar:               
                real_imgs = next(iterator, [])
                if real_imgs == []: break 
                real_imgs = real_imgs.cuda()
                b,c,h,w   = real_imgs.shape
                n_sample += b
        

                if phase == "train":
                    real_imgs = random_rotate(real_imgs, args, p = 0.25)
                    pretrain_mode = args.pretrain_epoch > epoch
                    if pretrain_mode:
                        requires_grad(g_ema, False)
                        requires_grad(encoder, True)
                        requires_grad(discriminator, False)
                    
                        latent = encoder(random_erasing(real_imgs))
                        if args.w_plus: latent = latent.reshape(-1,16,512)
                        proj_imgs, _ = g_ema([latent], input_is_latent=True)

                        p_loss = percept(to256(real_imgs), to256(proj_imgs)).mean()
                        
                        loss = p_loss

                        "backward"
                        encoder.zero_grad()
                        loss.backward()
                        optim_E.step()

                    else:
                        """train D"""  
                        requires_grad(g_ema, False)
                        requires_grad(encoder, False)
                        requires_grad(discriminator, True)

                        "real_img to fake_img: G(E(x))"
                        latent = encoder(real_imgs)
                        if args.w_plus: latent = latent.reshape(-1,16,512)
                        fake_imgs, _ = g_ema([latent], input_is_latent=True)

                        "forward: discriminator"
                        fake_pred = discriminator(fake_imgs) 
                        real_pred = discriminator(real_imgs)

                        d_loss = d_logistic_loss(real_pred, fake_pred)

                        "backward: discriminator"
                        discriminator.zero_grad()
                        d_loss.backward()
                        optim_D.step()

                        "regularization"
                        d_regularize = step % args.d_reg_every == 0
                        if d_regularize:
                            discriminator.zero_grad()

                            real_imgs.requires_grad = True
                            real_preds = discriminator(real_imgs)
                            r1_loss = d_r1_loss(real_preds, real_imgs)

                            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_preds[0]).backward()
                            optim_D.step()

                        "------------update advE-------------"
                        requires_grad(g_ema, False)
                        requires_grad(encoder, True)
                        requires_grad(discriminator, False)

                        "load real samples"
                        real_imgs = next(iterator, [])
                        if real_imgs == []: break 
                        real_imgs = real_imgs.cuda()
                        b,c,h,w   = real_imgs.shape
                        n_sample += b
        
                        real_imgs = random_rotate(real_imgs, args, p = 0.25)

                        # real img -> latent code -> fake img
                        latent = encoder(random_erasing(real_imgs))
                        if args.w_plus: latent = latent.reshape(-1,16,512)
                        fake_imgs, _ = g_ema([latent], input_is_latent=True)
                        
                        p_loss = (percept(to256(real_imgs), to256(fake_imgs))).mean()
                        
                        # re-projection
                        pred_latent = encoder(random_erasing(fake_imgs))
                        if args.w_plus: pred_latent = pred_latent.reshape(-1,16,512)
                        in_domain_loss = F.mse_loss(latent, pred_latent)

                        fake_pred = discriminator(fake_imgs)
                        adv_loss = F.softplus(-fake_pred).mean()
                        
                        loss = (args.perceptual * p_loss 
                             +  args.adv        * adv_loss
                             +  args.in_domain  * in_domain_loss)
                 
                        "backward"
                        encoder.zero_grad()
                        loss.backward()
                        optim_E.step()

                if phase == 'valid':
                    with torch.no_grad():
                        "real -> fake"
                        latent = encoder(real_imgs)
                        if args.w_plus: latent = latent.reshape(-1,16,512)
                        fake_imgs, _ = g_ema([latent], input_is_latent=True)                     

                        "re-projection"
                        pred_latent  = encoder(fake_imgs)
                        if args.w_plus: pred_latent = pred_latent.reshape(-1,16,512)

                        "forward"
                        p_loss = percept(to256(real_imgs), to256(fake_imgs)).mean()
                        real_pred = discriminator(real_imgs)
                        fake_pred = discriminator(fake_imgs)
                        in_domain_loss = F.mse_loss(latent, pred_latent)
                    
                if get_rank() == 0:     
                    epoch_p_loss += p_loss.item() 
                    epoch_adv_loss += adv_loss.item() 
                    epoch_real_pred += real_pred.mean().item()
                    epoch_fake_pred += fake_pred.mean().item()
                    epoch_in_domain_loss += in_domain_loss.item()

                    state  = f"[{phase} No.{epoch}]"
                    state += f" progress = {100 * (n_sample / total_sample) * args.n_gpu:.2f}%"
                    state += f"/ p({epoch_p_loss / (step + 1):.3f})"
                    state += f"/ adv({epoch_adv_loss / (step + 1):.3f})"
                    state += f"/ d({epoch_d_loss / (step + 1):.3f})"                   
                    state += f"/ real({epoch_real_pred / (step + 1):.3f})"     
                    state += f"/ fake({epoch_fake_pred / (step + 1):.3f})"     
                    state += f"/ in_domain({epoch_in_domain_loss / (step + 1):.3f})"     
                    
                    pbar.set_description(state)  
                
        if get_rank() == 0:            
            print("save result at", epoch_dir)
            torch.save({
                "epoch"  : epoch ,
                "e": encoder_module.state_dict(),
                "d": discriminator_module.state_dict(),
                "g_ema": g_ema_module.state_dict(),
                "optim_D": optim_D.state_dict(),
                "optim_E": optim_E.state_dict(),
            }, os.path.join(epoch_dir,f"{str(epoch).zfill(4)}.pth"))

if __name__ == "__main__":
    # args = my_args()
    parser = argparse.ArgumentParser(description='Train Encoder',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8) # batch size 16 for 48GB 
    parser.add_argument('--pretrain_epoch', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--lr_E',   type=float, default=1e-4)
    parser.add_argument('--lr_D',   type=float, default=1e-5)
    parser.add_argument('--ckpt',   type=str, default = "")
    parser.add_argument('--w_plus',    type = bool, default = True)
    parser.add_argument('--perceptual', type=float, default=1)
    parser.add_argument('--in_domain', type = float, default = 1)
    parser.add_argument('--adv', type = float, default = 0.05)
    parser.add_argument('--d_reg_every', type = int, default = 16)
    parser.add_argument('--r1', type = float, default = 10)
    parser.add_argument('--random_erasing', action = "store_true")
    parser.add_argument('--max_angle', type = float, default = 45)
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")

    args = parser.parse_args()

    # set models to GPU(s)
    args.n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.is_distributed = args.n_gpu > 1
    
    if args.is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    "set generator, discriminator and encoder"
    g_ema = Generator(size=512, style_dim=512, n_mlp=8)
    discriminator = Discriminator(size = 512)
    encoder, args = set_encoder(args)

    "set weight checkpoint"
    ckpt = {}
    if args.ckpt:
        if get_rank() == 0: print(f"INFO. load model checkpoint from {args.ckpt}")   
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        g_ema.load_state_dict(ckpt["g_ema"])
        discriminator.load_state_dict(ckpt["d"])
        if "e" in ckpt: encoder = load_state_dict(encoder, ckpt["e"])

    "set models to device"
    encoder = encoder.to(device)
    g_ema = g_ema.to(device)
    discriminator = discriminator.to(device)

    if args.is_distributed:
        g_ema = nn.parallel.DistributedDataParallel(
            g_ema,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,  
            find_unused_parameters=True
        )
  
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
    
    "set optimizer"
    optim_E = Adam(encoder.parameters(), lr = args.lr_E)
    optim_D = Adam(discriminator.parameters(), lr=args.lr_D)
    
    if "optim_E" in ckpt: optim_E.load_state_dict(ckpt["optim_E"])
    if "optim_D" in ckpt: optim_D.load_state_dict(ckpt["optim_D"])
    
    train(args, g_ema, encoder, optim_E, optim_D)

