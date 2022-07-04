import numpy as np
import argparse
import math
import os
import random
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from torchvision import transforms

import lpips
from model import Generator, Encoder

from custom.utils import *
from custom.dataset import BrainCT


device = 'cuda' if torch.cuda.is_available() else 'cpu'


random_seed = 999
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.benchmark = True # cudnn finds the best algorithm to use for your hardware.
np.random.seed(random_seed)
random.seed(random_seed)

def init_noise(g_ema_module, b, requires_grad = False):
    return [noise.repeat(b, 1, 1, 1).normal_().requires_grad_(requires_grad) for noise in g_ema_module.make_noise()]

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def in_domain_loss_(latent, latent_e):
    mask = torch.abs(latent) - torch.abs(latent_e) > 0
    return ((F.mse_loss(latent, latent_e, reduction = "none")) * mask).mean()
    

def w2p(w, w_mean, pca_components, pca_stds):
    w_c     = F.leaky_relu(w, 5) - w_mean
    p       = w_c.unsqueeze(2) @ pca_components
    p_norm  = p / pca_stds
    return p_norm.squeeze(2)


def p2w(p, w_mean, pca_components, pca_stds):
    
    return F.leaky_relu(w_mean + ((p.unsqueeze(2) * pca_stds) @ torch.transpose(pca_components, 2, 3)).squeeze(2), 
                        negative_slope = 0.2)
    

def my_args():
    parser = argparse.ArgumentParser(description='Projection', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--query_save_path', type=str, required=True)
    parser.add_argument('--save_dir',       type=str, default="test_results")    
    parser.add_argument('--ckpt', type = str, default = "checkpoint.pth")   
    parser.add_argument('--latent_dim', type=int,   default=512)
    parser.add_argument('--lr_w',       type=float, default=0.1)
    parser.add_argument('--lr_n',       type=float, default=2)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--num_workers',type=int,   default=12)
    parser.add_argument('--step_w',     type=int,   default=100)
    parser.add_argument('--step_n',     type=int,   default=100)

    parser.add_argument('--filter_size',type=int,   default=19)
    parser.add_argument('--thres',      type=int,   default=5)
    
    return parser.parse_args()



def project(args):
    "set model"
    g_ema = Generator(size=512, style_dim=512, n_mlp=8)
    encoder = Encoder(w_plus = True)
    
    "set pretrained weights"
    ckpt = torch.load(args.ckpt)
    g_ema.load_state_dict(ckpt['g_ema'])
    encoder.load_state_dict(ckpt['e'])

    latent_statistics = load_obj("latent_statistics.pkl")

    """Set model to Device"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_gpu = torch.cuda.device_count()
    is_distributed = args.n_gpu > 1
    if is_distributed:
        g_ema = torch.nn.DataParallel(g_ema)
        encoder = torch.nn.DataParallel(encoder)

    g_ema = g_ema.to(device)
    encoder = encoder.to(device)

    "set eval mode"
    g_ema.eval()
    encoder.eval()

    g_ema_module = g_ema.module if torch.cuda.device_count() > 1 else g_ema

    "lpips loss"
    percept = lpips.PerceptualLoss(model='net-lin', net="vgg", use_gpu=device.startswith(device))
    
    args.step = args.step_w + args.step_n

    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),  # scale by 1/255 and make the format to tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # set img range [-1,1]
        ]
    )

    """Set directories"""
    mkdir('results')
    result_dir = os.path.join('results', args.save_dir)
    mkdir(result_dir)

    print("-" * 20)
    print(args.save_dir)

    query_save_dirs = glob(os.path.join(args.query_save_path, "*"))
    for query_save_dir in query_save_dirs:
        patient_id = query_save_dir.split('/')[-1]
        inference_dir = os.path.join(result_dir, patient_id)
        mkdir(inference_dir)
        print(f"[INFO] query images = {query_save_dir}")

        """load data"""
        brainct_dataset = BrainCT(query_save_dir, transform, reverse = True)
        brainct_dataloader = DataLoader(brainct_dataset, 
                                        batch_size  = args.batch_size,
                                        shuffle     = False,
                                        num_workers = args.num_workers,
                                        drop_last   = False,
                                        pin_memory  = True) 

        n_imgs     = len(brainct_dataset)
        n_latent   = g_ema_module.n_latent

        history =  {"scores"  : np.zeros([args.step, n_imgs])}
        for fNums, fNames, img_reals, bet_masks in brainct_dataloader:        
            img_reals = img_reals.to(device, non_blocking=True)
            bet_masks = bet_masks.bool().to(device, non_blocking=True)
            b,c,h,w   = img_reals.shape

            bet_normalizer = np.array([1 / area.item() if area.item() else 0 for area in bet_masks.reshape(b,-1).float().sum(1)])
            voting_maps = torch.zeros_like(bet_masks)
            pred_maps   = torch.zeros_like(bet_masks)
            
            w_means        = latent_statistics["mean"][fNums].float().to(device, non_blocking=True) # [b, n_latent, latent_dim]
            pca_stds       = latent_statistics["pca_explained_variance"][fNums].float().to(device, non_blocking=True).unsqueeze(2) ** 0.5
            pca_components = latent_statistics["pca_components"][fNums].float().to(device, non_blocking=True)

            "SET LATENT CODES"
            with torch.no_grad():
                "set initial latent by encoder"
                latent_e = encoder(img_reals)
                if latent_e.shape == [b, 512]:
                    latent_e = latent_e.unsqueeze(1).repeat(1, n_latent, 1)
                else:
                    latent_e = latent_e.reshape(b, n_latent, 512)

                latent = w2p(latent_e, w_means, pca_components, pca_stds)

                img_targets = img_reals

            "load filter"
            filter = MedianPool2d(args.filter_size)
            filter = torch.nn.DataParallel(filter).to(device) if args.n_gpu > 1 else filter.to(device)
            filter.eval()

            "set optimizer for latent"
            latent.requires_grad_(True)
            optim_w = optim.Adam([latent], lr = args.lr_w)

            "loss fuctions"            
            pbar = tqdm(range(args.step))
            for step in pbar:
                if step < args.step_w: 
                    "update learning rate"
                    t = step / args.step_w
                    lr_w = get_lr(t, args.lr_w)
                    optim_w.param_groups[0]["lr"] = lr_w

                    img_fakes, _ = g_ema([p2w(latent, w_means, pca_components, pca_stds)], 
                                            input_is_latent=True,)

                    "forward"
                    p_loss   = percept(to256(img_targets), to256(img_fakes)).reshape(b,-1).mean(1)
                    with torch.no_grad():
                        latent_e = encoder(img_reals)
                        if latent_e.shape == [b, 512]:
                            latent_e = latent_e.unsqueeze(1).repeat(1, n_latent, 1)
                        else:
                            latent_e = latent_e.reshape(b, n_latent, 512)

                        latent_e = w2p(latent_e, w_means, pca_components, pca_stds)

                    in_domain_loss = in_domain_loss_(latent, latent_e)

                    loss_w = (p_loss + in_domain_loss).mean()

                    "backward"
                    optim_w.zero_grad(set_to_none=True)
                    loss_w.backward()
                    optim_w.step()

                    description = "[w]"
                    description += f'p_loss: {p_loss.mean().item():.4f};'
                    description += f'in_domain_loss: {in_domain_loss.mean().item():.4f};'
                    description += f"lr = {lr_w:.5f}"
                    pbar.set_description(description)

                else:
                    if step == args.step_w:
                        latent_in = latent.detach().clone().requires_grad_(False)
                        latent_in = p2w(latent_in, w_means, pca_components, pca_stds)
                        noises    = init_noise(g_ema_module, b, requires_grad=True)
                        
                        img_refs  = img_fakes.detach().clone()
                        
                        optim_n = optim.Adam(noises, lr = args.lr_n)

                    t = (step - args.step_w) / args.step_n
                    lr_n = get_lr(t, args.lr_n)
                    optim_n.param_groups[0]["lr"] = lr_n

                    img_fakes, _ = g_ema([latent_in], 
                                          input_is_latent = True, 
                                          noise=noises)

                    with torch.no_grad():
                        ref_mask = pred_maps.unsqueeze(1)
                        img_targets = img_reals * (~ref_mask) + img_refs * ref_mask

                    "forward"
                    loss_n = F.l1_loss(img_targets, img_fakes, reduction="mean")
                    
                    "backward"
                    optim_n.zero_grad(set_to_none=True)
                    loss_n.backward()
                    optim_n.step()

                    description = f"[n] loss: {loss_n.item():.4f}; lr = {lr_n:.5f}"
                    pbar.set_description(description)


                with torch.no_grad():
                    img_reals_hu = normalize_hu(img_reals)
                    img_fakes_hu = normalize_hu(img_fakes)

                    diff_maps = (img_reals_hu - img_fakes_hu) * bet_masks
                    diff_maps_filtered = torch.cat([filter(diff_maps[args.n_gpu * i: args.n_gpu * (i + 1)].unsqueeze(0)).squeeze(0) 
                                                    for i in range(math.ceil(b / args.n_gpu))]) * bet_masks
                    residual_maps_filtered = torch.abs(diff_maps_filtered)
                    voting_maps = voting(residual_maps_filtered, voting_maps, args.thres)
                    pred_maps   = voting_maps > step

                    voting_scores = tensor_to_np((voting_maps * residual_maps_filtered).reshape(b,-1).sum(1))
                    history["scores"][step][fNums] = voting_scores * bet_normalizer 
                        
            "save figure"
            img_reals_np = make_image(img_reals)
            img_fakes, _ = g_ema([latent_in], input_is_latent = True)
            img_fakes_np = make_image(img_fakes)
            pred_maps_np = tensor_to_np(pred_maps) 

            figure_save_dir = os.path.join(inference_dir, "figure")    
            for b_ in range(b):
                "save individual figure"
                save_summary_figure(img_reals_np[b_], 
                                    img_fakes_np[b_],
                                    pred_maps_np[b_], 
                                    figure_save_dir, 
                                    fNames[b_])

        history_dir = os.path.join(inference_dir, 'history')
        mkdirs(history_dir) 
        save_obj(history, os.path.join(history_dir,'history.pkl'))



if __name__ == '__main__':
    args = my_args()
    project(args)