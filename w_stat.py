import os
import argparse
from tqdm.auto import tqdm
from glob import glob 
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import Encoder
from custom.utils import *

from custom.utils import load_obj , save_obj, mkdir

device = "cpu" 

def make_array(tensor):
    return(
        tensor.clone()
            .detach()
            .to('cpu')
            .numpy()
    )

def save_w(args):
    class BrainDataset(Dataset):
        def __init__(self, data_path, transform=None, reverse = True):
            self.transform = transform
            self.reverse  = reverse
            self.patients = glob(os.path.join(data_path, "*"))

            print(f"n = {len(self.patients)} patients")
 
        def __len__(self):
            return len(self.patients)

        def __getitem__(self, idx):
            patient = self.patients[idx]
            pngs = sorted(glob(os.path.join(patient, "*.png")), reverse = self.reverse)[:32]
            return torch.stack([self.transform(Image.open(png)) for png in pngs])

    w_stat_path = os.path.join(args.path, "w_stat")
    mkdir(w_stat_path)

    "set device"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    "load encoder"
    encoder = Encoder(w_plus = True)

    ckpt_path = glob(os.path.join(args.path, "*.pth"))[0]
    ckpt = torch.load(ckpt_path)
    encoder.load_state_dict(ckpt["e"])

    "load model to device"
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"Let's use {n_gpu}GPUs!")
        encoder = torch.nn.DataParallel(encoder)
 
    encoder = encoder.to(device)

    "set dataset"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # scale by 1/255 and make the format to tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # set img range [-1,1]
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.RandomRotation(args.degree)
                ]), p = 0.5),
        ]
    )
    
    brain_dataset = BrainDataset(data_path = args.data_path, transform = transform, reverse = True)
    brain_loader = DataLoader(
                            brain_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            shuffle = False,
                            drop_last=True,
                                )

    "set model to eval mode"
    encoder.eval()

    with torch.no_grad():
        for iter in range(args.iters):
            latents  = {i : [] for i in range(32)}
          
            pbar = tqdm(brain_loader)
            for imgs in pbar:
                imgs = imgs.to(device).reshape([-1, 3, 512, 512])
                for i, latent in enumerate(encoder(imgs)):
                    latents[i].append(latent.detach().to('cpu'))    

            for i, latent in latents.items():
                "set dir"
                pkl_dir = os.path.join(w_stat_path, str(i).zfill(2))
                mkdir(pkl_dir)
        
                latent = torch.cat(latent)
                torch.save(latent, os.path.join(pkl_dir, f"{str(iter).zfill(5)}.pkl"))
    return 0

def load_latents(fNums, latent_dirs):
    latents = []
    for fNum, latent_dir in enumerate(latent_dirs):
        if fNum in fNums:
            pkls = glob(os.path.join(latent_dir, "*.pkl"))
            for pkl in pkls:
                latents.append(torch.load(pkl))
    return torch.cat(latents)

def latent_statistics(args):
    w_stat_path = os.path.join(args.path, "w_stat")
    history_path = os.path.join(w_stat_path, args.pkl)
    history = {}

    latent_dirs = sorted(glob(os.path.join(w_stat_path, "*")))

    fNums = [[fNum + j for j in [-1,0,1]] for fNum in range(32)]
    
    layer_dim = 16
    latent_dim = 512

    n_component = latent_dim
    pca_components = torch.zeros([len(fNums), layer_dim, latent_dim, n_component]).to(device)
    pca_explained_variance = torch.zeros([len(fNums), layer_dim, n_component]).to(device)
    
    with torch.no_grad():
        for fNum, fNums_ in tqdm(enumerate(fNums)):
            latents = load_latents(fNums_, latent_dirs).to(device)
            latents = F.leaky_relu(latents, negative_slope=5, inplace=True)
        
            latent_mean = latents.mean(0,keepdim=True) 
            latent_std  = latents.std(0,keepdim=True) 

            mean = torch.cat((mean,latent_mean)) if fNum else latent_mean
            std  = torch.cat((std,latent_std)) if fNum else latent_std

            for layer in range(layer_dim):
                layer_latents = latents[:, layer, :]

                centered_layer_latents = layer_latents - layer_latents.mean(0)

                U, S, V = torch.pca_lowrank(centered_layer_latents, q = n_component)
                projected_latents = U*S
                explained_variance = projected_latents.var(0)

                pca_components[fNum][layer] = V # [latent_dim, n_component]
                pca_explained_variance[fNum][layer] = explained_variance

    history = {
                "pca_components": pca_components.clone().cpu(),
                "pca_explained_variance": pca_explained_variance.clone().cpu(),
                "mean": mean.clone().cpu(),
                "std" : std.clone().cpu()                
               }

    return save_obj(history, history_path)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='w stat',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=128) # for save_w
    parser.add_argument('--num_workers', type=int, default=12) # for save_w
    parser.add_argument('--size', type=int, default=512, help=" ")
    
    parser.add_argument('--data_path', type = str, required = True)
    parser.add_argument('--path', type = str, required = True)
    parser.add_argument('--iters', type = int, default = 5)
    parser.add_argument('--degree', type = float, default = 30)

    # latent statistics
    parser.add_argument("--save_w", action = "store_true")
    parser.add_argument("--latent_statistics", action = "store_true")

    args = parser.parse_args()

    """
    Usage:
        save_w:             python3 w_stat.py --save_w --path checkpoint/E/
        latent_statistics:  python3 w_stat.py --latent_statistics --path checkpoint/E/
    """

    args.pkl = "latent_statistics.pkl"
        
    if args.save_w: save_w(args)
    if args.latent_statistics: latent_statistics(args)