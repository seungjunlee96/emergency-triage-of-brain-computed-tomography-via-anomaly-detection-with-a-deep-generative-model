"""utility functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pydicom
import pandas as pd
import pickle5 as pickle
import torch
import matplotlib.pyplot as plt
from PIL import Image

DPI = 300
CLUSTER_SIZE = 600
# ----------------------------------------------------------------------------
# Image utils.
def normalize_hu(img):
    """
    Normalize the img to represent HU value
    """
    def in_range(img, min = None, max = None):
        if min == None:
            return img * (img < max)
        if max == None:
            return img * (img > min)
        else:
            return img * (img >= min) * (img <= max)
    b,c,h,w = img.shape
    sample = img.clone()
    sample = torch.clamp(sample, -1, 1)

    sample_0 = 50 *  (sample[:, 0] + 1) / 2 + 5
    sample_1 = 80 *  (sample[:, 1] + 1) / 2 
    sample_2 = 200 * (sample[:, 2] + 1) / 2 - 20

    sample = (in_range(sample_2,  max = 0) + 
             (in_range(sample_1,   0,   5) + in_range(sample_2, 0, 5)) / 2 +
             (in_range(sample_0,   5,  55) + in_range(sample_1,   5,  55) + in_range(sample_2, 5 , 55)) / 3 +
             (in_range(sample_1,  55,  80) + in_range(sample_2,  55,  80)) / 2 +
              in_range(sample_2, min = 80))

    return sample


def windowing_brain(npy, channel=3):
    dcm = npy.copy()
    img_rows = 512
    img_cols = 512

    if channel == 1:
        npy = npy.squeeze()
        npy = cv2.resize(npy, (512, 512), interpolation=cv2.INTER_LINEAR)
        npy = npy + 40
        npy = np.clip(npy, 0, 160)
        npy = npy / 160
        npy = 255 * npy
        npy = npy.astype(np.uint8)

    elif channel == 3:
        dcm0 = dcm[0] - 5
        dcm0 = np.clip(dcm0, 0, 50)
        dcm0 = dcm0 / 50.
        dcm0 *= (2 ** 8 - 1)
        dcm0 = dcm0.astype(np.uint8)

        dcm1 = dcm[0] + 0
        dcm1 = np.clip(dcm1, 0, 80)
        dcm1 = dcm1 / 80.
        dcm1 *= (2 ** 8 - 1)
        dcm1 = dcm1.astype(np.uint8)

        dcm2 = dcm[0] + 20
        dcm2 = np.clip(dcm2, 0, 200)
        dcm2 = dcm2 / 200.
        dcm2 *= (2 ** 8 - 1)
        dcm2 = dcm2.astype(np.uint8)



        npy = np.zeros([img_rows, img_cols, 3], dtype=int)
        npy[:, :, 0] = dcm0 #  5 55
        npy[:, :, 1] = dcm1 #  0 80
        npy[:, :, 2] = dcm2 # -20 180

    return np.uint8(npy)


def write_png_image(img_png, npy):
    if not os.path.exists(img_png):
        return cv2.imwrite(img_png, npy)
    else:
        return False


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def convert_to_numpy_array(image, drange=[0, 1], rgbtogray=False):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]  # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0)  # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0, 255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)

    if rgbtogray:
        return convert_rgb_to_gray(image)

    return image

def convert_to_pil_image(image, drange=[0, 1]):
    image = convert_to_numpy_array(image, drange)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return Image.fromarray(image, fmt)

def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def tensor_to_np(tensor):
    return (
        tensor.detach()
              .to('cpu')
              .numpy()
    )

def make_image(tensor):
    return (
        tensor.clone()
            .detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute( 0, 2 , 3 , 1 )
            .to('cpu')
            .numpy()
            .astype(np.uint8)
    )


def convert_rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_mirror_augment(minibatch):
    mask = np.random.rand(minibatch.shape[0]) < 0.5
    minibatch = np.array(minibatch)
    minibatch[mask] = minibatch[mask, :, :, ::-1]
    return minibatch

# ----------------------------------------------------------------------------


def create_summary_figure(real, fake, target, pred):
    real_gray = rgb2gray(real)
    fake_gray = rgb2gray(fake)

    figures = 4
    figure_size = 6
    # plt.rcParams["font.family"] = "Times New Roman"
    fontsize = 24
    fig = plt.figure(figsize=(figure_size * figures, figure_size))

    """First Figure : Real """
    fig.add_subplot(1, figures, 1)
    plt.title('Real', fontsize=fontsize)
    plt.axis('off')
    plt.imshow(real_gray, cmap="gray")  # transpose(real))

    """Second Figure : Input + Target """
    fig.add_subplot(1, figures, 2)
    plt.axis('off')
    plt.title("Target", fontsize=fontsize)
    plt.imshow(real_gray, cmap="gray")

    # target = target.astype(np.int) * 255
    plt.imshow(colorize_mask(target), alpha = 0.75)

    """Third Figure : Fake """
    fig.add_subplot(1, figures, 3)
    plt.title('Fake', fontsize=fontsize)
    plt.axis('off')
    plt.imshow(fake_gray, cmap="gray")

    """4th : Prediction for lesion"""
    fig.add_subplot(1, figures, 4)
    plt.axis('off')
    plt.title("Prediction", fontsize=fontsize)
    plt.imshow(real_gray, cmap="gray")

    plt.imshow(pred, alpha = 0.75)
    fig.tight_layout()

    return fig

def load_state_dict(model, pretrained_dict):
    model_state_dict = model.state_dict()    
    for name, param in model_state_dict.items():
        if name not in model_state_dict:
            pretrained_dict[name] = param
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    return model

def save_summary_figure(real, fake, overlay, figure_save_dir, fName, save_format = 'png'):
    # real_gray = real
    # fake_gray = fake
    residual  = real - fake
    fname = fName.replace('.dcm', '.png')

    mkdirs(os.path.join(figure_save_dir, 'input',))
    input_path = os.path.join(figure_save_dir, "input", fname)
    array2image(rgb2gray(real), input_path)
    
    mkdirs(os.path.join(figure_save_dir, 'reconstruction'))   
    recon_path = os.path.join(figure_save_dir, "reconstruction", fname)
    array2image(rgb2gray(fake), recon_path)

    """4th : Prediction for lesion"""
    overlay_path   = os.path.join(figure_save_dir, "overlay", fname)
    mkdirs(os.path.join(figure_save_dir, 'overlay'))   
    array2image(overlay, path = overlay_path,) 

    input_overlaid_path = os.path.join(figure_save_dir, "input_overlaid", fname)
    mkdirs(os.path.join(figure_save_dir, 'input_overlaid'))

    overlay = cv2.imread(overlay_path, 0)
    _, img = cv2.threshold(overlay, 127, 255, cv2.THRESH_BINARY)  # ensure binary
    num_labels, labels = cv2.connectedComponents(img, connectivity=8)
    
    red_overlay    = np.zeros(labels.shape)
    yellow_overlay = np.zeros(labels.shape)
    for label in range(1, np.max(labels) + 1):
        label_cluster_size = np.sum(labels == label)
        labels[labels == label] = 255 * (label_cluster_size > CLUSTER_SIZE)
        if (residual[:,:,2] * (labels == label)).sum():
            red_overlay[labels == label]    = 255 * (label_cluster_size > CLUSTER_SIZE)
            yellow_overlay[labels == label] = 0
        else:
            red_overlay[labels == label]    = 0
            yellow_overlay[labels == label] = 255 * (label_cluster_size > CLUSTER_SIZE)

    labels = labels.astype(np.uint8)
    overlay = np.stack([labels, labels * 0, labels * 0], 2)

    array2image(overlay, path = overlay_path, colorize = False, transparent=True)
    transparent_blend(input_path, overlay_path, input_overlaid_path)

def minmax_normalization(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def artifact_mask(dcm, threshold = 800):
    mask = dcm > threshold
    return mask

class FigureGenerator():
    def __init__(self):
        pass

def colorize_mask(mask, color = "Red", masked = True):
    assert mask.shape == (512, 512), print(mask.shape)
    mask = (255 * mask).astype(np.uint8)
    if color == "Yellow":
        mask = np.stack([mask * 1, mask * 1, mask * 0], 2) 
    if color == "White":
        mask = np.stack([mask * 1, mask * 1, mask * 1], 2) 
    if color == "Red":
        mask = np.stack([mask * 1, mask * 0, mask * 0], 2) 

    return mask

def transparent_blend(im1_path, im2_path, save_path, alpha = 0.66):
    background = Image.open(im1_path).convert('RGB')
    foreground = Image.open(im2_path).convert('RGBA')
    foreground_trans = Image.new("RGBA", foreground.size)
    foreground_trans = Image.blend(foreground_trans, foreground, alpha)

    background.paste(foreground_trans, (0, 0), foreground_trans)
    background.save(save_path, dpi = (DPI, DPI))

def transparent_mask(pil):
    rgba = pil.convert("RGBA")
    rgba_data = rgba.getdata()

    newData = []
    for item in rgba_data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value
            # storing a transparent value when we find a black colour
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)  # other colours remain unchanged
    
    rgba.putdata(newData)
    return rgba

def diff_score(img_reals, img_fakes, bet_masks):
    b,c,h,w = img_reals.shape
    diff = torch.abs(img_reals - img_fakes) * bet_masks.unsqueeze(1) # simple residual difference
    """
    _,diff = ssim(real_tmp, fake_tmp, full = True)
    
    about SSIM: 
        -   https://bskyvision.com/396
        -   https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
        -   Don't know why but it is worse than residual difference..
            -   SSIM is for "structural" similarity, SSIM considers white matter and gray matter as "noise"
    """
    return diff.reshape(b,-1).mean(1)

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    https://gist.github.com/keunwoochoi/dcbaf3eaa72ca22ea4866bd5e458e32c
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=7, stride=1, padding=0, same = True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        b,c,w,h = x.shape
        x = F.pad(x, self._padding(x))
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def voting(diff_maps, voting_maps, thres):
    votes  = (diff_maps > thres).float()

    adder = torch.zeros_like(votes)
    for fNum in range(votes.size(0)):
        if fNum != 0 and (fNum+1) != votes.size(0):
            adder[fNum] += ((votes[fNum-1] * votes[fNum]) 
                        +   (votes[fNum+1] * votes[fNum])
                        +   (votes[fNum-1] * votes[fNum+1]))
    votes += adder.bool().float()

    return voting_maps + votes         


def mask_cluster(masks, threshold):
    masks_np = masks.clone().detach().cpu().numpy()
    for i in range(len(masks_np)):
        mask = masks_np[i]
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)  # ensure binary
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for label in range(1, np.max(labels) + 1):
            label_cluster_size = np.sum(labels == label)
            labels[labels == label] = (label_cluster_size > CLUSTER_SIZE)
        masks_np[i] = labels
        with torch.no_grad():
            masks[i] = torch.from_numpy(masks_np[i])
    return masks.bool()

def to256(img):
    batch, channel, height, width = img.shape
    img_256 = img.clone()

    if height > 256:
        factor = height // 256

        img_256 = img_256.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img_256 = img_256.mean([3, 5])

    return img_256


def shrink(img, size):
    batch, channel, height, width = img.shape
    img_shrink = img.clone()

    if height > size:
        factor = height // size

        img_shrink = img_shrink.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img_shrink = img_shrink.mean([3, 5])

    return img_shrink


def save_obj(obj, PATH):
    with open(PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(PATH):
    import pickle5 as pickle
    with open(PATH, 'rb') as f:
        return pickle.load(f)


def mkdir(*args):
    if not os.path.exists(*args):
        os.mkdir(*args)

def mkdirs(path):
    assert path[0] != '/' 
    for i in range(len(path.split('/'))):
        mkdir(os.path.join(*path.split('/')[:i+1]))
        

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_path = ""):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save_path:
        plt.savefig(save_path, dpi = DPI, transparent = True)
    plt.show()
    plt.close()

def array2image(arr, path = "", colorize=False, transparent=False):
    if colorize:
        arr = colorize_mask(arr)

    im = Image.fromarray(arr)

    if transparent:
        im = transparent_mask(im)

    if path:
        im.save(path, dpi = (DPI, DPI))
 
    return im

def show_tensor_images(image_tensor):
    n, c, h, w = image_tensor.shape
    image_tensor = image_tensor + 1 / 2
    image = image_tensor.detach().cpu()
    image_grid = make_grid(image, nrow = int(n ** 0.5))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    return plt.show()


def make_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index = False)  


def crop_center(img, cropx = 512, cropy = 512):
    _, y, x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[:, starty:starty+cropy, startx:startx+cropx]

def dcm2img(dcm, windowing = True):
    dcm_data = pydicom.dcmread(dcm)
    if hasattr(dcm_data, "pixel_array"):
        img = dcm_data.pixel_array.astype(np.int32) - 1024
    else: 
        return None
    img = crop_center(np.expand_dims(img, 0))
    
    if windowing: return windowing_brain(img)
    else: return img

def dcm2np(dcm, windowing = True):
    img = sitk.GetArrayFromImage(sitk.ReadImage(dcm))
    img = crop_center(img)
    if windowing: return windowing_brain(img)
    else: return img

def png2arr(png, rgb = True):
    np_frame = np.array(Image.open(png))
    if rgb:
        return np.uint8(np_frame)
    else:
        return rgb2gray(np.uint8(np_frame))
