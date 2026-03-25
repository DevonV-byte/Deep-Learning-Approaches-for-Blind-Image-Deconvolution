from __future__ import print_function
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import cv2
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM
from skimage import img_as_ubyte, img_as_uint
import re

# Print CUDA information
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Device Count:', torch.cuda.device_count())
    print('CUDA Device Name:', torch.cuda.get_device_name(0))

parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[27, 27], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="Datasets/levin/GaussianBlur/total", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/levinNoise/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=1000, help='lfrequency to save results')
parser.add_argument('--output_type', type=int, default=3, choices=[1, 2, 3], 
                    help='1 for clear image, 2 for estimated kernel, 3 for both')
parser.add_argument('--NoiseMode', action='store_true', help='when enabled, processes all noise variations of the specified image')
parser.add_argument('--N_imgs', type=int, default=1, help='specific image to process (0 for all images, 1 for im1, etc.)')
parser.add_argument('--N_kernels', type=int, default=0, help='specific kernel to process (0 for all kernels, 1 for kernel1, etc.)')
parser.add_argument('--Interm', action='store_true', help='when enabled, only processes images ending with img_x.png')
parser.add_argument('--Interm_path', type=str, default="img_x.png", help='path to intermediate results')
opt = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# Filter images based on N_imgs and N_kernels parameters
if opt.N_imgs > 0:
    # Filter for specific image number
    img_pattern = f'im{opt.N_imgs}'
    files_source = [f for f in files_source if img_pattern in os.path.basename(f)]
    
    if opt.NoiseMode:
        # When NoiseMode is enabled, collect all noise variations
        noise_pattern = f'im{opt.N_imgs}_kernel\d+_img_noise\d+\.png$'
        files_source = [f for f in files_source if re.match(noise_pattern, os.path.basename(f))]
    else:
        # Further filter by kernel number if specified
        if opt.N_kernels > 0:
            kernel_pattern = f'im{opt.N_imgs}_kernel{opt.N_kernels}_img_noise\d+\.png$'
            files_source = [f for f in files_source if re.match(kernel_pattern, os.path.basename(f))]
        else:
            # If no specific kernel is specified, get all kernel variations
            kernel_pattern = f'im{opt.N_imgs}_kernel\d+_img_noise\d+\.png$'
            files_source = [f for f in files_source if re.match(kernel_pattern, os.path.basename(f))]

# If Interm flag is enabled, filter only for files ending with 'img.x.png'
if opt.Interm:
    files_source = [f for f in files_source if os.path.basename(f).endswith(opt.Interm_path)]


# start #image
for f in files_source:
    print(f)

    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('kernel1') != -1:
        opt.kernel_size = [19, 19]
    if imgname.find('kernel2') != -1:
        opt.kernel_size = [17, 17]
    if imgname.find('kernel3') != -1:
        opt.kernel_size = [15, 15]
    if imgname.find('kernel4') != -1:
        opt.kernel_size = [27, 27]
    if imgname.find('kernel5') != -1:
        opt.kernel_size = [13, 13]
    if imgname.find('kernel6') != -1:
        opt.kernel_size = [21, 21]
    if imgname.find('kernel7') != -1:
        opt.kernel_size = [23, 23]
    if imgname.find('kernel8') != -1:
        opt.kernel_size = [23, 23]

    _, imgs = get_image(path_to_image, -1) # load image and convert to np.
    y = np_to_torch(imgs).type(dtype)

    img_size = imgs.shape
    print(imgname)
    # ######################################################################
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

    '''
    x_net:
    '''
    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

    net = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

    '''
    k_net:
    '''
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()

    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
    net_kernel = net_kernel.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    # Initialize a list to store the loss values
    loss_history = []

    # start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)

        out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

        if step < 1000:
            total_loss = mse(out_y, y)
        else:
            total_loss = 1 - ssim(out_y, y)

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        if (step + 1) % opt.save_frequency == 0:
            print(f'Iteration {step+1}, Loss: {total_loss.item():.6f}')

            # Save based on the output_type argument
            if opt.output_type == 1 or opt.output_type == 3:
                # Save clear image
                save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)
                out_x_np = torch_to_np(out_x)
                out_x_np = out_x_np.squeeze()
                out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
                
                # Option 1: Convert to 8-bit (values between 0-255)
                # Normalize if your data is not already in 0-1 range
                if out_x_np.min() < 0 or out_x_np.max() > 1:
                    out_x_np_normalized = (out_x_np - out_x_np.min()) / (out_x_np.max() - out_x_np.min())
                else:
                    out_x_np_normalized = out_x_np
                
                out_x_np_8bit = img_as_ubyte(out_x_np_normalized)
                imsave(save_path, out_x_np_8bit)
                print(f"Clear image saved at {save_path}")

            if opt.output_type == 2 or opt.output_type == 3:
                # Save estimated kernel
                save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
                out_k_np = torch_to_np(out_k_m)
                out_k_np = out_k_np.squeeze()
                out_k_np /= np.max(out_k_np)
                
                # Convert to uint8 before saving to avoid the 'cannot write mode F as PNG' error
                out_k_np_8bit = img_as_ubyte(out_k_np)
                imsave(save_path, out_k_np_8bit)
                print(f"Estimated kernel saved at {save_path}")

            # Save models
            torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))