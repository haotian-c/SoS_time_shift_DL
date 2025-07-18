# ========== Imports ==========
import os
import sys
import random
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.transform import resize
import scipy.io
import mat73
from sklearn.metrics import mean_squared_error

# ========== Parameters ==========
EPOCH_NUM = 50
BATCH_SIZE = 32
lr = 0.0005
weight_decay = 0.1
factor_ssim = 5000

BEAMFORMING_SOS = 1540
INPUT_SCALING = 100000000
CENTERIZED_OUTPUT = BEAMFORMING_SOS

STARTING_PIXEL = 19
WIDTH_IN_PIXEL = 50

BASE_GPU = 0
DEVICE_IDS = [0, 1, 2, 3]

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils import (
    remove_blind_region_psf0,
    remove_blind_region_psf7p5,
    remove_blind_region_minuspsf7p5,
    calculate_recon_metrics,
    create_rand_gaussian_mask,
    add_noise
)
from model import build_net
from pytorch_reproducibility import set_seed_n_cudnn
import pytorch_SSIM_module

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sns.set()
sns.set_style("whitegrid", {'axes.grid': False})

print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())

SSIM_LOSS = pytorch_SSIM_module.SSIM()
criterion = nn.MSELoss()


data = []
label = []
for dir_data_i, index_range_i in dict_data_dir_and_nums.items():
    for i in index_range_i:
        # fill in your data loading details below
        image_i = np.array([
            remove_blind_region_psf7p5(-mat73.loadmat() * INPUT_SCALING, remove_blind_region_psf0(mat73.loadmat())) * INPUT_SCALING,
            remove_blind_region_minuspsf7p5(-mat73.loadmat()) * INPUT_SCALING
        ])  

        image_i = np.nan_to_num(image_i, nan=0)
        image_i = torch.tensor(image_i)
        image_i = F.pad(image_i, (1, 1, 0, 0))
        image_i = image_i[:, :73, :]
        data.append(image_i)

        sos_map_i = scipy.io.loadmat(
            f"{dir_data_i}object_{i}/sos_phamton_gt_{i}.mat"
        )['sound_speed_map'].astype('int16')

        new_height = sos_map_i.shape[0] // 11
        new_width = sos_map_i.shape[1] // 11

        sos_map_resized = resize(sos_map_i, (new_height, new_width + 1), anti_aliasing=True, preserve_range=True)
        sos_map_resized = np.nan_to_num(sos_map_resized, nan=0)
        sos_map_tensor = torch.tensor(sos_map_resized)
        sos_map_tensor = F.pad(sos_map_tensor, (1, 1, 0, 0))
        sos_map_tensor = sos_map_tensor[:73, 2:-2]
        sos_map_tensor = sos_map_tensor.unsqueeze(0)
        label.append(sos_map_tensor)

    print(f'Finished loading: {dir_data_i}')

data = torch.stack(data)
label = torch.stack(label)

label_trainingset = label[20:, :, :]
data_trainingset = data[20:, :, :, :]
label_testingset = label[:20, :, :]
data_testingset = data[:20, :, :, :]

# ========== Model Utilities ==========
def initize_model(generator, if_use_pretrain, pretrain_path):
    if if_use_pretrain and pretrain_path:
        generator.load_state_dict(torch.load(pretrain_path))

# ========== Evaluation ==========
def eval_on_testingset(generator):
    generator.eval()
    with torch.no_grad():
        preds = generator(data_testingset.to(f'cuda:{BASE_GPU}', dtype=torch.float)) * 100 + CENTERIZED_OUTPUT
        mse = mean_squared_error(
            preds[:, :, :, STARTING_PIXEL:STARTING_PIXEL+WIDTH_IN_PIXEL].cpu().numpy().flatten(),
            label_testingset[:, :, :, STARTING_PIXEL:STARTING_PIXEL+WIDTH_IN_PIXEL].numpy().flatten()
        )
        rmse = np.sqrt(mse)
    return rmse

# ========== Training Loop ==========
def training_loop(generator, if_ssim, lr, weight_decay, factor_ssim, dict_params_save):
    set_seed_n_cudnn(1)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=weight_decay)
    recon_criterion = nn.MSELoss()

    generator.train()
    for epoch in range(EPOCH_NUM):
        index_list = np.arange(len(data_trainingset))
        np.random.shuffle(index_list)
        for batch_i in range(len(data_trainingset) // BATCH_SIZE):
            batch_idx = index_list[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
            batch_data = data_trainingset[batch_idx]
            batch_data = add_noise(batch_data)

            batch_data = batch_data.to(f'cuda:{BASE_GPU}', dtype=torch.float)
            batch_label = label_trainingset[batch_idx].to(f'cuda:{BASE_GPU}', dtype=torch.float)

            optimizer.zero_grad()
            batch_pred = generator(batch_data) * 100 + CENTERIZED_OUTPUT

            recon_loss = recon_criterion(
                batch_pred[:, :, :, STARTING_PIXEL:STARTING_PIXEL+WIDTH_IN_PIXEL], 
                batch_label[:, :, :, STARTING_PIXEL:STARTING_PIXEL+WIDTH_IN_PIXEL]
            )
            ssim_loss = -SSIM_LOSS(
                (batch_pred[:, :, :, STARTING_PIXEL:STARTING_PIXEL+WIDTH_IN_PIXEL] - 1400) / 200,
                (batch_label[:, :, :, STARTING_PIXEL:STARTING_PIXEL+WIDTH_IN_PIXEL] - 1400) / 200
            )

            loss = int(if_ssim) * factor_ssim * ssim_loss + recon_loss
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print(f'epoch {epoch} out of {EPOCH_NUM}')
            print('recon_loss', recon_loss)
            print('RMSE on the testing set is', eval_on_testingset(generator))
            generator.train()

# ========== Main Training ==========

dict_params_save = {
    'if_save': True,
    'dir_for_save': f'./experimental_results/{ablation_mode_i}/',
    'prefix': ''
}

Generator = build_net().to(f'cuda:{BASE_GPU}')
initize_model(
    Generator,
    if_use_pretrain=map_ablation_mode_to_parameters[ablation_mode_i]['if_use_pretrain'],
    pretrain_path=map_ablation_mode_to_parameters[ablation_mode_i].get('pretrain_path')
)
Generator = nn.DataParallel(Generator, device_ids=DEVICE_IDS)

training_loop(
    Generator,
    if_ssim=map_ablation_mode_to_parameters[ablation_mode_i]['if_SSIM'],
    lr=lr,
    weight_decay=weight_decay,
    factor_ssim=factor_ssim,
    dict_params_save=dict_params_save
)