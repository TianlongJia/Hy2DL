#!/usr/bin/env python
# coding: utf-8

# **General Description**
# 
# The following notebook contains the code to create, train, validate, and test a rainfall-runoff model using an LSTM network architecture. The notebook support running experiments in different large-sample hydrology datasets including: CAMELS-GB, CAMELS-US, CAMELS-DE. The details for each dataset can be read from a .yml file.
# 
# ***Authors:***
# - Eduardo Acuña Espinoza (eduardo.espinoza@kit.edu)
# - Manuel Alvarez Chaves (manuel.alvarez-chaves@simtech.uni-stuttgart.de)

# # Config initialization

# In[1]:


# Import necessary packages
import datetime
import pickle
import random
import sys
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")
# Import classes and functions from other files
from hy2dl.datasetzoo import get_dataset
from hy2dl.evaluation.metrics import nse
from hy2dl.modelzoo import get_model
from hy2dl.training.loss import nse_basin_averaged
from hy2dl.utils.config import Config
from hy2dl.utils.optimizer import Optimizer
from hy2dl.utils.utils import set_random_seed, upload_to_device

# colorblind friendly palette
color_palette = {"observed": "#377eb8", "simulated": "#4daf4a"}


# Part 1. Initialize information

# In[2]:


# Create a dictionary where all the information will be stored
experiment_settings = {}

# Experiment name
# experiment_settings["experiment_name"] = "bs_256_uniqueBlocksTrue_random_0.8"
experiment_settings["experiment_name"] = "CPC_Generate_memmap"

# paths to access the information
experiment_settings["path_data"] = "../data/CAMELS_DE"
experiment_settings["path_entities"] = "../data/basin_id/basins_camels_de_1582.txt"
# experiment_settings["path_entities"] = "../data/basin_id/basins_camels_de_hourly_3.txt"

experiment_settings["dynamic_input"] = [
    "precipitation_mean",
    "precipitation_stdev",
    "radiation_global_mean",
    "temperature_min",
    "temperature_max",
]

experiment_settings["target"] = ["discharge_spec_obs"]

# static attributes that will be used. If one is not using static_inputs, initialize the variable as an empty list.
experiment_settings["static_input"] = [
    "area",
    "elev_mean",
    "clay_0_30cm_mean",
    "sand_0_30cm_mean",
    "silt_0_30cm_mean",
    "artificial_surfaces_perc",
    "agricultural_areas_perc",
    "forests_and_seminatural_areas_perc",
    "wetlands_perc",
    "water_bodies_perc",
    "p_mean",
    "p_seasonality",
    "frac_snow",
    "high_prec_freq",
    "low_prec_freq",
    "high_prec_dur",
    "low_prec_dur",
]

# # # time periods 
experiment_settings["training_period"] = ["1970-10-01", "1995-12-31"] # 25 years for pre-training

# # # time periods (for short test)
# experiment_settings["training_period"] = ["2000-01-01", "2000-01-05"]
# experiment_settings["validation_period"] = ["2001-01-01", "2001-01-05"]
# experiment_settings["testing_period"] = ["2002-01-01", "2002-01-05"]

# model configuration
experiment_settings["hidden_size"] = 128
experiment_settings["batch_size_training"] = 256    # original: 256
experiment_settings["batch_size_evaluation"] = 1024 # original: 1024
experiment_settings["epochs"] = 3  # 12
experiment_settings["dropout_rate"] = 0.4
experiment_settings["learning_rate"] = 0.001
experiment_settings["steplr_step_size"] = 10
experiment_settings["steplr_gamma"] = 0.5
experiment_settings["validate_every"] = 1
experiment_settings["validate_n_random_basins"] = -1

experiment_settings["seq_length_hindcast"] = 365

# experiment_settings["predict_last_n"] = 24

# experiment_settings["dynamic_embedding"] = {"hiddens": [10, 10, 10]}

experiment_settings["CPC_embedding"] = {"hiddens": [512, 512, 512, 512]}

# device to train the model
experiment_settings["device"] = "cuda:0"
# experiment_settings["device"] = "cpu"
experiment_settings["num_workers"] = 0  # ori: 4

# define random seed
experiment_settings["random_seed"] = 110

# dataset
experiment_settings["dataset"] = "camels_de"
# model
experiment_settings["model"] = "cpc_lstm" 
# experiment_settings["model"] = "CudaLSTM" 
experiment_settings["initial_forget_bias"] = 3.0


# In[3]:


# Read experiment settings
config = Config(experiment_settings)
config.init_experiment()
config.dump()


# # Create pre-training datasets and dataloaders

# In[4]:


# Get dataset class
Dataset = get_dataset(config)

# Dataset training
config.logger.info(f"Loading training data from {config.dataset} dataset")
total_time = time.time()

training_dataset = Dataset(cfg=config, time_period="training")

training_dataset.calculate_basin_std()
training_dataset.calculate_global_statistics(save_scaler=True)
training_dataset.standardize_data()

config.logger.info(f"Number of entities with valid samples: {len(training_dataset.df_ts)}")
config.logger.info(
    f"Time required to process {len(training_dataset.df_ts)} entities: "
    f"{datetime.timedelta(seconds=int(time.time() - total_time))}"
)
config.logger.info(f"Number of valid training samples: {len(training_dataset)}\n")

# Dataloader training
train_loader = DataLoader(
    dataset=training_dataset,
    batch_size=config.batch_size_training,
    shuffle=True,
    drop_last=True,
    collate_fn=training_dataset.collate_fn,
    num_workers=config.num_workers,
)

# Print details of a loader´s sample to check that the format is correct
config.logger.info("Details training dataloader".center(60, "-"))
config.logger.info(f"Batch structure (number of batches: {len(train_loader)})")
config.logger.info(f"{'Key':^30}|{'Shape':^30}")
# config.logger.info("-" * 60)
# Loop through the sample dictionary and print the shape of each element
for key, value in next(iter(train_loader)).items():
    if key.startswith(("x_d", "x_conceptual")):
        config.logger.info(f"{key}")
        for i, v in value.items():
            config.logger.info(f"{i:^30}|{str(v.shape):^30}")
    else:
        config.logger.info(f"{key:<30}|{str(value.shape):^30}")

config.logger.info("")  # prints a blank line


# In[5]:


# # Check the one train sample
# dataset_sample = training_dataset[1]
# print(f"One sample in training dataset look like: {dataset_sample}")


# # Generate memmap.npy

# In[5]:


# Choose mode: "pre_train" or "fine-tune"
mode = "pre_train"  # will generate data without labels
# mode = "fine_tune"  # will generate data with labels

# 使用绝对路径避免 Windows 的 memmap bug
memmap_path = "./memmap.npy"
meta_path = "./memmap_meta.npz"

# ===== 创建 memmap =====
data_mm = np.memmap(
    memmap_path,
    dtype="float32",
    mode="w+",
    shape=((len(training_dataset) * config.seq_length_hindcast),
           (len(config.dynamic_input) + len(config.static_input)))
)

start_list = []
length_list = []
file_idx_list = []
label_list = []

row_index = 0

for i in tqdm(range(len(training_dataset)), desc="Processing training samples", ncols=80):
    sample = training_dataset[i]

    # ===== x_d 动态变量 =====
    x_d_dict = sample["x_d"]
    dyn_keys = sorted(x_d_dict.keys())
    dyn_vars = [x_d_dict[k].numpy() for k in dyn_keys]
    x_d = np.stack(dyn_vars, axis=-1)  

    # ===== x_s 静态变量 =====
    x_s = sample["x_s"].numpy()
    x_s = np.tile(x_s, (config.seq_length_hindcast, 1))       

    # ===== 拼接 =====
    merged = np.concatenate([x_d, x_s], axis=-1) 

    # ===== 写入 memmap =====
    data_mm[row_index : row_index + config.seq_length_hindcast] = merged

    # ===== meta =====
    start_list.append(row_index)
    length_list.append(config.seq_length_hindcast)
    file_idx_list.append(0)

    # ===== 记录 label（y_obs） =====
    y_obs = sample["y_obs"].numpy().astype(np.float32)  # (N,)
    label_list.append(y_obs)

    row_index += config.seq_length_hindcast

data_mm.flush()

label_array = np.stack(label_list).astype(np.float32).squeeze()

# ===== Save meta =====
if mode=="pre_train":
    np.savez_compressed(
        meta_path,
        start=np.array(start_list),
        length=np.array(length_list),
        shape=np.array([[(len(training_dataset) * config.seq_length_hindcast), (len(config.dynamic_input) + len(config.static_input))]]),
        file_idx=np.array(file_idx_list),
        dtype=np.array("float32"),
        filenames=np.array([memmap_path])
    )
elif mode=="fine_tune":
    np.savez_compressed(
        meta_path,
        start=np.array(start_list),
        length=np.array(length_list),
        shape=np.array([[(len(training_dataset) * config.seq_length_hindcast), (len(config.dynamic_input) + len(config.static_input))]]),
        file_idx=np.array(file_idx_list),
        dtype=np.array("float32"),
        filenames=np.array([memmap_path]),
        label=label_array
        )


# In[7]:


# 1. 加载元数据
meta = np.load(meta_path, allow_pickle=True)
start = meta["start"]
length = meta["length"]
shape = meta["shape"]
file_idx = meta["file_idx"]
dtype = np.dtype(str(meta["dtype"]))
print("dataset shape: ", shape)  # the first element is total length (sequence lenghth * num_samples), the second is feature dimension (12 for ECG)

# 2. 打开 memmap 文件
memmap_data = np.memmap(memmap_path, dtype=dtype, mode='r', shape=tuple(shape[0]))

# # 3. 访问第一个样本（示例）
idx = 0
sample_start = start[idx]
sample_length = length[idx]
print("No. samples: ", len(start))

sample = memmap_data[sample_start : sample_start + sample_length]
print("One sample shape:", sample.shape)
print("The first 5 timesteps of this sample: ", sample[:5, :])  # 显示前5个时间步的12维数据


# In[8]:


print("==== Keys in meta file ====")
print(meta.files)

print("\n==== Content ====")
for key in meta.files:
    print(f"\n--- {key} ---")
    print(meta[key])


# In[ ]:




