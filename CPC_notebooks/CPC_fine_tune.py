#!/usr/bin/env python
# coding: utf-8

# # Fine-tune process
# 
# To do:
# 
# (1) Change the loss function (fine-tuning) to exclude basin_std for fine-tuning one single basin. Now, I laod scaler obtained from pre-training for fine-tuning, val, and test dataset. I also calculate basin_std and use "Basin-averaged NSE" in loss function, if we fine-tune on over one basin. 

# ## Config initialization

# In[ ]:


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


# In[ ]:


# if you have external parameter to inject in this notebook
percent = float(os.environ.get("Percent"))


# Part 1. Initialize information

# In[ ]:


# Create a dictionary where all the information will be stored
experiment_settings = {}

# Experiment name
# experiment_settings["experiment_name"] = "bs_256_uniqueBlocksTrue_random_0.8"
experiment_settings["experiment_name"] = f"CPC_FT_20th_ep_with_{int(percent*100)}%"

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
# experiment_settings["training_period"] = ["1970-10-01", "1999-12-31"]
# experiment_settings["validation_period"] = ["1965-10-01", "1970-09-30"]
# experiment_settings["testing_period"] = ["2000-01-01", "2020-12-31"]

# # # For SSL experiment 
# I used ["1970-10-01", "1995-12-31"] for pre-training
experiment_settings["training_period"] = ["1998-12-31", "1999-12-31"]  # 100% fine-tune (1 years)
experiment_settings["validation_period"] = ["1965-10-01", "1970-09-30"]  # validation (5 years)
experiment_settings["testing_period"] = ["2000-01-01", "2020-12-31"]  # test (20 years)

# model configuration
experiment_settings["hidden_size"] = 128    # 128
experiment_settings["batch_size_training"] = 256    # original: 256
experiment_settings["batch_size_evaluation"] = 1024 # original: 1024
experiment_settings["epochs"] = 20  # 20
experiment_settings["dropout_rate"] = 0.4
experiment_settings["learning_rate"] = 0.001
experiment_settings["steplr_step_size"] = 10
experiment_settings["steplr_gamma"] = 0.5
experiment_settings["validate_every"] = 1
experiment_settings["validate_n_random_basins"] = -1

experiment_settings["seq_length"] = 365

experiment_settings["CPC_embedding"] = {"hiddens": [512, 512, 512, 512]}  

# device to train the model
experiment_settings["device"] = "cuda:0"
# experiment_settings["device"] = "cpu"
experiment_settings["num_workers"] = 4  # ori: 4

# define random seed
experiment_settings["random_seed"] = 110

# dataset
experiment_settings["dataset"] = "camels_de"
# model
experiment_settings["model"] = "cpc_lstm"  # "CudaLSTM"
experiment_settings["initial_forget_bias"] = 3.0

# configs for fine-tuning
experiment_settings["pre_trained_path"] = r"/hkfs/home/haicore/iwu/qa8171/Project/CPC2HY/results/pre_train/version_0/" # To do: Change it
experiment_settings["finetune_modules"] = ["embedding_hindcast", "lstm", "linear"]


# In[ ]:


# Read experiment settings
config = Config(experiment_settings)
config.init_experiment()
config.dump()
# Set a seed for various packages below, enable to reproduce the results
set_random_seed(cfg=config) 


# ## Create training datasets and dataloaders

# In[ ]:


from torch.utils.data import SubsetRandomSampler

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
    "Time required to process {} entities: {}".format(
        len(training_dataset.df_ts),
        datetime.timedelta(seconds=int(time.time() - total_time))
    )
)
config.logger.info(f"Number of total valid samples: {len(training_dataset)}\n")

# define sampling ratio of the training dataset
train_ratio = percent
num_samples = len(training_dataset)
indices = np.arange(num_samples)
np.random.shuffle(indices)
train_indices = indices[:int(train_ratio * num_samples)]
config.logger.info(f"Samping {train_ratio * 100:.0f}% of total valid training samples, resulting in {len(train_indices)} samples \n")

# Use a samler to constrcut a DataLoader
train_sampler = SubsetRandomSampler(train_indices)

# Dataloader training
train_loader = DataLoader(
    dataset=training_dataset,
    batch_size=config.batch_size_training,
    sampler=train_sampler,   # I replace "shuffle=True"
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


# In[ ]:


# # Check the one train sample
# dataset_sample = training_dataset[1]
# print(f"One sample in training dataset look like: {dataset_sample}")


# ## Construct validation dataset

# In[ ]:


# In evaluation (validation and testing) we will create an individual dataset per basin
config.logger.info(f"Loading validation data from {config.dataset} dataset")
entities_ids = np.loadtxt(config.path_entities_validation, dtype="str").tolist()
iterator = tqdm(
    [entities_ids] if isinstance(entities_ids, str) else entities_ids,
    desc="Processing entities",
    unit="entity",
    ascii=True,
)

total_time = time.time()
validation_dataset = {}
for entity in iterator:
    dataset = Dataset(cfg=config, time_period="validation", check_NaN=False, entities_ids=entity)

    dataset.scaler = training_dataset.scaler
    dataset.standardize_data(standardize_output=False)
    validation_dataset[entity] = dataset

config.logger.info(
    f"Time required to process {len(iterator)} entities: {datetime.timedelta(seconds=int(time.time() - total_time))}\n"
)


# In[ ]:


# validation_dataset["DE110000"][0]


# ## Load pre-trained model weights

# In[ ]:


# Initialize model
model = get_model(config).to(config.device)
config.logger.info(f"model architecture: {model}")

# Load pre-trained model from a checkpoint file (.pth)
# checkpoint_path = config.pre_trained_path / "best_model.pth"  # To do
checkpoint_path = config.pre_trained_path / "epoch_20.pth"  # To do


config.logger.info(f"Loaded pre-trained weights from checkpoint {checkpoint_path}")
state = torch.load(checkpoint_path, map_location=config.device)
missing, unexpected = model.load_state_dict(state, strict=False)  

config.logger.info(f"Missing component for hy2dl are kept as initialized, including: {missing}.")
config.logger.info(f"Unexpected component in CPC are not transferred into hy2dl model, including: {unexpected}.")

# config.logger.info(f"Model layers: {state.keys()}")
# config.logger.info(f"Model architecture: {model}")


# ## Freeze model's modules (if needed)

# In[ ]:


# freeze all model weights
for param in model.parameters():
    param.requires_grad = False

unresolved_modules = []

# unfreeze parameters specified in config as tuneable parameters
if isinstance(config.finetune_modules, list):
    for module_part in config.finetune_modules:
        if module_part in model.module_parts:
            module = getattr(model, module_part)
            config.logger.info(f"Trainable modules: {module}")
            for param in module.parameters():
                param.requires_grad = True
        else:
            unresolved_modules.append(module_part)
else:
    # if it was no list, it has to be a dictionary
    for module_group, module_parts in config.finetune_modules.items():
        if module_group in model.module_parts:
            if isinstance(module_parts, str):
                module_parts = [module_parts]
            for module_part in module_parts:
                module = getattr(model, module_group)[module_part]
                config.logger.info(f"Trainable modules: {module}")
                for param in module.parameters():
                    param.requires_grad = True
            else:
                unresolved_modules.append(module_group)
if unresolved_modules:
    config.logger.warning(f"Could not resolve the following module parts for finetuning: {unresolved_modules}")


# ## Fine-tune model

# In[ ]:


# Initialize optimizer
optimizer = Optimizer(cfg=config, model=model)

# Training report structure
config.logger.info("Training model".center(60, "-"))
config.logger.info(f"{'':^16}|{'Trainining':^21}|{'Validation':^21}|")
config.logger.info(f"{'Epoch':^5}|{'LR':^10}|{'Loss':^10}|{'Time':^10}|{'Metric':^10}|{'Time':^10}|")

# Define the initail val_nse for selecting the best epoch during validation
best_val_nse = float("-inf")  # Best NSE so far
best_epoch = -1
best_model_state = None
NSE_basins_at_best_epoch = []

total_time = time.time()
# Loop through epochs
for epoch in range(1, config.epochs + 1):
    train_time = time.time()
    loss_evol = []
    # Training -------------------------------------------------------------------------------------------------------
    model.train()
    # Loop through the different batches in the training dataset
    iterator = tqdm(
        train_loader, desc=f"Epoch {epoch}/{config.epochs}. Training", unit="batches", ascii=True, leave=False
    )

    for idx, sample in enumerate(iterator):
        # reach maximum iterations per epoch
        if config.max_updates_per_epoch is not None and idx >= config.max_updates_per_epoch:
            break

        sample = upload_to_device(sample, config.device)  # upload tensors to device
        optimizer.optimizer.zero_grad()  # sets gradients to zero

        # Forward pass of the model
        pred = model(sample)
        # Calcuate loss
        loss = nse_basin_averaged(y_sim=pred["y_hat"], y_obs=sample["y_obs"], per_basin_target_std=sample["std_basin"])

        # Backpropagation (calculate gradients)
        loss.backward()

        # Update model parameters (e.g, weights and biases)
        optimizer.clip_grad_and_step(epoch, idx)

        # Keep track of the loss per batch
        loss_evol.append(loss.item())
        iterator.set_postfix({"loss": f"{np.mean(loss_evol):.3f}"})

        # remove elements from cuda to free memory
        del sample, pred
        torch.cuda.empty_cache()

    # training report
    lr = optimizer.optimizer.param_groups[0]["lr"]
    mean_loss = np.mean(loss_evol)
    train_duration = str(datetime.timedelta(seconds=int(time.time() - train_time)))
    report = f"{epoch:^5}|{lr:^10.5f}|{mean_loss:^10.3f}|{train_duration:^10}|"

    # Validation -----------------------------------------------------------------------------------------------------
    if epoch % config.validate_every == 0:
        val_time = time.time()
        model.eval()
        validation_results = {}
        with torch.no_grad():
            # If we define validate_n_random_basins as 0 or negative, we take all the basins. Otherwise, we randomly
            # select the number of basins defined in validate_n_random_basins
            if config.validate_n_random_basins <= 0:
                validation_basin_ids = validation_dataset.keys()
            else:
                validation_basin_ids = random.sample(list(validation_dataset.keys()), config.validate_n_random_basins)

            # Go through each basin
            iterator = tqdm(
                validation_basin_ids,
                desc=f"Epoch {epoch}/{config.epochs}. Validation",
                unit="basins",
                ascii=True,
                leave=False,
            )

            for basin in iterator:
                loader = DataLoader(
                    dataset=validation_dataset[basin],
                    batch_size=config.batch_size_evaluation,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=validation_dataset[basin].collate_fn,
                    num_workers=config.num_workers,
                )

                df_ts = pd.DataFrame()
                for sample in loader:
                    sample = upload_to_device(sample, config.device)
                    # Forward pass of the model
                    pred = model(sample)
                    # Backtransform information (unstandardize the output)
                    y_std = validation_dataset[basin].scaler["y_std"].to(config.device)
                    y_mean = validation_dataset[basin].scaler["y_mean"].to(config.device)
                    y_sim = pred["y_hat"] * y_std + y_mean

                    # join results in a dataframe (easier to evaluate/plot later)
                    df = pd.DataFrame(
                        {"y_obs": sample["y_obs"].flatten().cpu().detach(), "y_sim": y_sim.flatten().cpu().detach()},
                        index=pd.to_datetime(sample["date"].flatten()),
                    )
                    # print("sample[y_obs]: ", sample["y_obs"])
                    # print("sample[y_sim]: ", y_sim)
                    df_ts = pd.concat([df_ts, df], axis=0)

                    # remove elements from cuda to free memory
                    del sample, pred, y_sim
                    torch.cuda.empty_cache()

                validation_results[basin] = df_ts

            # average loss validation (i.e, the median NSE of all basins in validation set)
            loss_validation = nse(df_results=validation_results)
            report += f"{loss_validation:^10.3f}|{str(datetime.timedelta(seconds=int(time.time() - val_time))):^10}|"

    # Save and update best model after each epoch
    if loss_validation > best_val_nse:
        best_val_nse = loss_validation
        best_epoch = epoch
        best_model_state= model.state_dict()
        torch.save(best_model_state, config.path_save_folder / "model" / "best_model")        

    # No validation
    else:
        report += f"{'':^10}|{'':^10}|"

    # Print report and save model
    config.logger.info(report)
    torch.save(model.state_dict(), config.path_save_folder / "model" / f"model_epoch_{epoch}")
    # modify learning rate
    optimizer.update_optimizer_lr(epoch=epoch)

if best_epoch is not None:
    config.logger.info(f"Best (validation) median NSE: {best_val_nse:.3f} at epoch {best_epoch}")
else:
    config.logger.info("No best model was selected (i.e, best_model_state is None).")

# print total training time
config.logger.info(f"Total training time: {datetime.timedelta(seconds=int(time.time() - total_time))}\n")


# ## Check model weights after training

# In[ ]:


# # Load pre-trained model from a checkpoint file
# checkpoint_path = "../results/fine-tune_seed_110/model/model_epoch_3"

# state = torch.load(checkpoint_path, map_location=config.device)
# model.load_state_dict(state)
# # print("model weights: ", state)
# state.keys()


# In[ ]:


# model


# ## Test model

# In[ ]:


# In case I already trained an LSTM I can re-construct the model. I just need to define the epoch for which I want to
# re-construct the model
model = get_model(config).to(config.device)
model.load_state_dict(torch.load(config.path_save_folder / "model" / "best_model", map_location=config.device))


# In[ ]:


# Read previously generated scaler
with open(config.path_save_folder / "scaler.pickle", "rb") as file:
    scaler = pickle.load(file)


# In[ ]:


# In evaluation (validation and testing) we will create an individual dataset per basin
config.logger.info(f"Loading testing data from {config.dataset} dataset")

entities_ids = np.loadtxt(config.path_entities_testing, dtype="str").tolist()
iterator = tqdm(
    [entities_ids] if isinstance(entities_ids, str) else entities_ids,
    desc="Processing entities",
    unit="entity",
    ascii=True,
)

total_time = time.time()
testing_dataset = {}
for entity in iterator:
    dataset = Dataset(cfg=config, time_period="testing", check_NaN=False, entities_ids=entity)

    dataset.scaler = scaler
    dataset.standardize_data(standardize_output=False)
    testing_dataset[entity] = dataset

config.logger.info(
    f"Time required to process {len(iterator)} entities: {datetime.timedelta(seconds=int(time.time() - total_time))}\n"
)


# In[ ]:


config.logger.info("Testing model".center(60, "-"))
total_time = time.time()

model.eval()
test_results = {}
with torch.no_grad():
    # Go through each basin
    iterator = tqdm(testing_dataset, desc="Testing", unit="basins", ascii=True)
    for basin in iterator:
        loader = DataLoader(
            dataset=testing_dataset[basin],
            batch_size=config.batch_size_evaluation,
            shuffle=False,
            drop_last=False,
            collate_fn=testing_dataset[basin].collate_fn,
            num_workers=config.num_workers,
        )

        df_ts = pd.DataFrame()
        for sample in loader:
            sample = upload_to_device(sample, config.device)  # upload tensors to device
            pred = model(sample)
            # backtransformed information
            y_sim = pred["y_hat"] * testing_dataset[basin].scaler["y_std"].to(config.device) + (
                testing_dataset[basin].scaler["y_mean"].to(config.device)
            )

            # join results in a dataframe and store them in a dictionary (is easier to plot later)
            df = pd.DataFrame(
                {"y_obs": sample["y_obs"].flatten().cpu().detach(), "y_sim": y_sim.flatten().cpu().detach()},
                index=pd.to_datetime(sample["date"].flatten()),
            )

            df_ts = pd.concat([df_ts, df], axis=0)

            # remove from cuda
            del sample, pred, y_sim
            torch.cuda.empty_cache()

        test_results[basin] = df_ts

# Save results as a pickle file
with open(config.path_save_folder / "test_results.pickle", "wb") as f:
    pickle.dump(test_results, f)

config.logger.info(f"Total testing time: {datetime.timedelta(seconds=int(time.time() - total_time))}")


# ## Initial analysis

# In[ ]:


# Loss testing
loss_testing = nse(df_results=test_results, average=False)
df_NSE = pd.DataFrame(data={"basin_id": testing_dataset.keys(), "NSE": np.round(loss_testing, 3)})
df_NSE = df_NSE.set_index("basin_id")
df_NSE.to_csv(config.path_save_folder / "NSE_testing.csv", index=True, header=True)

config.logger.info(f"Mean test NSE: {df_NSE['NSE'].mean():.3f}")
config.logger.info(f"Median test Nse: {df_NSE['NSE'].median():.3f}")

# Plot the histogram
plt.hist(df_NSE["NSE"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Add NSE statistics to the plot
plt.text(
    0.01,
    0.8,
    (
        f"Mean: {'%.2f' % df_NSE['NSE'].mean():>7}\n"
        f"Median: {'%.2f' % df_NSE['NSE'].median():>0}\n"
        f"Max: {'%.2f' % df_NSE['NSE'].max():>9}\n"
        f"Min: {'%.2f' % df_NSE['NSE'].min():>10}"
    ),
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.5),
)

# Format plot
plt.rcParams["figure.figsize"] = (20, 5)
plt.xlabel("NSE", fontsize=12, fontweight="bold")
plt.ylabel("Frequency", fontsize=12, fontweight="bold")
plt.title("NSE Histogram", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(config.path_save_folder / "NSE Histogram.jpg")
plt.show()


# In[ ]:


# Plot simulated and observed discharges
basin_to_analyze = random.sample(list(test_results.keys()), 1)[0]

plt.figure(figsize=(20, 7))
plt.plot(test_results[basin_to_analyze]["y_obs"], label="observed", color=color_palette["observed"])
plt.plot(test_results[basin_to_analyze]["y_sim"], label="simulated", alpha=0.5, color=color_palette["simulated"])

# Format plot
plt.xlabel("Date", fontsize=12, fontweight="bold")
plt.ylabel("Discharge [mm/d]", fontsize=12, fontweight="bold")
plt.title("Modeling results", fontsize=16, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="upper right", fontsize=12)
plt.tight_layout()
plt.savefig(config.path_save_folder / "Q_prediction.jpg")
plt.show()


# In[ ]:




