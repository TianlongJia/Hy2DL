#!/usr/bin/env python
# coding: utf-8

# # Test MF-LSTM performance for Rainfall-runoff prediction

# **General Description**
# 
# (1) Loading model from model weight file, and laod test dataset from model_configuration_file (see model training file);
# 
# (2)
# 

# In[ ]:


# Import necessary packages
import pickle
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append("..")
from hy2dl.aux_functions.functions_evaluation import nse
from hy2dl.aux_functions.utils import upload_to_device, create_folder
from hy2dl.datasetzoo.hourlycamelsde import HourlyCAMELS_DE as Datasetclass
from hy2dl.modelzoo.mflstm import MFLSTM as modelclass


# ## 1. Load model and test dataset

# In[ ]:


import json

model_configuration_file = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/Alex_test/test_seed_110/model_config.json"
model_weight_path = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/Alex_test/test_seed_110/weights/epoch_4"
scaler_path = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/Alex_test/test_seed_110/scaler.pickle"
test_result_save_path = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/Alex_test/test_seed_110/test_epoch4"
# test_result_save_path = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/pred_24_stride_1/30_basins/2_day_seed_110/test_results"

# model_configuration_file = r"D:\Research\Projects\Hy2DL\results\14_day_in_hourly_pred_last_24_seed_110\model_config.json"
# model_weight_path = r"D:\Research\Projects\Hy2DL\results\14_day_in_hourly_pred_last_24_seed_110\best_model_at_epoch_30"
# scaler_path = r"D:\Research\Projects\Hy2DL\results\14_day_in_hourly_pred_last_24_seed_110\scaler.pickle"
# test_result_save_path = r"D:\Research\Projects\Hy2DL\results\14_day_in_hourly_pred_last_24_seed_110\test02"

if not os.path.exists(test_result_save_path):
    os.makedirs(test_result_save_path)

# device to test the model
running_device = "gpu"

# check if model will be run in gpu or cpu and define device
if running_device == "gpu":
    print(torch.cuda.get_device_name(0))
    device = "cuda:0"
elif running_device == "cpu":
    device = "cpu"


# In[ ]:


# Load the model configuration from the JSON file
with open(model_configuration_file, "r") as f:
    full_config = json.load(f)

model_configuration = full_config["model_configuration"]

model = modelclass(model_configuration=model_configuration).to(device)
model.load_state_dict(torch.load(model_weight_path, map_location=device))

# Load the scaler
# (1) we can use training scaler
# scaler = training_dataset.scaler

# (2) we can also read a previously stored one
with open(scaler_path, "rb") as file:
  scaler = pickle.load(file)


# In[ ]:


# In evaluation (validation and testing) we will create an individual dataset per basin. This will give us more 
# flexibility

path_entities = full_config["path_entities"]
entities_ids = np.loadtxt(path_entities, dtype="str").tolist()
entities_ids = [entities_ids] if isinstance(entities_ids, str) else entities_ids
testing_dataset = {}
for entity in entities_ids:
    dataset = Datasetclass(
        dynamic_input=full_config["dynamic_input"],
        static_input=full_config["static_input"],
        target=full_config["target"],
        time_period=full_config["validation_period"],  # testing_period  # To do
        # time_period=full_config["testing_period"],
        path_data=full_config["path_data"],
        entity=entity,
        check_NaN=False,
        predict_last_n=model_configuration["predict_last_n_evaluation"],
        sequence_length=model_configuration["seq_length"],
        custom_freq_processing=model_configuration["custom_freq_processing"],
        dynamic_embedding=model_configuration["dynamic_embeddings"],
        # unique_prediction_blocks=model_configuration["unique_prediction_blocks"],
        unique_prediction_blocks=model_configuration["unique_prediction_blocks_evaluation"],  # To do
    )

    dataset.scaler = scaler
    dataset.standardize_data(standardize_output=False)
    testing_dataset[entity] = dataset


# ## 2. Model evaluation

# In[ ]:


model.eval()
test_results = {}
with torch.no_grad():
    for basin, dataset in testing_dataset.items():
        loader = DataLoader(
            dataset=dataset,
            batch_size=model_configuration["batch_size_evaluation"],
            shuffle=False,
            drop_last=False,
            collate_fn=testing_dataset[basin].collate_fn,
        )

        df_ts = pd.DataFrame()
        for sample in loader:
            sample = upload_to_device(sample, device)  # upload tensors to device
            pred = model(sample)
            # backtransformed information
            y_sim = pred["y_hat"] * dataset.scaler["y_std"].to(device) + dataset.scaler["y_mean"].to(device)

            # join results in a dataframe and store them in a dictionary (is easier to plot later)
            df = pd.DataFrame(
                {
                    "y_obs": sample["y_obs"].flatten().cpu().detach(),
                    "y_sim": y_sim[:, -model_configuration["predict_last_n_evaluation"] :, :].flatten().cpu().detach(),
                },
                index=pd.to_datetime(sample["date"].flatten()),
            )

            df_ts = pd.concat([df_ts, df], axis=0)

            # remove from cuda
            del sample, pred, y_sim
            torch.cuda.empty_cache()

        test_results[basin] = df_ts

# Save results as a pickle file
with open(os.path.join(test_result_save_path, "test_results.pickle"), "wb") as f:
    pickle.dump(test_results, f)


# ## 3. Prediction results analysis
# 
# You can choose 3.1 or 3.2 to load model prediction results

# ### 3.1 Load results from "test_results" variable, outputed by Step 2

# In[ ]:


print("***************  Evaluation process begin  ****************")

# Loss testing
loss_testing = nse(df_results=test_results, average=False)
df_NSE = pd.DataFrame(data={"basin_id": test_results.keys(), "NSE": np.round(loss_testing, 3)})
df_NSE = df_NSE.set_index("basin_id")

# Save the NSE for each basin in a csv file
df_NSE.to_csv(os.path.join(test_result_save_path, "NSE_testing.csv"), index=True, header=True)
mean_nse = df_NSE["NSE"].mean()
median_nse = df_NSE["NSE"].median()
print(f"Mean NSE across all basins: {mean_nse:.3f}")
print(f"Median  NSE across all basins: {median_nse:.3f}")


# In[ ]:


# Plot the histogram
plt.hist(df_NSE["NSE"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Add NSE statistics to the plot
plt.text(
    0.01,
    0.8,
    (
        f'Mean: {"%.2f" % df_NSE["NSE"].mean():>7}\n'
        f'Median: {"%.2f" % df_NSE["NSE"].median():>0}\n'
        f'Max: {"%.2f" % df_NSE["NSE"].max():>9}\n'
        f'Min: {"%.2f" % df_NSE["NSE"].min():>10}'
    ),
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.5),
)

# Format plot
plt.rcParams["figure.figsize"] = (20, 5)
plt.xlabel("NSE", fontsize=12, fontweight="bold")
plt.ylabel("Frequency", fontsize=12, fontweight="bold")
plt.title("NSE Histogram", fontsize=16, fontweight="bold")
plt.savefig(os.path.join(test_result_save_path, "NSE_Histogram.png"), bbox_inches="tight", pad_inches=0)
# plt.show()