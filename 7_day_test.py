#!/usr/bin/env python
# coding: utf-8

# # Rainfall-runoff model using MF-LSTM

# **General Description**
# 
# The following notebook contains the code to create, train, validate, and test a rainfall-runoff model using an LSTM network architecture. The code allows for the creation of single-basin models but is conceptualized to create regional models. The code is intended as an initial introduction to the topic, prioritizing interpretability over modularity.
# 
# The logic of the code is heavily based on [Neural Hydrology](https://doi.org/10.21105/joss.04050) [1]. For a more flexible, robust, and modular implementation of deep learning methods in hydrological modeling, we advise the use of Neural Hydrology. 
# 
# **Experiment Details**
# - In this example we use the MF-LSTM (multi-frequency LSTM) architecture, which allow us to work the data at different frequencies (daily/hourly). 
# - This experiment use a similar setup as the experiments presented in [2].
# 
# **Authors:**
# - Eduardo Acuña Espinoza (eduardo.espinoza@kit.edu)
# 
# **References:**
# 
# [1]: Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology – A Python library for deep learning research in hydrology. Journal of Open Source Software, 7, 4050. https://doi.org/10.21105/joss.04050
# 
# [2]: Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Lin, J., & Hochreiter, S. (2021). Rainfall–runoff prediction at multiple timescales with a single long short-term memory network. Hydrology and Earth System Sciences, 25(4), 2045–2062. https://doi.org/10.5194/hess-25-2045-2021
# 

# In[1]:


# Import necessary packages
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hy2dl.aux_functions.functions_evaluation import nse
from hy2dl.aux_functions.utils import upload_to_device
from hy2dl.datasetzoo.hourlycamelsde import HourlyCAMELS_DE as Datasetclass
from hy2dl.modelzoo.mflstm import MFLSTM as modelclass


# ## 1. Test LSTM

# In[2]:


import json

model_configuration_file = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/7_day_in_hourly_pred_last_24_seed_110/model_config.json"
model_weight_path = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/7_day_in_hourly_pred_last_24_seed_110/best_model_at_epoch_25"
scaler_path = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/7_day_in_hourly_pred_last_24_seed_110/scaler.pickle"

test_result_save_path = r"/hkfs/home/haicore/iwu/qa8171/Project/Hy2DL/results/7_day_in_hourly_pred_last_24_seed_110/test"

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


# In[3]:


# path_save_folder = "results/MFLSTM_hourly_de_test_seed_110/"

# Load the model configuration from the JSON file
with open(model_configuration_file, "r") as f:
    full_config = json.load(f)

model_configuration = full_config["model_configuration"]

model = modelclass(model_configuration=model_configuration).to(device)
model.load_state_dict(torch.load(model_weight_path, map_location=device))

# Load the scaler
# (1) we can use training scaler
# scaler = training_dataset.scaler

# (2) we can read a previously stored one
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
        time_period=full_config["testing_period"],
        path_data=full_config["path_data"],
        entity=entity,
        check_NaN=False,
        predict_last_n=model_configuration["predict_last_n_evaluation"],
        sequence_length=model_configuration["seq_length"],
        custom_freq_processing=model_configuration["custom_freq_processing"],
        dynamic_embedding=model_configuration["dynamic_embeddings"],
        unique_prediction_blocks=model_configuration["unique_prediction_blocks"],
    )

    dataset.scaler = scaler
    dataset.standardize_data(standardize_output=False)
    testing_dataset[entity] = dataset


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


# In[ ]:


# import math

# # do testing in batches of catchments
# # Load all entity IDs
# entities_ids = np.loadtxt(path_entities, dtype="str").tolist()
# entities_ids = [entities_ids] if isinstance(entities_ids, str) else entities_ids

# # Define batch size for processing catchments
# catchment_batch_size = 200  # Adjust based on your memory constraints
# total_catchments = len(entities_ids)
# num_batches = math.ceil(total_catchments / catchment_batch_size)

# # Create directory for intermediate results if it doesn't exist
# intermediate_results_dir = os.path.join(path_save_folder, "intermediate_results")
# os.makedirs(intermediate_results_dir, exist_ok=True)

# # Process entities in batches
# model.eval()
# all_test_results = {}

# for batch_idx in range(num_batches):
#     batch_filename = f"test_results_batch_{batch_idx+1}.pickle"
#     if os.path.exists(os.path.join(intermediate_results_dir, batch_filename)):
#         print(f"Skipping catchment batch {batch_idx+1}/{num_batches}")
#         continue
        
#     print(f"Processing catchment batch {batch_idx+1}/{num_batches}")
    
#     # Get current batch of entities
#     start_idx = batch_idx * catchment_batch_size
#     end_idx = min((batch_idx + 1) * catchment_batch_size, total_catchments)
#     batch_entities = entities_ids[start_idx:end_idx]
    
#     # Create dataset for current batch of entities
#     batch_testing_dataset = {}
#     for entity in batch_entities:
#         dataset = Datasetclass(
#             dynamic_input=dynamic_input,
#             target=target,
#             sequence_length=model_configuration["seq_length"],
#             time_period=testing_period,
#             path_data=path_data,
#             entity=entity,
#             check_NaN=False,
#             predict_last_n=model_configuration["predict_last_n_evaluation"],
#             static_input=static_input,
#             custom_freq_processing=model_configuration["custom_freq_processing"],
#             dynamic_embedding=model_configuration["dynamic_embeddings"],
#             unique_prediction_blocks=model_configuration["unique_prediction_blocks"],
#         )
#         dataset.scaler = scaler
#         dataset.standardize_data(standardize_output=False)
#         batch_testing_dataset[entity] = dataset
    
#     # Process each entity in the current batch
#     batch_results = {}
#     with torch.no_grad():
#         for basin, dataset in batch_testing_dataset.items():
#             loader = DataLoader(
#                 dataset=dataset,
#                 batch_size=model_configuration["batch_size_evaluation"],
#                 shuffle=False,
#                 drop_last=False,
#                 collate_fn=batch_testing_dataset[basin].collate_fn,
#             )
#             df_ts = pd.DataFrame()
#             for sample in loader:
#                 sample = upload_to_device(sample, device)  # upload tensors to device
#                 pred = model(sample)
#                 # backtransformed information
#                 y_sim = pred["y_hat"] * dataset.scaler["y_std"].to(device) + dataset.scaler["y_mean"].to(device)
#                 # join results in a dataframe
#                 df = pd.DataFrame(
#                     {
#                         "y_obs": sample["y_obs"].flatten().cpu().detach(),
#                         "y_sim": y_sim[:, -model_configuration["predict_last_n_evaluation"] :, :].flatten().cpu().detach(),
#                     },
#                     index=pd.to_datetime(sample["date"].flatten()),
#                 )
#                 df_ts = pd.concat([df_ts, df], axis=0)
#                 # remove from cuda
#                 del sample, pred, y_sim
#                 torch.cuda.empty_cache()
#             batch_results[basin] = df_ts
            
#             # Optional: Update all_test_results dictionary to maintain complete results
#             all_test_results[basin] = df_ts
    
#     # Save intermediate results for this batch
#     batch_filename = f"test_results_batch_{batch_idx+1}.pickle"
#     with open(os.path.join(intermediate_results_dir, batch_filename), "wb") as f:
#         pickle.dump(batch_results, f)
    
#     # Clear memory
#     del batch_testing_dataset, batch_results
#     torch.cuda.empty_cache()
    
#     print(f"Completed batch {batch_idx+1}/{num_batches}, saved to {batch_filename}")


# In[ ]:


# def combine_batch_results(results_dir):
#     """Combine all batch results into a single dictionary"""
#     combined_results = {}
#     batch_files = [f for f in os.listdir(results_dir) if f.startswith("test_results_batch_") and f.endswith(".pickle")]
    
#     for batch_file in sorted(batch_files):
#         print("Processing batch file" + batch_file)
#         with open(os.path.join(results_dir, batch_file), "rb") as f:
#             batch_results = pickle.load(f)
#             combined_results.update(batch_results)
    
#     return combined_results

# # Example usage of the helper function:

# test_results = combine_batch_results(intermediate_results_dir)
# with open(os.path.join(path_save_folder, "test_results.pickle"), "wb") as f:
#     pickle.dump(test_results, f)


# ## 2. Initial analysis

# In[ ]:


# Loss testing
loss_testing = nse(df_results=test_results, average=False)
df_NSE = pd.DataFrame(data={"basin_id": test_results.keys(), "NSE": np.round(loss_testing, 3)})
df_NSE = df_NSE.set_index("basin_id")
df_NSE.to_csv(os.path.join(test_result_save_path, "NSE_testing.csv"), index=True, header=True)
mean_nse = df_NSE["NSE"].mean()
print(f"Mean NSE across all basins: {mean_nse:.3f}")


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
plt.show()


# In[ ]:


# Plot simulated and observed discharges
basin_to_analyze = "DE210300"

# colorblind friendly palette
color_palette = {"observed": "#377eb8", "simulated": "#4daf4a"}

# (1) Output time window of test dataset period
plt.plot(test_results[basin_to_analyze]["y_obs"], label="observed", color=color_palette["observed"])
plt.plot(test_results[basin_to_analyze]["y_sim"], label="simulated", alpha=0.5, color=color_palette["simulated"])

# # (2) Output custom time window
# start_date = "2019-01-01 01:00:00"
# end_date = "2019-02-01 01:00:00"
# plt.plot(test_results[basin_to_analyze]["y_obs"][start_date:end_date], label="observed", color=color_palette["observed"])
# plt.plot(test_results[basin_to_analyze]["y_sim"][start_date:end_date], label="simulated", alpha=0.5, color=color_palette["simulated"])

# Format plot
plt.xlabel("Date", fontsize=12, fontweight="bold")
plt.ylabel("Discharge [mm/d]", fontsize=12, fontweight="bold")
plt.title(f"Result comparison (basin {basin_to_analyze})", fontsize=16, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(loc="upper right", fontsize=12)
plt.savefig(os.path.join(test_result_save_path, f"Result comparison (basin {basin_to_analyze}).png"), bbox_inches="tight", pad_inches=0)


# In[ ]:




