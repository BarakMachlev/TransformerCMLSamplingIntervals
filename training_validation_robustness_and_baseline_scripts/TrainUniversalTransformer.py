import sys
sys.path.insert(0, "/home/lucy3/BarakMachlev/PyNNcml")
import numpy as np
import pynncml as pnc
import torch
import math
import os
import torch.nn as nn
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Subset
from io import StringIO
from scipy.signal import savgol_filter

def lr_schedule(epoch):
    if epoch < 600:
        return 1.0        # relative to base LR = 1e-4 → stays 1e-4
    elif epoch < 800:
        return 0.5        # 0.5 × 1e-4 = 5e-5
    else:
       return 0.1        # 0.1 × 1e-4 = 1e-5

xy_min = [1.29e6, 0.565e6]  # Link Region
xy_max = [1.34e6, 0.5875e6]
time_slice = slice("2015-06-01", "2015-08-31")  # Time Interval

# Set output directory (lab computer path)
base_output_dir = "/home/lucy3/BarakMachlev/Thesis/Article_Results/UT/openMRG_DataSet/Att_NewLoss/Set_1"
output_dir = base_output_dir
os.makedirs(output_dir, exist_ok=True)

dataset = pnc.datasets.loader_open_mrg_dataset(restriction_minimum_length=0.75,
                                            xy_min = xy_min,
                                            xy_max = xy_max,
                                            time_slice = time_slice)

rg = np.stack([p.data_array for p in dataset.point_set]).flatten()
param = scipy.stats.expon.fit(rg)
exp_gamma = param[1]

dynamic_input_size = 90 #90 # 90 RSL and 90 TSL
batch_size = 16
lr = 1e-4  # @param{type:"number"}
weight_decay = 1e-4  # @param{type:"number"}
n_epochs = 1000  # @param{type:"integer"}
window_size = 32
metadata_n_features = 16
protocol_n_features = 16
metadata_input_size = 2
d_model = 256
dropout = 0.1
num_encoder_layers = 4
h = 8
num_protocols = 12

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)
if device.type == "cuda":
    print("  - GPU Name:", torch.cuda.get_device_name(0))
else:
    print("  - Running on CPU")

# Predefined validation sets
validation_set_1 = [0, 4, 10, 12, 18, 22, 28, 30, 55, 33, 35, 45, 49, 67, 68, 70]
validation_set_2 = [5, 9, 16, 34, 39, 40, 42, 47, 54, 31, 56, 61, 62, 64, 73, 79]
validation_set_3 = [3, 6, 7, 8, 13, 17, 19, 25, 36, 38, 44, 50, 53, 65, 66, 76]
validation_set_4 = [11, 15, 24, 26, 21, 32, 41, 43, 46, 48, 57, 58, 72, 75, 77, 78]
validation_set_5 = [1, 2, 14, 20, 27, 23, 29, 37, 51, 52, 59, 60, 63, 69, 71, 74]

# Choose which validation set to use (1 to 5)
fold = 1  # 👈 Change this to 2, 3, 4, or 5 for other runs

val_phys = eval(f"validation_set_{fold}")
train_phys = sorted(list(set(range(80)) - set(val_phys)))

assert len(val_phys) == 16
assert len(train_phys) == 64
assert set(train_phys).isdisjoint(set(val_phys))

# --- build indices in the *expanded* link_list using link_index ---
val_indices = [i for i, lnk in enumerate(dataset.link_set.link_list) if lnk.link_index in val_phys]
train_indices = [i for i, lnk in enumerate(dataset.link_set.link_list) if lnk.link_index in train_phys]

# Make sure every physical link appears exactly 12 times
from collections import Counter
counts = Counter([lnk.link_index for lnk in dataset.link_set.link_list])
assert all(v == 12 for v in counts.values()), "Some physical links do not have 12 protocol variants"

# Make sure no leakage
assert set(val_phys).isdisjoint(set(train_phys)), "Leakage between train and validation"

# sanity: expect 12 variants per physical link
print(f"Total items: {len(dataset.link_set.link_list)}")
print(f"Train items: {len(train_indices)}  (expected ~ {64*12})")
print(f"Val items:   {len(val_indices)}    (expected ~ {16*12})")

training_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, val_indices)

data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


normalization_cfg=pnc.training_helpers.compute_attenuation_data_normalization(data_loader, network_dynamic_input_size = dynamic_input_size) # Compute the normalization statistics from the training dataset.
#normalization_cfg=pnc.training_helpers.compute_data_normalization(data_loader, network_dynamic_input_size = dynamic_input_size) # Compute the normalization statistics from the training dataset.
model = pnc.scm.rain_estimation.two_step_network_with_attention(normalization_cfg = normalization_cfg,
                                                                dynamic_input_size = dynamic_input_size,
                                                                metadata_input_size = metadata_input_size,
                                                                d_model = d_model,
                                                                protocol_n_features = protocol_n_features,
                                                                metadata_n_features = metadata_n_features,
                                                                num_protocols = num_protocols,
                                                                window_size = window_size,
                                                                dropout = dropout,
                                                                num_encoder_layers = num_encoder_layers,
                                                                h = h).to(device)

norm_path = os.path.join(base_output_dir, "normalization_cfg.pth")
torch.save(normalization_cfg, norm_path)
print(f"✅ Saved normalization_cfg to: {norm_path}")

# Original Hai optimizer - The lr here is a dummy starting point — it will be overridden by the scheduler.
opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_schedule)

class RegressionLoss(torch.nn.Module):
    def __init__(self, in_gamma, gamma_s=0.9):
        super(RegressionLoss, self).__init__()
        self.in_gamma = in_gamma
        self.gamma_s = gamma_s

    #def forward(self, input, target):
    #    delta = (target - input) ** 2
    #    w = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)
    #    return torch.sum(torch.mean(w * delta, dim=0))
    def forward(self, input, target, protocol_id, protocol_log_precision):
        delta = (target - input) ** 2
        w_r = 1 - self.gamma_s * torch.exp(-self.in_gamma * target)

        s_p = protocol_log_precision[protocol_id.long()]   # [B]
        w_p = torch.exp(s_p).unsqueeze(1)                  # [B,1]

        loss = 0.5 * w_r * w_p * delta - 0.5 * s_p.unsqueeze(1)

        return torch.sum(torch.mean(loss, dim=0))

from pynncml.metrics.results_accumlator import ResultsAccumulator, AverageMetric, GroupAnalysis

# Initialize metrics and results accumulators
ra = ResultsAccumulator()
am = AverageMetric()

# ADD: validation accumulators
ra_val = ResultsAccumulator()
am_val = AverageMetric()

model_path = os.path.join(output_dir, "trained_model.pth")

if os.path.exists(model_path):
    # Load model if weights already exist
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from: {model_path} — skipping training and loss plotting")

else:

    print("🟡 No saved weights found — starting training")

    loss_function_rain_est = RegressionLoss(exp_gamma)
    loss_function_wet_dry = torch.nn.BCELoss()

    model.eval()
    # Find lambda values such that at initialization both loss will be equal:
    loss_est = 0
    loss_detection = 0
    with torch.no_grad():
        for rain_rate, attenuation, metadata, protocol_id in data_loader:
        #for rain_rate, rsl, tsl, metadata, protocol_id in data_loader:
            m_step = math.floor(rain_rate.shape[1] / window_size)
            for step in range(m_step):
                _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                _att = attenuation[:, step * window_size:(step + 1) * window_size, :].to(device)  # [B, W, 90]
                #_rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                #_tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)                
                rain_estimation_detection = model(
                    _att,  # no concat anymore
                    #torch.cat([_rsl, _tsl], dim=-1),  # [B, W, 180]
                    metadata.to(device),
                    protocol_id.to(device),
                )
                rain_hat = rain_estimation_detection[:, :, 0]
                rain_detection = rain_estimation_detection[:, :, 1]

                #loss_est += loss_function_rain_est(rain_hat, _rr)
                loss_est += loss_function_rain_est(
                rain_hat,
                _rr,
                protocol_id.to(device),
                model.protocol_log_precision
                )
                loss_detection += loss_function_wet_dry(rain_detection, (_rr > 0.1).float())
    lambda_value = loss_detection / loss_est

    steps_counter=0

    # Train model if weights do not exist
    model.train()
    for epoch in tqdm(range(n_epochs)): # Repeat the whole training process again
        am.clear()
        for rain_rate, attenuation, metadata, protocol_id in data_loader: # for loop for true batches - each batch has batch_size links
        #for rain_rate, rsl, tsl, metadata, protocol_id in data_loader:    
            m_step = math.floor(rain_rate.shape[1] / window_size)
            for step in range(m_step): # for loop for sliding windows in time (a.k.a mini-batches / chunks)
                opt.zero_grad()  # Zero gradients
                # Perform sliding window in the CML time series.
                _rr = rain_rate[:, step * window_size:(step + 1) * window_size].float().to(device)
                _att = attenuation[:, step * window_size:(step + 1) * window_size, :].to(device)  # [B, W, 90]
                #_rsl = rsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                #_tsl = tsl[:, step * window_size:(step + 1) * window_size, :].to(device)
                # Forward pass of the model
                rain_estimation_detection = model(
                    _att,  # no concat anymore
                    #torch.cat([_rsl, _tsl], dim=-1),  # [B, W, 180]
                    metadata.to(device),
                    protocol_id.to(device),
                )
                rain_hat = rain_estimation_detection[:, :, 0]
                rain_detection = rain_estimation_detection[:, :, 1]
                # Compute loss function
                #loss_est = loss_function_rain_est(rain_hat, _rr)
                loss_est = loss_function_rain_est(
                    rain_hat,
                    _rr,
                    protocol_id.to(device),
                    model.protocol_log_precision
                )
                loss_detection = loss_function_wet_dry(rain_detection, (_rr > 0.1).float())
                loss = lambda_value * loss_est + loss_detection
                # Take the derivative w.r.t. model parameters $\Theta$
                loss.backward()
                opt.step()
                steps_counter += 1
                am.add_results(loss=loss.item(),
                                loss_est=loss_est.item(),
                                loss_detection=loss_detection.item())  # Log results to average.

        scheduler.step()        
        ra.add_results(loss=am.get_results("loss"),
                        loss_est=am.get_results("loss_est"),
                        loss_detection=am.get_results("loss_detection"))

        ##############################################
        # ----- VALIDATION for this epoch -----
        ##############################################
        model.eval()
        am_val.clear()
        with torch.no_grad():
            for rain_rate_v, attenuation_v, metadata_v, protocol_id_v in val_loader:
            #for rain_rate_v, rsl_v, tsl_v, metadata_v, protocol_id_v in val_loader:
                m_step_v = math.floor(rain_rate_v.shape[1] / window_size)
                for step_v in range(m_step_v):
                    _rr_v  = rain_rate_v[:, step_v*window_size:(step_v+1)*window_size].float().to(device)
                    _att_v = attenuation_v[:, step_v*window_size:(step_v+1)*window_size, :].to(device)  # [B, W, 90]
                    #_rsl_v = rsl_v[:, step_v*window_size:(step_v+1)*window_size, :].to(device)
                    #_tsl_v = tsl_v[:, step_v*window_size:(step_v+1)*window_size, :].to(device)

                    out_v = model(
                                _att_v,
                                #torch.cat([_rsl_v, _tsl_v], dim=-1),
                                metadata_v.to(device),
                                protocol_id_v.to(device),
                            )
                    rain_hat_v       = out_v[:, :, 0]
                    rain_detection_v = out_v[:, :, 1]

                    #loss_est_v = loss_function_rain_est(rain_hat_v, _rr_v)
                    loss_est_v = loss_function_rain_est(
                        rain_hat_v,
                        _rr_v,
                        protocol_id_v.to(device),
                        model.protocol_log_precision
                    )
                    loss_det_v = loss_function_wet_dry(rain_detection_v, (_rr_v > 0.1).float())
                    loss_v     = lambda_value * loss_est_v + loss_det_v

                    am_val.add_results(loss=loss_v.item(),
                                        loss_est=loss_est_v.item(),
                                        loss_detection=loss_det_v.item())

        # Store validation epoch means
        ra_val.add_results(loss=am_val.get_results("loss"),
                        loss_est=am_val.get_results("loss_est"),
                        loss_detection=am_val.get_results("loss_detection"))
        model.train()  # back to train mode for next epoch
        ##############################################
        # ----- END of VALIDATION for this epoch -----
        ##############################################

    
    # Save trained weights
    torch.save(model.state_dict(), model_path)
    print(f"✅ Weights saved to: {model_path}")

    plt.plot(ra.get_results("loss"), label="Total Loss")
    plt.plot(ra.get_results("loss_est"), label="Rain Rate Loss")
    plt.plot(ra.get_results("loss_detection"), label="Wet/Dry Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss per Epoch (Fold {fold})")
    figure_name = f"loss_plot_over_epochs_fold_{fold}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # 1) Wet/Dry (classification) — train vs val
    plt.figure()
    plt.plot(ra.get_results("loss_detection"),     label="Train Wet/Dry Loss")
    plt.plot(ra_val.get_results("loss_detection"), label="Val Wet/Dry Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Wet/Dry Loss per Epoch (Fold {fold})")
    figure_name = f"wet_dry_loss_train_vs_val_fold_{fold}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # 2) Rain rate (regression) — train vs val
    plt.figure()
    plt.plot(ra.get_results("loss_est"),     label="Train Rain-Rate Loss")
    plt.plot(ra_val.get_results("loss_est"), label="Val Rain-Rate Loss")
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Rain-Rate Loss per Epoch (Fold {fold})")
    figure_name = f"rain_rate_loss_train_vs_val_fold_{fold}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    print("-----------------------------------------------")
    print(steps_counter)
    print("-----------------------------------------------")




    PROTOCOL_MAP = {
        ("instantaneous", 900): 0,
        ("instantaneous", 450): 1,
        ("instantaneous", 300): 2,
        ("instantaneous", 180): 3,
        ("instantaneous", 150): 4,
        ("instantaneous", 100): 5,
        ("instantaneous", 90): 6,
        ("instantaneous", 60): 7,
        ("instantaneous", 50): 8,
        ("instantaneous", 30): 9,
        ("instantaneous", 20): 10,
        ("instantaneous", 10): 11,
    }

    # get learned precision
    s_p = model.protocol_log_precision.detach().cpu()
    precision = torch.exp(s_p)

    save_path = os.path.join(output_dir, "protocol_precision.txt")

    with open(save_path, "w") as f:
        f.write("protocol_id | protocol | precision\n")

        # sort by protocol_id
        for key, idx in sorted(PROTOCOL_MAP.items(), key=lambda x: x[1]):
            f.write(f"{idx} | {str(key)} | {precision[idx].item()}\n")

    print(f"✅ Protocol precision table saved to: {save_path}")

