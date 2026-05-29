import numpy as np
import pynncml as pnc
import torch
import math
import os
import sys
sys.path.insert(0, "/home/lucy3/BarakMachlev/PyNNcml")

import torch.nn as nn
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Subset
from io import StringIO
from scipy.signal import savgol_filter

xy_min = [1.29e6, 0.565e6]  # Link Region
xy_max = [1.34e6, 0.5875e6]
time_slice = slice("2015-06-01", "2015-08-31")  # Time Interval

combinations = [("instantaneous", sec) for sec in [10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900]]
#combinations.append(("min_max", None))
#combinations.append(("average", None))

for samples_type, sampling_interval_in_sec in combinations:
    
    if samples_type in ("min_max", "average"):
        sampling_interval_in_sec = 900
    
    # Set output directory based on sampling configuration (lab computer path)
    base_output_dir = "/home/lucy3/BarakMachlev/Thesis/Article_Results/UT/openMRG_DataSet/Att_NewLoss/Set_1"
    if samples_type == "instantaneous":
        output_dir = os.path.join(base_output_dir, f"Instantaneous_{sampling_interval_in_sec}_sec")
    elif samples_type == "min_max":
        output_dir = os.path.join(base_output_dir, "Max_Min")
    elif samples_type == "average":
        output_dir = os.path.join(base_output_dir, "Average")

    os.makedirs(output_dir, exist_ok=True)

    dataset = pnc.datasets.loader_open_mrg_dataset(restriction_minimum_length=0.75,
                                            xy_min = xy_min,
                                            xy_max = xy_max,
                                            time_slice = time_slice,
                                            samples_type = samples_type,
                                            sampling_interval_in_sec = sampling_interval_in_sec)

    rg = np.stack([p.data_array for p in dataset.point_set]).flatten()
    param = scipy.stats.expon.fit(rg)
    exp_gamma = param[1]

    dynamic_input_size = 90 # 90 RSL and 90 TSL
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

    
    # Confirm we have exactly 80 links
    assert len(dataset.link_set.link_list) == 80, f"Expected 80 links, got {len(dataset.link_set.link_list)}"

    # Predefined validation sets
    validation_set_1 = [0, 4, 10, 12, 18, 22, 28, 30, 33, 35, 45, 49, 55, 67, 68, 70]
    validation_set_2 = [5, 9, 16, 31, 34, 39, 40, 42, 47, 54, 56, 61, 62, 64, 73, 79]
    validation_set_3 = [3, 6, 7, 8, 13, 17, 19, 25, 36, 38, 44, 50, 53, 65, 66, 76]
    validation_set_4 = [11, 15, 21, 24, 26, 32, 41, 43, 46, 48, 57, 58, 72, 75, 77, 78]
    validation_set_5 = [1, 2, 14, 20, 23, 27, 29, 37, 51, 52, 59, 60, 63, 69, 71, 74]

    # Choose fold
    fold = 1  # 1..5
    val_indices = eval(f"validation_set_{fold}")

    # Derive training set as complement
    all_indices = set(range(80))
    train_indices = sorted(list(all_indices - set(val_indices)))

    # Sanity checks
    assert len(val_indices) == 16
    assert len(train_indices) == 64
    assert set(train_indices).isdisjoint(set(val_indices)), "❌ Overlap between training and validation sets!"

    # Create subsets
    training_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)

    # DataLoaders (deterministic)
    data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    val_loader  = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ Fold {fold} | Train set: {len(train_indices)} links | Val set: {len(val_indices)} links")

    norm_path = os.path.join(base_output_dir, "normalization_cfg.pth")
    normalization_cfg = torch.load(norm_path, map_location="cpu", weights_only=False)

    model = pnc.scm.rain_estimation.two_step_network_with_attention(normalization_cfg=normalization_cfg,
                                                                dynamic_input_size = dynamic_input_size,
                                                                metadata_input_size = metadata_input_size,
                                                                d_model = d_model,
                                                                #protocol_n_features = protocol_n_features,
                                                                metadata_n_features = metadata_n_features,
                                                                #num_protocols = num_protocols,
                                                                window_size = window_size,
                                                                dropout = dropout,
                                                                num_encoder_layers = num_encoder_layers,
                                                                h = h).to(device)

    # Original Hai optimizer - The lr here is a dummy starting point — it will be overridden by the scheduler.
    opt = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_schedule)

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

    model_path = os.path.join(base_output_dir, "trained_model.pth")

    if os.path.exists(model_path):
        # Load model if weights already exist
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded from: {model_path} — skipping training and loss plotting")

    else:
        raise FileNotFoundError(f"❌ No saved weights found at: {model_path}")



    print("📊 Evaluating model...")
    model.eval()
    #ema_model.eval()
    ga = GroupAnalysis()

    with torch.no_grad():
        #for rain_rate, rsl, tsl, metadata, protocol_id in val_loader:
        for rain_rate, attenuation, metadata, protocol_id in val_loader:
            m_step = math.floor(rain_rate.shape[1] / window_size)
            am.clear()
            rain_ref_list = []
            rain_hat_list = []
            detection_list = []

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
                rain_detection = rain_estimation_detection[:, :, 1]
                rain_hat = rain_estimation_detection[:, :, 0] * torch.round(rain_detection)  # Rain Rate is computed only for wet samples
                rain_hat_list.append(rain_hat.detach().cpu().numpy())
                rain_ref_list.append(_rr.detach().cpu().numpy())
                ga.append(rain_ref_list[-1], rain_hat_list[-1])
                detection_list.append(torch.round(rain_detection).detach().cpu().numpy())
                delta = rain_hat.squeeze(dim=-1) - _rr
                bias = torch.mean(delta)
                mse = torch.mean(delta ** 2)
                am.add_results(bias=bias.item(), mse=mse.item())
    actual = np.concatenate(detection_list).flatten()
    predicted = (np.concatenate(rain_ref_list) > 0.1).astype("float").flatten()
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    max_rain = np.max(np.concatenate(rain_ref_list))
    g_array = np.linspace(0, max_rain, 6)

    print("Results Detection:")
    print("Validation Results of Two-Step RNN")
    print("Accuracy[%]:", 100 * (np.sum(actual == predicted) / actual.size))
    print("F1 Score:", metrics.f1_score(actual, predicted))

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])

    cm_display.plot()
    plt.title(f"Confusion Matrix ({samples_type} Sampling)")
    figure_name = f"confusion_matrix_{samples_type}.png"
    save_path = os.path.join(output_dir, figure_name)
    plt.savefig(save_path)
    print(f"✅ Figure saved to {save_path}")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    results_path = os.path.join(output_dir, f"Estimation_Results_{samples_type}.txt")

    # Redirect stdout to capture printed PrettyTable
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    print("Results Estimation:")
    _ = ga.run_analysis(np.stack([g_array[:-1], g_array[1:]], axis=-1))

    # Restore normal stdout
    sys.stdout = old_stdout

    # Write captured output to file
    with open(results_path, "w") as f:
        f.write(mystdout.getvalue())

    print(f"✅ Results summary saved to: {results_path}")
