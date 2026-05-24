import os
import sys
from io import StringIO
from types import SimpleNamespace

sys.path.insert(0, "/Users/barakmachlev/Projects/PyNNcml")

import numpy as np
import torch
import pynncml as pnc

from tqdm import tqdm
from sklearn import metrics

from pynncml.metrics.results_accumlator import GroupAnalysis
from pynncml.single_cml_methods.power_law import PowerLawType
from pynncml.single_cml_methods.rain_estimation import two_step_constant_baseline


xy_min = [1.29e6, 0.565e6]
xy_max = [1.34e6, 0.5875e6]
time_slice = slice("2015-06-01", "2015-08-31")

sampling_intervals = [10, 20, 30, 50, 60, 90, 100, 150, 180, 300, 450, 900]
base_output_dir = "./Article_Results/UT/openMRG_DataSet/ConstantBaseline"

batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

constant_baseline_model = two_step_constant_baseline(
    power_law_type=PowerLawType.INSTANCE,
    r_min=0.1,
    window_size=4,
    threshold=0.1,
    wa_factor=2.0
).to(device)

constant_baseline_model.eval()

summary_rows = []

for sampling_interval_in_sec in sampling_intervals:

    print("===================================================")
    print(f"Running Constant Baseline for {sampling_interval_in_sec} sec")
    print("===================================================")

    output_dir = os.path.join(
        base_output_dir,
        f"Instantaneous_{sampling_interval_in_sec}_sec"
    )
    os.makedirs(output_dir, exist_ok=True)

    dataset = pnc.datasets.loader_open_mrg_dataset(
        restriction_minimum_length=0.75,
        xy_min=xy_min,
        xy_max=xy_max,
        time_slice=time_slice,
        samples_type="instantaneous",
        sampling_interval_in_sec=sampling_interval_in_sec
    )

    assert len(dataset.link_set.link_list) == 80

    full_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    ga = GroupAnalysis()

    all_rain_ref = []
    all_rain_hat = []
    all_detection_pred = []

    with torch.no_grad():

        for rain_rate, attenuation, metadata, protocol_id in tqdm(full_loader):

            rain_rate = rain_rate.float().to(device)
            attenuation = attenuation.float().to(device)
            metadata = metadata.float().to(device)

            stride = sampling_interval_in_sec // 10

            att_token = torch.mean(
                attenuation[:, :, ::stride],
                dim=-1
            )

            batch_rain_hat = []
            batch_detection = []

            for link_idx in range(att_token.shape[0]):

                input_att = att_token[link_idx:link_idx + 1, :]

                frequency = metadata[link_idx, 0].item()  # GHz
                length = metadata[link_idx, 1].item()     # km
                polarization = False

                meta = SimpleNamespace(
                    length=length,
                    frequency=frequency,
                    polarization=polarization
                )

                rain_hat_link, wet_dry_link, _ = constant_baseline_model(
                    input_att,
                    meta
                )

                batch_rain_hat.append(rain_hat_link)
                batch_detection.append(wet_dry_link)

            rain_hat = torch.cat(batch_rain_hat, dim=0)
            rain_detection = torch.cat(batch_detection, dim=0)

            rain_ref_np = rain_rate.detach().cpu().numpy()
            rain_hat_np = rain_hat.detach().cpu().numpy()
            detection_np = rain_detection.detach().cpu().numpy()

            all_rain_ref.append(rain_ref_np)
            all_rain_hat.append(rain_hat_np)
            all_detection_pred.append(detection_np)

            ga.append(rain_ref_np, rain_hat_np)

    rain_ref_all = np.concatenate(all_rain_ref, axis=0)
    rain_hat_all = np.concatenate(all_rain_hat, axis=0)
    detection_pred_all = np.concatenate(all_detection_pred, axis=0)

    detection_true_all = (rain_ref_all > 0.1).astype(float)

    rmse = np.sqrt(np.mean((rain_hat_all - rain_ref_all) ** 2))
    mae = np.mean(np.abs(rain_hat_all - rain_ref_all))
    bias = np.mean(rain_hat_all - rain_ref_all)

    accuracy = 100 * np.mean(
        detection_pred_all.flatten() == detection_true_all.flatten()
    )

    f1 = metrics.f1_score(
        detection_true_all.flatten(),
        detection_pred_all.flatten(),
        zero_division=0
    )

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Bias:", bias)
    print("Accuracy[%]:", accuracy)
    print("F1 Score:", f1)

    max_rain = np.max(rain_ref_all)
    g_array = np.linspace(0, max_rain, 6)

    results_path = os.path.join(
        output_dir,
        "Estimation_Results_ConstantBaseline.txt"
    )

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    print("Results Estimation:")
    _ = ga.run_analysis(
        np.stack([g_array[:-1], g_array[1:]], axis=-1)
    )

    sys.stdout = old_stdout

    with open(results_path, "w") as f:
        f.write(mystdout.getvalue())
        f.write("\n\n")
        f.write(f"Sampling interval [sec]: {sampling_interval_in_sec}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Accuracy[%]: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n")

    summary_rows.append([
        sampling_interval_in_sec,
        rmse,
        mae,
        bias,
        accuracy,
        f1
    ])

summary_path = os.path.join(base_output_dir, "summary_constant_baseline.csv")

with open(summary_path, "w") as f:
    f.write("sampling_interval_sec,rmse,mae,bias,accuracy_percent,f1_score\n")

    for row in summary_rows:
        f.write(",".join([str(x) for x in row]) + "\n")

print("Finished Constant Baseline evaluation")
print(f"Summary saved to: {summary_path}")