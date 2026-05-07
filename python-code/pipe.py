import sys, random, glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import pandas as pd

sys.path.insert(0, '.')

from simulation import *
import helper

random.seed(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, '..', 'output')



def run_batch_means(filename, rule, warmup, batch_size, n_batches, seed=0):
    W_total = warmup + n_batches * batch_size

    sim = Simulation(filename, W_total, 1, rule)
    sim.setWeekSchedule()
    #the run with normal vars (non antithetic)
    sim.resetSystem()
    random.seed(seed)
    helper.USE_ANTITHETIC=False
    sim.runOneSimulation()

    # extract post-warmup weekly values
    sl = slice(warmup, warmup + n_batches * batch_size)
    weekly1 = {
        'elAppWT':  np.array(sim.movingAvgElectiveAppWT[sl]),
        'elScanWT': np.array(sim.movingAvgElectiveScanWT[sl]),
        'urScanWT': np.array(sim.movingAvgUrgentScanWT[sl]),
        'OT':       np.array(sim.movingAvgOT[sl]),
    }

    #the run with normal vars (non antithetic)
    sim.resetSystem()
    random.seed(seed)
    helper.USE_ANTITHETIC=True
    sim.runOneSimulation()

    # extract post-warmup weekly values
    sl = slice(warmup, warmup + n_batches * batch_size)
    weekly2 = {
        'elAppWT':  np.array(sim.movingAvgElectiveAppWT[sl]),
        'elScanWT': np.array(sim.movingAvgElectiveScanWT[sl]),
        'urScanWT': np.array(sim.movingAvgUrgentScanWT[sl]),
        'OT':       np.array(sim.movingAvgOT[sl]),
    }

    #weekly = {(weekly1[k] + weekly2[k])/2 for k in weekly1}  # sum the two runs to get antithetic estimates
    weekly = {k: (weekly1[k] + weekly2[k]) / 2 for k in weekly1}

    
    # cut into batches and average each
    batches = {}
    for key, arr in weekly.items():
        batches[key] = arr.reshape(n_batches, batch_size).mean(axis=1)
    batches['OV'] = batches['elAppWT'] / 168 + batches['urScanWT'] / 9

    # summarise: mean + 95% CI per metric
    summary = {}
    for key, arr in batches.items():
        mean = arr.mean()
        se   = arr.std(ddof=1) / np.sqrt(n_batches)
        hw   = stats.t.ppf(0.975, df=n_batches - 1) * se
        summary[key] = {'mean': mean, 'hw': hw, 'lower': mean - hw, 'upper': mean + hw}

    return summary, batches, weekly, weekly1, weekly2


def save_results(filename, rule, summary, weekly):
    tag = os.path.splitext(os.path.basename(filename))[0].replace('input-', '')
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_path = os.path.join(OUT_DIR, f'summary-{tag}-rule{rule}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{'metric':<12} {'mean':>10} {'hw':>10} {'lower':>10} {'upper':>10}\n")
        f.write("-" * 55 + "\n")
        for metric, v in summary.items():
            f.write(f"{metric:<12} {v['mean']:>10.4f} {v['hw']:>10.4f} "
                    f"{v['lower']:>10.4f} {v['upper']:>10.4f}\n")

    output_path = os.path.join(OUT_DIR, f'output-{tag}-rule{rule}.csv')
    pd.DataFrame(weekly).rename_axis('week').to_csv(output_path, float_format='%.6f')


warmup     = 100
batch_size = 65
n_batches  = 600
rules      = [1, 2, 3, 4]

input_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '..', 'input', 'generated_input_files', 'input-*.txt')))

for filename in input_files:
    for rule in rules:
        print(f"Running {os.path.basename(filename)}, rule {rule} ...")
        summary, batches, weekly, weekly1, weekly2 = run_batch_means(
            filename   = filename,
            rule       = rule,
            warmup     = warmup,
            batch_size = batch_size,
            n_batches  = n_batches,
        )
        save_results(filename, rule, summary, weekly)