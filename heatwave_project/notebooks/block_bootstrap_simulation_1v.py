import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from numba import jit
from tqdm import tqdm
import os

@jit(nopython=True, cache=True)
def calculate_annual_sum_jit(values, min_days=1):
    total_sum = 0
    current_length = 0
    current_sum = 0
    for i in range(len(values)):
        if not np.isnan(values[i]):
            current_length += 1
            current_sum += values[i]
        else:
            if current_length >= min_days:
                total_sum += current_sum
            current_length = 0
            current_sum = 0
    if current_length >= min_days:
        total_sum += current_sum
    return total_sum

def vectorized_block_bootstrap(daily_data, n_simulations, block_length=7, window_days=20):
    available_years = sorted(daily_data.index.year.unique())
    n_days = len(daily_data)
    valid_starts_cache = {}
    for day_of_year in range(1, 366, block_length):
        ref_year = available_years[0]
        center_date = datetime(ref_year, 1, 1) + timedelta(days=day_of_year - 1)
        valid_starts = []
        for year in available_years:
            window_start = datetime(year, center_date.month, center_date.day) - timedelta(days=window_days//2)
            window_end = window_start + timedelta(days=window_days)
            year_mask = (daily_data.index >= window_start) & (daily_data.index <= window_end)
            year_indices = np.where(year_mask)[0]
            for s in year_indices:
                e = s + block_length - 1
                if e < n_days:
                    valid_starts.append(s)
        valid_starts_cache[day_of_year] = valid_starts
    all_indices = np.zeros((n_simulations, 365), dtype=int)
    for sim in range(n_simulations):
        np.random.seed(sim)
        current_day = 1
        sim_indices = []
        while current_day <= 365:
            block = min(block_length, 365 - current_day + 1)
            cache_day = ((current_day - 1) // block_length) * block_length + 1
            starts = valid_starts_cache.get(cache_day, list(range(n_days)))
            if starts:
                start = np.random.choice(starts)
                sim_indices.extend(range(start, start + block))
            current_day += block
        all_indices[sim, :] = sim_indices[:365]
    return all_indices

def calculate_aep_curve(values):
    values_sorted = np.sort(values)[::-1]
    prob = np.arange(1, len(values_sorted)+1) / len(values_sorted)
    return pd.DataFrame({'exceedance_probability': prob, 'annual_sum': values_sorted})

def run_bootstrap_simulation_single(data, variable, n_simulations=1000, block_length=7, window_days=20, output_dir="bootstrap_single"):
    data_clean = data.dropna(subset=[variable]).copy()
    all_sim_indices = vectorized_block_bootstrap(data_clean, n_simulations, block_length, window_days)
    values = data_clean[variable].values.astype(np.float32)
    annual_sums, daily_data = [], []
    for sim in tqdm(range(n_simulations)):
        idx = all_sim_indices[sim]
        sim_values = values[idx]
        annual_sums.append(calculate_annual_sum_jit(sim_values))
        daily_data.append(sim_values)
    os.makedirs(output_dir, exist_ok=True)
    aep = calculate_aep_curve(np.array(annual_sums))
    aep.to_csv(f"{output_dir}/aep_curve_{variable}.csv", index=False)
    pd.DataFrame(np.array(daily_data).T, columns=[f"sim_{i:04d}" for i in range(n_simulations)]).to_csv(
        f"{output_dir}/daily_simulations_{variable}.csv", index=False)
    pd.DataFrame({
        "stat": ["mean","std","min","max","median","q25","q75"],
        "value": [np.mean(annual_sums), np.std(annual_sums), np.min(annual_sums),
                  np.max(annual_sums), np.median(annual_sums),
                  np.percentile(annual_sums,25), np.percentile(annual_sums,75)]
    }).to_csv(f"{output_dir}/summary_stats_{variable}.csv", index=False)
    print(f"✅ Completed {n_simulations} simulations for {variable}")

# Ejemplo de uso:
# data = pd.read_csv("data.csv", parse_dates=["date"]).set_index("date")
# run_bootstrap_simulation_single(data, "tmax")
