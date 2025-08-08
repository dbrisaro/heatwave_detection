#!/usr/bin/env python3
"""
Block Bootstrap Simulation Tool (Dual Variables)
==============================================

A simple, standalone script for performing block bootstrap simulations on any dataset.
Takes a CSV with date and two variable columns, performs block bootstrap resampling,
and outputs CSV files for AEP plotting and simulation results for both variables.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from numba import jit
import os
import argparse
from tqdm import tqdm

# ============================================================================
# JIT-OPTIMIZED FUNCTIONS
# ============================================================================

@jit(nopython=True, cache=True)
def calculate_annual_sum_jit(values, min_days=1):
    """Calculate annual sum of values with minimum duration filter"""
    if len(values) == 0:
        return 0
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
    
    # Handle final event
    if current_length >= min_days:
        total_sum += current_sum
    
    return total_sum

@jit(nopython=True, cache=True)
def apply_seasonal_filter_jit(values, seasonal_start_day=90, seasonal_end_day=305):
    """Apply seasonal filter (Apr-Oct ≈ days 90-305)"""
    filtered_values = np.full_like(values, np.nan)
    for i in range(len(values)):
        day_of_year = (i % 365) + 1
        if seasonal_start_day <= day_of_year <= seasonal_end_day:
            filtered_values[i] = values[i]
    return filtered_values

# ============================================================================
# BLOCK BOOTSTRAP FUNCTIONS
# ============================================================================

def vectorized_block_bootstrap(daily_data, n_simulations, block_length=7, window_days=20, days_per_year=365):
    """
    Vectorized block bootstrap that pre-computes all simulation indices
    
    Parameters:
    -----------
    daily_data : pd.DataFrame
        DataFrame with datetime index and variable columns
    n_simulations : int
        Number of bootstrap simulations to perform
    block_length : int
        Length of blocks to sample (default: 7 days)
    window_days : int
        Seasonal window for matching (default: 20 days)
    days_per_year : int
        Number of days per year (default: 365)
    
    Returns:
    --------
    all_simulation_indices : np.ndarray
        Array of shape (n_simulations, days_per_year) with bootstrap indices
    """
    available_years = sorted(daily_data.index.year.unique())
    n_days = len(daily_data)
    
    print("  Pre-computing valid block positions...")
    valid_starts_cache = {}
    
    for day_of_year in range(1, days_per_year + 1, block_length):
        ref_year = available_years[0]
        try:
            center_date = datetime(ref_year, 1, 1) + timedelta(days=day_of_year - 1)
        except:
            center_date = datetime(ref_year, 12, 31)
        
        valid_starts = []
        for year in available_years:
            try:
                year_center = datetime(year, center_date.month, center_date.day)
            except ValueError:
                if center_date.month == 2 and center_date.day == 29:
                    year_center = datetime(year, 2, 28)
                else:
                    continue
            
            window_start = year_center - timedelta(days=window_days//2)
            window_end = year_center + timedelta(days=window_days//2)
            year_mask = (daily_data.index >= window_start) & (daily_data.index <= window_end)
            year_indices = np.where(year_mask)[0]
            
            for start_idx in year_indices:
                end_idx = start_idx + block_length - 1
                if end_idx < n_days and daily_data.index[end_idx] <= window_end:
                    valid_starts.append(start_idx)
        
        valid_starts_cache[day_of_year] = valid_starts
    
    print("  Generating all simulation indices...")
    all_simulation_indices = np.zeros((n_simulations, days_per_year), dtype=int)
    
    for sim in range(n_simulations):
        np.random.seed(sim)
        current_day = 1
        sim_indices = []
        
        while current_day <= days_per_year:
            days_remaining = days_per_year - current_day + 1
            actual_block_size = min(block_length, days_remaining)
            cache_day = ((current_day - 1) // block_length) * block_length + 1
            valid_starts = valid_starts_cache.get(cache_day, list(range(n_days)))
            
            if valid_starts:
                chosen_start = np.random.choice(valid_starts)
                block_indices = list(range(chosen_start, min(chosen_start + actual_block_size, n_days)))
                sim_indices.extend(block_indices)
            else:
                fallback_indices = [current_day % n_days for _ in range(actual_block_size)]
                sim_indices.extend(fallback_indices)
            
            current_day += actual_block_size
        
        all_simulation_indices[sim, :len(sim_indices[:days_per_year])] = sim_indices[:days_per_year]
    
    return all_simulation_indices

def run_bootstrap_simulation_dual(data, variable1, variable2, n_simulations=1000, block_length=7, 
                                window_days=20, seasonal_filter=False, min_days=1, 
                                output_dir='bootstrap_results'):
    """
    Run block bootstrap simulation on a dataset with two concurrent variables
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with datetime index and two variable columns
    variable1 : str
        Name of the first variable column
    variable2 : str
        Name of the second variable column
    n_simulations : int
        Number of bootstrap simulations
    block_length : int
        Length of blocks for bootstrap
    window_days : int
        Seasonal window for matching
    seasonal_filter : bool
        Whether to apply seasonal filter
    min_days : int
        Minimum duration for event counting
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    dict : Dictionary with simulation results and file paths for both variables
    """
    print(f"🚀 Starting dual variable block bootstrap simulation...")
    print(f"   Dataset: {len(data)} days")
    print(f"   Variable 1: {variable1}")
    print(f"   Variable 2: {variable2}")
    print(f"   Simulations: {n_simulations}")
    print(f"   Block length: {block_length} days")
    print(f"   Seasonal filter: {seasonal_filter}")
    
    # Clean data (remove rows where either variable is missing)
    data_clean = data.dropna(subset=[variable1, variable2]).copy()
    print(f"✅ Clean data: {len(data_clean)} days")
    
    # Generate bootstrap indices
    print(f"\n📊 Generating bootstrap indices...")
    all_simulation_indices = vectorized_block_bootstrap(
        data_clean, n_simulations, block_length, window_days
    )
    print(f"✅ Generated {n_simulations} simulation indices")
    
    # Extract variable values
    variable1_values = data_clean[variable1].values.astype(np.float32)
    variable2_values = data_clean[variable2].values.astype(np.float32)
    
    # Run simulations for both variables
    print(f"\n🔄 Running simulations for both variables...")
    simulation_results_1 = []
    simulation_results_2 = []
    simulation_daily_data_1 = []
    simulation_daily_data_2 = []
    
    for sim_idx in tqdm(range(n_simulations), desc="Processing simulations"):
        # Get indices for this simulation
        sim_indices = all_simulation_indices[sim_idx]
        
        # Sample values for both variables
        sim_values_1 = variable1_values[sim_indices]
        sim_values_2 = variable2_values[sim_indices]
        
        # Apply seasonal filter if requested
        if seasonal_filter:
            sim_values_1 = apply_seasonal_filter_jit(sim_values_1)
            sim_values_2 = apply_seasonal_filter_jit(sim_values_2)
        
        # Calculate annual sums for both variables
        annual_sum_1 = calculate_annual_sum_jit(sim_values_1, min_days)
        annual_sum_2 = calculate_annual_sum_jit(sim_values_2, min_days)
        
        simulation_results_1.append(annual_sum_1)
        simulation_results_2.append(annual_sum_2)
        
        # Store daily data for both variables
        simulation_daily_data_1.append(sim_values_1)
        simulation_daily_data_2.append(sim_values_2)
    
    # Convert to arrays
    annual_sums_1 = np.array(simulation_results_1)
    annual_sums_2 = np.array(simulation_results_2)
    daily_data_array_1 = np.array(simulation_daily_data_1)
    daily_data_array_2 = np.array(simulation_daily_data_2)
    
    print(f"✅ Completed {len(annual_sums_1)} simulations")
    print(f"   Variable 1 - Mean: {annual_sums_1.mean():.2f}, Std: {annual_sums_1.std():.2f}")
    print(f"   Variable 2 - Mean: {annual_sums_2.mean():.2f}, Std: {annual_sums_2.std():.2f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results for variable 1
    aep_data_1 = calculate_aep_curve(annual_sums_1)
    aep_df_1 = pd.DataFrame({
        'exceedance_probability': aep_data_1['probability'],
        'annual_sum': aep_data_1['values']
    })
    aep_filename_1 = os.path.join(output_dir, f'aep_curve_{variable1}.csv')
    aep_df_1.to_csv(aep_filename_1, index=False)
    print(f"💾 Saved AEP curve for {variable1}: {aep_filename_1}")
    
    # Save results for variable 2
    aep_data_2 = calculate_aep_curve(annual_sums_2)
    aep_df_2 = pd.DataFrame({
        'exceedance_probability': aep_data_2['probability'],
        'annual_sum': aep_data_2['values']
    })
    aep_filename_2 = os.path.join(output_dir, f'aep_curve_{variable2}.csv')
    aep_df_2.to_csv(aep_filename_2, index=False)
    print(f"💾 Saved AEP curve for {variable2}: {aep_filename_2}")
    
    # Save daily simulation data for variable 1
    daily_df_1 = pd.DataFrame(daily_data_array_1.T, 
                              columns=[f'sim_{i:04d}' for i in range(n_simulations)])
    daily_filename_1 = os.path.join(output_dir, f'daily_simulations_{variable1}.csv')
    daily_df_1.to_csv(daily_filename_1, index=False)
    print(f"💾 Saved daily simulations for {variable1}: {daily_filename_1}")
    
    # Save daily simulation data for variable 2
    daily_df_2 = pd.DataFrame(daily_data_array_2.T, 
                              columns=[f'sim_{i:04d}' for i in range(n_simulations)])
    daily_filename_2 = os.path.join(output_dir, f'daily_simulations_{variable2}.csv')
    daily_df_2.to_csv(daily_filename_2, index=False)
    print(f"💾 Saved daily simulations for {variable2}: {daily_filename_2}")
    
    # Save combined summary statistics
    summary_df = pd.DataFrame({
        'variable': [variable1, variable1, variable1, variable1, variable1, variable1, variable1,
                    variable2, variable2, variable2, variable2, variable2, variable2, variable2],
        'statistic': ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75'] * 2,
        'value': [
            annual_sums_1.mean(), annual_sums_1.std(), annual_sums_1.min(), annual_sums_1.max(),
            np.median(annual_sums_1), np.percentile(annual_sums_1, 25), np.percentile(annual_sums_1, 75),
            annual_sums_2.mean(), annual_sums_2.std(), annual_sums_2.min(), annual_sums_2.max(),
            np.median(annual_sums_2), np.percentile(annual_sums_2, 25), np.percentile(annual_sums_2, 75)
        ]
    })
    summary_filename = os.path.join(output_dir, f'summary_stats_combined.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"💾 Saved combined summary statistics: {summary_filename}")
    
    # Save correlation analysis
    correlation_df = pd.DataFrame({
        'metric': ['correlation', 'covariance'],
        'value': [
            np.corrcoef(annual_sums_1, annual_sums_2)[0, 1],
            np.cov(annual_sums_1, annual_sums_2)[0, 1]
        ]
    })
    correlation_filename = os.path.join(output_dir, f'correlation_analysis.csv')
    correlation_df.to_csv(correlation_filename, index=False)
    print(f"💾 Saved correlation analysis: {correlation_filename}")
    
    return {
        'annual_sums_1': annual_sums_1,
        'annual_sums_2': annual_sums_2,
        'daily_data_1': daily_data_array_1,
        'daily_data_2': daily_data_array_2,
        'aep_file_1': aep_filename_1,
        'aep_file_2': aep_filename_2,
        'daily_file_1': daily_filename_1,
        'daily_file_2': daily_filename_2,
        'summary_file': summary_filename,
        'correlation_file': correlation_filename
    }

def calculate_aep_curve(values):
    """Calculate Annual Exceedance Probability curve"""
    if len(values) == 0:
        return {'values': [], 'probability': []}
    
    values_sorted = np.sort(values)[::-1]  # Sort descending
    exceedance_prob = np.arange(1, len(values_sorted) + 1) / len(values_sorted)
    
    return {
        'values': values_sorted,
        'probability': exceedance_prob
    }

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run dual variable block bootstrap simulation"""
    parser = argparse.ArgumentParser(description='Dual Variable Block Bootstrap Simulation Tool')
    parser.add_argument('input_file', help='Input CSV file with date and two variable columns')
    parser.add_argument('variable1', help='Name of the first variable column')
    parser.add_argument('variable2', help='Name of the second variable column')
    parser.add_argument('--date_column', default='date', help='Name of the date column (default: date)')
    parser.add_argument('--n_simulations', type=int, default=1000, help='Number of simulations (default: 1000)')
    parser.add_argument('--block_length', type=int, default=7, help='Block length in days (default: 7)')
    parser.add_argument('--window_days', type=int, default=20, help='Seasonal window in days (default: 20)')
    parser.add_argument('--seasonal_filter', action='store_true', help='Apply seasonal filter (Apr-Oct)')
    parser.add_argument('--min_days', type=int, default=1, help='Minimum days for event counting (default: 1)')
    parser.add_argument('--output_dir', default='bootstrap_results', help='Output directory (default: bootstrap_results)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📁 Loading data from: {args.input_file}")
    try:
        data = pd.read_csv(args.input_file)
        print(f"✅ Loaded {len(data)} rows")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Check required columns
    if args.date_column not in data.columns:
        print(f"❌ Date column '{args.date_column}' not found. Available columns: {list(data.columns)}")
        return
    
    if args.variable1 not in data.columns:
        print(f"❌ Variable 1 column '{args.variable1}' not found. Available columns: {list(data.columns)}")
        return
    
    if args.variable2 not in data.columns:
        print(f"❌ Variable 2 column '{args.variable2}' not found. Available columns: {list(data.columns)}")
        return
    
    # Convert date column
    data[args.date_column] = pd.to_datetime(data[args.date_column])
    data = data.set_index(args.date_column)
    
    # Sort by date
    data = data.sort_index()
    
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    print(f"📊 Variable 1 '{args.variable1}' range: {data[args.variable1].min():.2f} to {data[args.variable1].max():.2f}")
    print(f"📊 Variable 2 '{args.variable2}' range: {data[args.variable2].min():.2f} to {data[args.variable2].max():.2f}")
    
    # Run simulation
    results = run_bootstrap_simulation_dual(
        data=data,
        variable1=args.variable1,
        variable2=args.variable2,
        n_simulations=args.n_simulations,
        block_length=args.block_length,
        window_days=args.window_days,
        seasonal_filter=args.seasonal_filter,
        min_days=args.min_days,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 Dual variable simulation complete!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"   📊 AEP curves: {results['aep_file_1']}, {results['aep_file_2']}")
    print(f"   📅 Daily simulations: {results['daily_file_1']}, {results['daily_file_2']}")
    print(f"   📈 Summary statistics: {results['summary_file']}")
    print(f"   🔗 Correlation analysis: {results['correlation_file']}")

if __name__ == "__main__":
    main() 