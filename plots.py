import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict, List

# These libraries are required for the plotting functions to execute.

def plot_annual_cycles(
    annual_cycles: Dict[str, pd.Series],
    annual_cycles_mod: Dict[str, Dict[int, pd.Series]],
    point_ids: List[int],
    units: Dict[str, str],
    plots_dir: str,
    institution: str,
    RCM: str,
    model: str,
    version: str,
    ensemble: str
) -> None:
    """
    Generates and saves a plot of the annual cycle (monthly mean) comparing
    Observation data (OBS) with time series from model points (CORDEX).

    Parameters:
    ----------
    annual_cycles : Dict[str, pd.Series]
        Annual cycles (monthly mean) of observation data (OBS).
        Key: Variable name, Value: Series with month index (1-12).
    annual_cycles_mod : Dict[str, Dict[int, pd.Series]]
        Annual cycles (monthly mean) of model data (CORDEX).
        Outer Key: Variable name. Inner Key: Point ID (int).
        Value: Series with month index (1-12).
    point_ids : List[int]
        List of model point IDs (e.g., [0, 1, 2, ...]).
    units : Dict[str, str]
        Dictionary mapping variable names to their units.
    plots_dir : str
        Directory where the plot will be saved.
    institution, RCM, model, version, ensemble : str
        Metadata used to name the output file.
    """
    if not annual_cycles:
        print("Warning: No OBS annual cycles to plot.")
        return

    # Check the first observation cycle to see if it only contains data for one month
    first_var_name = next(iter(annual_cycles))
    if len(annual_cycles[first_var_name].index) <= 1:
        print("\n*** CRITICAL WARNING: Annual cycle data appears to contain only ONE month (likely the yearly mean stored at index 1). ***")
        print("To plot a full annual cycle, the input data (annual_cycles and annual_cycles_mod) must contain 12 data points (months 1-12).\n")

    num_vars = len(annual_cycles)
    fig_ann, axes = plt.subplots(num_vars, 1, figsize=(16, 5 * num_vars), sharex=True)

    fig_ann.suptitle(
        "Annual Cycle (Monthly Mean) – Observations vs Model Points",
        fontsize=22, y=1.02
    )

    # Ensures 'axes' is always an iterable list, even if only 1 variable is present
    if num_vars == 1:
        axes = [axes]

    # Generate distinct colors for each model point
    colors = plt.cm.viridis(np.linspace(0, 1, len(point_ids)))

    # Iterate over each variable
    for i, (name, obs_cycle) in enumerate(annual_cycles.items()):
        ax = axes[i]

        # 1. OBS Data Plot (Solid black line)
        ax.plot(
            obs_cycle.index, obs_cycle.values,
            marker='o', color='black', linewidth=2, label="Observations"
        )

        # 2. MODEL Data Plot (Colored dashed lines)
        for j, pid in enumerate(point_ids):
            # Get the cycle for the current variable and point (pid)
            mod_cycle = annual_cycles_mod.get(name, {}).get(pid, None)

            if mod_cycle is None:
                continue

            # Plot model data points connected by a dashed line
            ax.plot(
                mod_cycle.index, mod_cycle.values,
                marker='s', linestyle='--', linewidth=1.3,
                color=colors[j],
                label=f"Model P{pid}"
            )

        # Y-axis configuration
        unit = units.get(name, 'Unit')
        ax.set_ylabel(f"{name} ({unit})", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        # Legend in the upper right corner with 3 columns
        ax.legend(fontsize=10, ncol=3, loc='upper right')

    # X-axis configuration (Common for all)
    axes[-1].set_xticks(range(1, 13))
    axes[-1].set_xticklabels(
        ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        fontsize=12
    )
    axes[-1].set_xlabel("Month", fontsize=16)

    # Final layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjusts for the main title

    # Display the plot
    plt.show()

    # Create directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # File name based on metadata
    filename = f"annual_cycle_all_points_{institution}_{RCM}_{model}_{version}_{ensemble}.png"
    full_path = os.path.join(plots_dir, filename)

    try:
        fig_ann.savefig(full_path, dpi=300)
        print(f"\nSUCCESS: Annual cycle plot saved to {full_path}")
    except Exception as e:
        print(f"\nERROR: Could not save the plot to {full_path}. Error: {e}")


def plot_diurnal_cycles(
    daily_cycles: Dict[str, pd.Series],
    daily_cycles_mod: Dict[str, Dict[int, pd.Series]],
    point_ids: List[int],
    units: Dict[str, str],
    plots_dir: str,
    institution: str,
    RCM: str,
    model: str,
    version: str,
    ensemble: str
) -> None:
    """
    Generates and saves a plot of the diurnal cycle (hourly aggregation) comparing
    Observation data (OBS) with time series from model points (CORDEX).
    
    The plot shows the actual data points connected by a straight line, with 
    zero values removed for precipitation variables to better represent the cycle.

    Parameters:
    ----------
    daily_cycles : Dict[str, pd.Series]
        Diurnal cycles (hourly mean) of observation data (OBS).
        Key: Variable name, Value: Series with hour index (0-23).
    daily_cycles_mod : Dict[str, Dict[int, pd.Series]]
        Diurnal cycles (hourly mean) of model data (CORDEX).
        Outer Key: Variable name. Inner Key: Point ID (int).
        Value: Series with hour index (0-23 or sub-hourly/3-hourly).
    point_ids : List[int]
        List of model point IDs (e.g., [0, 1, 2, ...]).
    units : Dict[str, str]
        Dictionary mapping variable names to their units.
    plots_dir : str
        Directory where the plot will be saved.
    institution, RCM, model, version, ensemble : str
        Metadata used to name the output file.
    """
    if not daily_cycles:
        print("Warning: No OBS diurnal cycles to plot.")
        return

    # Check for limited observation data points
    first_var_name = next(iter(daily_cycles))
    num_obs_points = len(daily_cycles[first_var_name].index)
    if num_obs_points < 24:
        print(f"\n*** WARNING: OBS diurnal cycle data has only {num_obs_points} points (expected 24). The plot might look incomplete or flat. ***")
        if num_obs_points <= 1:
            print("This suggests the data might be a daily mean, not a cycle, which results in a straight line.\n")


    num_vars = len(daily_cycles)
    # Create figure and subplots
    fig_day, axes_day = plt.subplots(num_vars, 1, figsize=(16, 5 * num_vars), sharex=True)

    # Update the plot title to reflect hourly aggregation
    fig_day.suptitle("Diurnal Cycle (Hourly Aggregation) – Observations vs Model Points",
                      fontsize=22, y=1.02)

    # Ensure 'axes_day' is iterable even if only one variable is present
    if num_vars == 1:
        axes_day = [axes_day]

    # Define colors for the model points using a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(point_ids)))

    for i, (name, obs_cycle) in enumerate(daily_cycles.items()):
        ax = axes_day[i]
        
        # --- 1. OBS Data Plot ---
        
        # If it is precipitation, remove zeros (replace with NaN)
        if "precip" in name.lower():
            obs_cycle = obs_cycle.replace(0, np.nan)
            
        # Clean NaNs to ensure a continuous line in OBS
        obs_cycle_clean = obs_cycle.dropna()
        if obs_cycle_clean.empty:
             continue
             
        # Plot observations as a line with markers
        ax.plot(obs_cycle_clean.index, obs_cycle_clean.values,
                 marker='o', color='black', linewidth=2, label="Observations")

        # --- 2. MODEL Data Plot (Looping through all grid points) ---
        for j, pid in enumerate(point_ids):
            # Retrieve the model cycle for the current variable and point ID (pid)
            mod_cycle = daily_cycles_mod.get(name, {}).get(pid, None)

            if mod_cycle is None:
                # Skip if no model data is available for this point/variable combination
                continue

            # If it is precipitation, remove zeros (replace with NaN)
            if "precip" in name.lower():
                mod_cycle = mod_cycle.replace(0, np.nan)

            # Clean NaNs to ensure the connection of model points
            mod_cycle_clean = mod_cycle.dropna()
            if mod_cycle_clean.empty:
                 continue
            
            # 3. Plot the actual data points, connected with a solid line
            ax.plot(mod_cycle_clean.index, mod_cycle_clean.values,
                      marker='s', linestyle='-', linewidth=1.5, # Solid and thicker line
                      color=colors[j],
                      alpha=0.8,
                      label=f"Model P{pid}") # Simplified label

            # Y-axis: Set the variable name and units
            unit = units.get(name, 'Unit')
            ax.set_ylabel(f"{name} ({unit})", fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Better legend handling to avoid duplicates
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), fontsize=10, ncol=3, loc='upper right')

    # --- X-axis formatting (Hours) ---
    # Apply formatting to the last subplot only
    axes_day[-1].set_xticks(range(0, 24, 2)) # Set ticks every 2 hours (0, 2, 4, ..., 22)
    axes_day[-1].set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 24, 2)], # Clear labels HH:00
        fontsize=12
    )
    axes_day[-1].set_xlabel("Hour (UTC)", fontsize=16)
    axes_day[-1].set_xlim(-0.5, 23.5) # Ensure x-axis covers the full 0-23 range

    # Final layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Display the plot
    plt.show()

    # Create directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # File name based on metadata (changed to 'diurnal_cycle')
    filename = f"diurnal_cycle_all_points_{institution}_{RCM}_{model}_{version}_{ensemble}.png"
    full_path = os.path.join(plots_dir, filename)

    try:
        fig_day.savefig(full_path, dpi=300)
        print(f"\nSUCCESS: Diurnal cycle plot saved to {full_path}")
    except Exception as e:
        print(f"\nERROR: Could not save the plot to {full_path}. Error: {e}")