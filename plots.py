import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict, List

def _decimal_to_hhmm(h):
    if np.isnan(h):
        return ""
    h = float(h) % 24
    hrs = int(h)
    mins = int(round((h - hrs) * 60))
    if mins == 60:
        hrs = (hrs + 1) % 24
        mins = 0
    return f"{hrs:02d}:{mins:02d}"


def _clean_series(series):
    """
    Convierte índice a float, ordena y elimina NaNs en valores.
    Mantiene indexación decimal.
    """
    idx = pd.to_numeric(series.index, errors="coerce")
    s = pd.Series(series.values, index=idx)
    s = s[~s.index.isna()]          
    s = s.sort_index()              
    s = s.dropna()                  
    return s
    
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


def plot_diurnal_cycles_summary(daily_cycles_mod, daily_cycles, variables_plot):
    """
    Plot diurnal cycles with:
      - One line per model
      - One line for OBS
      - Legend only once in first subplot
      - Precipitation: replace 0 with NaN
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    n_vars = len(variables_plot)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 5 * n_vars), sharex=False)

    if n_vars == 1:
        axes = [axes]

    # Model names for color assignment
    model_names = list(daily_cycles_mod.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

    for i, var in enumerate(variables_plot):
        ax = axes[i]
        ax.set_title(f" ", fontsize=13)

        # --------------------------
        # 1. PLOT OBSERVATIONS
        # --------------------------
        obs_series = daily_cycles.get(var, None)
        if obs_series is not None:
            obs = obs_series.copy()

            # Replace zeros for precip
            if "precip" in var.lower():
                obs = obs.replace(0, np.nan)

            obs_clean = obs.dropna()

            if not obs_clean.empty:
                ax.plot(
                    obs_clean.index,
                    obs_clean.values,
                    color="black",
                    linewidth=2,
                    marker="o",
                    label="Observations"
                )

        # --------------------------
        # 2. PLOT MODELS
        # --------------------------
        legend_items = {}

        for color, model_name in zip(colors, model_names):

            if var not in daily_cycles_mod.get(model_name, {}):
                continue

            series = daily_cycles_mod[model_name][var].get("All_Points_Agg", None)
            if series is None:
                continue

            mod = series.copy()

            # Replace zeros for precipitation
            if "precip" in var.lower():
                mod = mod.replace(0, np.nan)

            mod_clean = mod.dropna()
            if mod_clean.empty:
                continue

            line = ax.plot(
                mod_clean.index,
                mod_clean.values,
                linewidth=1.5,
                marker="s",
                alpha=0.9,
                color=color,
                label=model_name
            )[0]

            legend_items[model_name] = line

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylabel(var)

        # -------------------------------------------------
        # ONLY FIRST SUBPLOT SHOWS THE LEGEND
        # -------------------------------------------------
        if i == 0:
            # Unique legend: OBS + models
            handles = []
            labels = []

            # Add OBS if it exists
            if obs_series is not None:
                handles.append(ax.lines[0])
                labels.append("Observations")

            # Add models
            for model_name, line in legend_items.items():
                handles.append(line)
                labels.append(model_name)

            ax.legend(handles, labels, ncol=3, fontsize=9)

    plt.tight_layout()
    plt.show()
    return fig


def plot_annual_cycles_summary(
    annual_cycles_mod_by_model: Dict[str, Dict[str, Dict[str, any]]],
    annual_cycles_obs: Dict[str, any],
    variables: Dict[str, str],
    figsize: tuple = (10, 4)
):
    """
    Plot annual cycles for all variables in vertically stacked subplots.

    Compares:
        - Observations
        - Each model (each CSV file is one model)
    """

    var_names = list(variables.keys())
    n_vars = len(var_names)

    if n_vars == 0:
        raise ValueError("Variable list is empty. Nothing to plot.")

    # Create figure with one subplot per variable
    fig, axes = plt.subplots(
        n_vars, 1,
        figsize=(figsize[0], figsize[1] * n_vars),
        sharex=True
    )

    if n_vars == 1:
        axes = [axes]

    # ---------------------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------------------
    for i, var_name in enumerate(var_names):
        ax = axes[i]

        # Plot observations
        obs_series = annual_cycles_obs[var_name]
        ax.plot(
            obs_series.index,
            obs_series.values,
            label="Observations",
            linewidth=2,
            color="black"
        )

        # Plot each model
        for model_name, model_dict in annual_cycles_mod_by_model.items():

            if var_name not in model_dict:
                continue
            if "All_Points_Agg" not in model_dict[var_name]:
                continue

            model_series = model_dict[var_name]["All_Points_Agg"]

            ax.plot(
                model_series.index,
                model_series.values,
                label=model_name,
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_ylabel(var_name)
        ax.set_title(f" ", fontsize=12)
        ax.grid(True, alpha=0.3)

        if i == n_vars - 1:
            ax.set_xlabel("Month")
            ax.set_xticks(range(1, 13))

        # Legend only once — same as daily format
        if i == 0:
            ax.legend(ncol=3, fontsize=9)

    plt.tight_layout()
    return fig

