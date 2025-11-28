# run_obs_models_safe.py
import papermill as pm
import pandas as pd
from pathlib import Path

# --- Notebook y directorio de salida ---
notebook_input = "observation.ipynb"
output_dir = Path("notebooks_run")
output_dir.mkdir(exist_ok=True)
error_dir = Path("notebooks_errors")
error_dir.mkdir(exist_ok=True)

# --- Definir combinaciones de modelos ---
models = pd.DataFrame([
    ["CNRM", "NCC-NorESM1-M", "ALADIN63", "v1", "r1i1p1"],
    ["CLMcom-ETH", "MOHC-HadGEM2-ES", "COSMO-crCLIM-v1-1", "v1", "r1i1p1"],
    ["CNRM", "MPI-M-MPI-ESM-LR", "ALADIN63", "v1", "r1i1p1"],
    ["KNMI", "MPI-M-MPI-ESM-LR", "RACMO22E", "v1", "r1i1p1"],
    ["MOHC", "NCC-NorESM1-M", "HadREM3-GA7-05", "v1", "r1i1p1"],
    ["CNRM", "MOHC-HadGEM2-ES", "ALADIN63", "v1", "r1i1p1"],
    ["KNMI", "MOHC-HadGEM2-ES", "RACMO22E", "v2", "r1i1p1"],
    ["MOHC", "MOHC-HadGEM2-ES", "HadREM3-GA7-05", "v1", "r1i1p1"],
    ["MOHC", "MPI-M-MPI-ESM-LR", "HadREM3-GA7-05", "v1", "r1i1p1"],
    ["CLMcom-ETH", "MPI-M-MPI-ESM-LR", "COSMO-crCLIM-v1-1", "v1", "r1i1p1"],
    ["KNMI", "NCC-NorESM1-M", "RACMO22E", "v1", "r1i1p1"],
    ["CLMcom-ETH", "NCC-NorESM1-M", "COSMO-crCLIM-v1-1", "v1", "r1i1p1"]
], columns=["Institution_RCM", "RCM_GCM", "Model_RCM", "Version", "Ensemble"])

# --- Iterar sobre cada combinación ---
for idx, row in models.iterrows():
    output_path = output_dir / f"observation_{idx+1}_{row['Model_RCM']}.ipynb"
    
    print(f"Running notebook for {row['Institution_RCM']} / {row['RCM_GCM']} / {row['Model_RCM']} ...")
    
    try:
        pm.execute_notebook(
            input_path=notebook_input,
            output_path=str(output_path),
            parameters={
                "institution": row["Institution_RCM"],
                "RCM": row["RCM_GCM"],
                "model": row["Model_RCM"],
                "version": row["Version"],
                "ensemble": row["Ensemble"]
            },
            kernel_name="python3"
        )
    except Exception as e:
        # Guardar notebook de error con sufijo _ERROR
        error_path = error_dir / f"ERROR_observation_{idx+1}_{row['Model_RCM']}.ipynb"
        print(f"ERROR executing notebook for {row['Model_RCM']}: {e}")
        print(f"Saving errored notebook to {error_path}")
        # Intentar exportar el notebook incluso si falla
        try:
            pm.execute_notebook(
                input_path=notebook_input,
                output_path=str(error_path),
                parameters={
                    "institution": row["Institution_RCM"],
                    "RCM": row["RCM_GCM"],
                    "model": row["Model_RCM"],
                    "version": row["Version"],
                    "ensemble": row["Ensemble"]
                },
                kernel_name="python3",
                report_mode=True  # intenta guardar notebook aunque haya error
            )
        except:
            print(f"Failed to save error notebook for {row['Model_RCM']}")
        continue

print("All notebooks attempted, errors saved separately.")
