# eval.py for TSP problem (Final Version with Hardcoded Dataset)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import os
import numpy as np

# --- ARES Integration: Dynamically import the generated heuristics ---
try:
    import main as solver
except ImportError:
    sys.stderr.write("FATAL ERROR: main.py not found. Ensure the core solver logic is in main.py.\n")
    print(float('inf'))
    sys.exit(1)


# --- End of ARES Integration Block ---


def evaluate_heuristic_on_tsp():
    """
    The core evaluation function for the TSP problem.
    It loads a single, hardcoded dataset (train50_dataset.npy),
    runs the solver on all its instances, and returns the average
    objective value (path length).
    """
    # 1. Load the hardcoded dataset
    basepath = os.path.join(os.path.dirname(__file__), "dataset")

    # --- [MODIFICATION] Hardcode the dataset path as requested ---
    dataset_path = os.path.join(basepath, "train50_dataset.npy")
    # --- [END MODIFICATION] ---

    if not os.path.isfile(dataset_path):
        try:
            from gen_init import generate_datasets
            sys.stderr.write(f"Dataset not found at {dataset_path}. Generating new dataset...\n")
            generate_datasets()
        except ImportError:
            sys.stderr.write("FATAL ERROR: gen_init.py not found. Cannot generate dataset.\n")
            return float('inf')
        except Exception as e:
            sys.stderr.write(f"FATAL ERROR: Failed to generate dataset: {e}\n")
            return float('inf')

    try:
        node_positions = np.load(dataset_path)
    except Exception as e:
        sys.stderr.write(f"FATAL ERROR: Failed to load dataset from {dataset_path}: {e}\n")
        return float('inf')

    # 2. Solve each instance and collect results
    objective_values = []
    for i, node_pos in enumerate(node_positions):
        try:
            obj = solver.solve(node_pos)
            objective_values.append(obj)
        except Exception as e:
            sys.stderr.write(f"ERROR: Instance {i} failed to solve: {e}\n")
            objective_values.append(float('inf'))

    # 3. Calculate the final fitness score
    if not objective_values:
        sys.stderr.write("ERROR: No instances were solved.\n")
        return float('inf')

    mean_obj = np.mean(objective_values)
    return mean_obj


if __name__ == "__main__":
    try:
        # --- [MODIFICATION] Removed command-line argument parsing ---
        # Since the dataset is hardcoded, we no longer need to parse
        # problem_size, root_dir, or mood from sys.argv.
        # This simplifies the script for both direct and ARES execution.

        # --- Run the evaluation ---
        fitness_score = evaluate_heuristic_on_tsp()

        # --- Standardized ARES Output ---
        print(fitness_score)

    except Exception as e:
        sys.stderr.write(f"FATAL ERROR during evaluation script execution: {e}\n")
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(float('inf'))