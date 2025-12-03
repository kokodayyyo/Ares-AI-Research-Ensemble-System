# main.py (Final Simplified Version)
import hydra
import logging
import os
from pathlib import Path
import subprocess

from utils.utils import print_hyperlink, file_to_string
from omegaconf import DictConfig

# -----------------------------------------------------------------------------
ROOT_DIR = os.getcwd()

import sys

sys.path.append(ROOT_DIR)
from src.ares import ARES

# -----------------------------------------------------------------------------
# Configure logging to use UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    stream=sys.stdout,
    encoding='utf-8'
)


def print_ares_banner():
    """Prints a cool ASCII art banner for ARES."""
    logging.info("=========================================================")
    logging.info("=   ARES: An AI Research Team for Heuristic Discovery   =")
    logging.info("=========================================================")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    print_ares_banner()

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using Algorithm: {cfg.algorithm.upper()}")

    # 1. Instantiate LLM clients
    logging.info("Instantiating LLM clients for ARES roles...")
    try:
        theorist_llm = hydra.utils.instantiate(cfg.llm_roles.theorist)
        logging.info(f"  - Theorist LLM: {cfg.llm_roles.theorist.model}")
        critic_llm = hydra.utils.instantiate(cfg.llm_roles.critic)
        logging.info(f"  - Critic LLM: {cfg.llm_roles.critic.model}")
        experimenter_llm = hydra.utils.instantiate(cfg.llm_roles.experimenter)
        logging.info(f"  - Experimenter LLM: {cfg.llm_roles.experimenter.model}")
    except Exception as e:
        logging.error(f"Failed to instantiate LLM clients. Error: {e}")
        return

    # 2. Instantiate the ARES engine
    logging.info("Initializing ARES Engine...")
    ares_system = ARES(cfg, ROOT_DIR,
                       generator_llm=experimenter_llm,
                       reflector_llm=theorist_llm,
                       theorist_llm=theorist_llm,
                       critic_llm=critic_llm,
                       experimenter_llm=experimenter_llm)

    # 3. Start the evolution process
    logging.info("\nStarting the evolutionary process...")
    best_code_overall, best_code_path_overall = ares_system.evolve()

    logging.info("\nEvolutionary process finished!")
    logging.info("=========================================================")
    logging.info(f"Best Code Overall found:\n{best_code_overall}")
    if best_code_path_overall:
        best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
        logging.info(f"Best Code was generated in: {print_hyperlink(best_path, best_code_path_overall)}")
    logging.info("=========================================================")

    # --- [ULTIMATE SIMPLIFICATION] Step 4: Final Confirmation Run ---
    # This step simply re-runs the best found code using the exact same
    # evaluation script to confirm its performance.

    logging.info("\nRunning final confirmation run on the best code...")

    if best_code_overall:
        # Prepare the gpt.py file in the original problem directory
        problem_dir = os.path.join(ROOT_DIR, "problems", cfg.problem.problem_name)
        gpt_path = os.path.join(problem_dir, "gpt.py")
        with open(gpt_path, 'w', encoding='utf-8') as file:
            file.write(best_code_overall)

        # Define the evaluation script and the output file
        eval_script = os.path.join(problem_dir, "eval.py")
        confirmation_stdout_path = os.path.join(workspace_dir, "best_code_overall_confirmation_stdout.txt")

        logging.info(f"Executing confirmation script: {print_hyperlink(eval_script)}")

        try:
            # Execute the standard eval script. It will use the 'train' set by default.
            with open(confirmation_stdout_path, 'w', encoding='utf-8') as stdout_file:
                subprocess.run(
                    ['python', eval_script],
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=cfg.get("validation_timeout", 600)
                )

            # Log the results from the confirmation output file
            logging.info(
                f"Confirmation run finished. Results are saved in {print_hyperlink(confirmation_stdout_path)}.")
            logging.info("--- Confirmation Run Results ---")
            # Use the robust file_to_string to read the entire output
            result_output = file_to_string(confirmation_stdout_path)
            for line in result_output.strip().split('\n'):
                logging.info(line)
            logging.info("------------------------------")

        except Exception as e:
            logging.error(f"An unexpected error occurred during the final confirmation run: {e}")
            logging.error(f"See log for details: {print_hyperlink(confirmation_stdout_path)}")

    else:
        logging.warning("No best code was generated to run confirmation on.")


if __name__ == "__main__":
    main()