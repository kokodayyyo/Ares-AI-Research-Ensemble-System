import multiprocessing
import sys
import tempfile
import shutil
import os
import subprocess
import numpy as np
from datetime import datetime
import copy
import pickle
from omegaconf import DictConfig

from utils.utils import *
from utils.llm_client.base import BaseClient
from src.utils.strategy_table_parser import parse_strategy_table
from src.utils.experiment_report_parser import parse_experiment_report
from src.utils.innovation_detector import detect_innovation_type, get_innovation_instruction
from typing import Optional, Tuple, List, Union
from src.utils.init_response import parse_init_response

import logging

# --- [FIX] Configure logging to use UTF-8 encoding globally ---
# This should be done once, at the very beginning of the application.
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    stream=sys.stdout,  # You can also direct it to a file
    encoding='utf-8'  # This is the critical line
)


def _evaluate_run_worker(args: tuple) -> tuple:
    """
    A universal worker function that evaluates a single code string in an
    isolated environment.
    """
    # 1. Unpack all arguments - now includes root_dir
    ind_index, run_id, individual_code, problem_name, root_dir, timeout, obj_type, final_stdout_path = args

    # 2. Create a unique, isolated temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # 3. --- [FIX] Prepare the evaluation environment using the absolute root_dir ---
        original_problem_dir = os.path.join(root_dir, "problems", problem_name)
        # --- [END FIX] ---

        eval_env_dir = os.path.join(temp_dir, "problem")
        shutil.copytree(original_problem_dir, eval_env_dir)

        # 4. Write the individual's code to the `gpt.py` file
        target_code_file = os.path.join(eval_env_dir, "gpt.py")
        with open(target_code_file, 'w', encoding='utf-8') as f:
            f.write(individual_code)

        # 5. Define the full path to the evaluation script
        eval_script = os.path.join(eval_env_dir, "eval.py")

        # 6. Execute the evaluation script
        with open(final_stdout_path, 'w', encoding='utf-8') as f_stdout:
            subprocess.run(
                ['python', eval_script],
                stdout=f_stdout,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=True,
                cwd=eval_env_dir
            )

        # 7. Parse the score from the output file
        result = parse_score_from_file(final_stdout_path)
        score = (result if obj_type == "min" else -result) if result is not None else None

        return (ind_index, run_id, score)

    except Exception as e:
        # If any error occurs (timeout, crash, etc.), log it to the stdout file for debugging
        error_message = f"\n--- EVALUATION FAILED IN SUBPROCESS ---\n{type(e).__name__}: {e}\n"
        try:
            with open(final_stdout_path, 'a', encoding='utf-8') as f_stdout:
                f_stdout.write(error_message)
        except Exception as log_e:
            print(f"CRITICAL: Failed to write error log for subprocess: {log_e}")

        return (ind_index, run_id, None)

    finally:
        # 8. Always clean up the temporary directory
        shutil.rmtree(temp_dir)


# --- [NEW] Robust score parsing helper function ---
def parse_score_from_file(filepath: str) -> Optional[float]:
    """
    Finds the last line in a file that can be cleanly converted to a float.
    This is the most robust method against all forms of output pollution,
    as it requires the score to be on its own line.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Iterate through the lines in REVERSE order
            for line in reversed(lines):
                cleaned_line = line.strip()
                if not cleaned_line:  # Skip empty lines
                    continue
                try:
                    # Attempt to convert the entire cleaned line to a float
                    return float(cleaned_line)
                except ValueError:
                    # If the line contains anything other than a float (like logs, warnings),
                    # this will fail, and we'll correctly move to the previous line.
                    continue
        # If no line in the entire file could be converted
        return None
    except FileNotFoundError:
        logging.error(f"Score file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while parsing score from {filepath}: {e}")
        return None


class ARES:

    def __init__(
            self,
            cfg: DictConfig,
            root_dir: str,
            generator_llm: BaseClient,
            reflector_llm: Optional[BaseClient] = None,
            theorist_llm: Optional[BaseClient] = None,
            critic_llm: Optional[BaseClient] = None,
            experimenter_llm: Optional[BaseClient] = None,
    ) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.workspace_dir = os.path.join(self.root_dir, "LLM-outputs", current_time)
        os.makedirs(self.workspace_dir, exist_ok=True)
        logging.info(f"ARES working directory for this run: {self.workspace_dir}")

        # Map LLM clients to roles
        self.theorist_llm = theorist_llm or reflector_llm or generator_llm
        self.critic_llm = critic_llm or reflector_llm or generator_llm
        self.experimenter_llm = experimenter_llm or generator_llm

        # 创建角色专属的输出子目录
        self.theorist_output_dir = f"{self.workspace_dir}/theorist_outputs"
        self.critic_output_dir = f"{self.workspace_dir}/critic_outputs"
        self.experimenter_output_dir = f"{self.workspace_dir}/experimenter_outputs"
        os.makedirs(self.theorist_output_dir, exist_ok=True)
        os.makedirs(self.critic_output_dir, exist_ok=True)
        os.makedirs(self.experimenter_output_dir, exist_ok=True)
        """self.benchmark_output_dir = f"{self.workspace_dir}/benchmark_outputs"
        os.makedirs(self.benchmark_output_dir, exist_ok=True)"""

        # Core State Variables
        self.strategy_table_str: str = ""
        self.anti_pattern_lib: List[dict] = []
        self.stagnation_counter: int = 0
        self.checkpoint: Optional[dict] = None
        self.is_in_radical_exploration: bool = False
        self.radical_exploration_timer: int = 0

        # Configuration
        self.meta_reflection_cycle_period = cfg.get("meta_reflection_cycle_period", 5)
        self.radical_exploration_trigger = cfg.get("radical_exploration_trigger", 3)
        self.radical_exploration_duration = cfg.get("radical_exploration_duration", 3)
        self.stagnation_threshold = cfg.get("stagnation_threshold", 1e-4)
        self.eval_runs = cfg.get("eval_runs", 3)

        # Evolutionary State
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.best_obj_overall = float('inf')
        self.best_code_overall = None
        self.best_code_path_overall = None

        # Progress tracking
        self.history = {
            'iterations': [],
            'best_scores': [],
            'avg_scores': [],
            'stagnation': [],
            'factor_weights': []
        }

        # Call init_prompt to load all necessary resources
        self.init_prompt()
        # Call init_population to start the ARES process
        self.init_population()

    def _parse_innovation_proposal(self, proposal_text: str) -> dict:
        """
        Parses the Theorist's proposal to extract structured information.
        This version is specifically tailored to parse the output format of
        the Theorist's proposal, using a robust, regex-based approach.
        """
        logging.info("Parsing Theorist's innovation proposal with robust regex...")

        info = {
            "type": "Unknown",
            "factor_to_ablate": None,
            "parameter_to_scan": None
        }

        # 1. Identify the types of innovation
        type_match = re.search(r"\[(Structural|Parametric|Mixed)\]", proposal_text, re.IGNORECASE)
        if type_match:
            info["type"] = type_match.group(1).strip().capitalize() + " Innovation"

        # 2. Extract the names of the factors to be ablated
        ablation_match = re.search(r"<factor_name_to_ablate>(.*?)</factor_name_to_ablate>", proposal_text, re.DOTALL)
        if ablation_match:
            info["factor_to_ablate"] = ablation_match.group(1).strip()

        # 3. Extract the parameter names to be scanned
        param_match = re.search(r"<parameter_name_to_scan>(.*?)</parameter_name_to_scan>", proposal_text, re.DOTALL)
        if param_match:
            info["parameter_to_scan"] = param_match.group(1).strip()

        if info["type"] != "Unknown":
            logging.info(
                f"Successfully parsed proposal. Type: '{info['type']}', "
                f"Ablation Target: '{info['factor_to_ablate']}', "
                f"Scan Target: '{info['parameter_to_scan']}'"
            )
        else:
            logging.warning("Could not parse a valid innovation type from the Theorist's proposal.")

        return info

    def init_prompt(self) -> None:
        """
        Initialize all Prompt assets for the ARES system.
        This method is responsible for reading all templates and static information from files,
        and loading them into class member variables to prepare for subsequent evolutionary cycles.
        """
        # 1. Set up basic information and paths
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        # self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.get("problem_type", "standard")

        logging.info("Initializing ARES (AI Research Ensemble System)...")
        logging.info(f"Problem: {self.problem}")

        self.prompt_dir = f"{self.root_dir}/prompts"
        ares_prompt_dir = f'{self.prompt_dir}/ares'
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}'

        # Define temporary code file used by ARES system during evaluation
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        # 2. Load problem-specific "background information" files
        logging.info("Loading problem-specific context files (signature, description, seed)...")
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')

        # 3. Load Prompt templates for all AI agents in the ARES framework
        logging.info("Loading ARES agent prompt templates...")
        # Theorist
        self.theorist_init_prompt = file_to_string(f'{ares_prompt_dir}/user_generator_theorist_init.txt')
        # [Core Modification] Load two split "Phase 1 Reflection" Prompts
        self.theorist_phase1_task1_prompt = file_to_string(
            f'{ares_prompt_dir}/user_reflector_theorist_p1_task1_update_table.txt')
        self.theorist_phase1_task2_prompt = file_to_string(
            f'{ares_prompt_dir}/user_reflector_theorist_p1_task2_propose_innovation.txt')
        # Load "Phase 2 Reflection" and special mode Prompts
        self.theorist_phase2_prompt = file_to_string(
            f'{ares_prompt_dir}/user_reflector_theorist_phase2_conventional.txt')
        self.theorist_radical_prompt = file_to_string(f'{ares_prompt_dir}/user_reflector_theorist_radical.txt')
        self.theorist_meta_prompt = file_to_string(f'{ares_prompt_dir}/user_reflector_theorist_meta.txt')

        # Critic
        self.critic_ablation_prompt = file_to_string(f'{ares_prompt_dir}/user_reflector_critic_ablation.txt')
        self.critic_scan_prompt = file_to_string(f'{ares_prompt_dir}/user_reflector_critic_scan.txt')
        self.critic_mixed_prompt = file_to_string(f'{ares_prompt_dir}/user_reflector_critic_mixed.txt')

        # Experimenter
        self.experimenter_prompt = file_to_string(f'{ares_prompt_dir}/user_generator_experimenter.txt')

        # 4. Load core data structure templates
        self.strategy_table_template = file_to_string(f'{ares_prompt_dir}/ares_strategy_table_template.md')

        # 5. Construct unified, immutable context information block (user_generator_context)
        #    This variable will be dynamically injected into all Prompts requiring complete problem background during runtime.

        # Build context block
        self.user_generator_context = f"""
[Problem Definition]
Write the {self.func_name} function for {self.problem_desc}

[Function Signature]
{self.func_signature}

[Core API and Concept Toolbox]
{self.func_desc}

[Seed Function Reference]
{self.seed_func}
"""



    # In ares.py, completely replace the old init_population function with this version

    def init_population(self) -> None:
        """
        Phase 0: Project initiation and knowledge system initialization.
        Benchmark-free version using Theorist reasoning.
        """
        # -----------------------------------------------------------------
        # Step 1: Theorist generates initial strategy table (V0)
        # -----------------------------------------------------------------
        prompt = self.theorist_init_prompt.format(
            initial_strategy_table_template=self.strategy_table_template
        )
        messages = [{"role": "user", "content": prompt}]

        logging.info("Theorist is establishing initial research direction from first principles...")
        try:
            response_choices = self.theorist_llm.chat_completion(
                n=1,
                messages=messages,
                temperature=self.theorist_llm.temperature
            )
            full_response = response_choices[0].message.content if response_choices else ""
        except Exception as e:
            logging.critical(f"Theorist failed to generate initial strategy table. Error: {e}")
            raise RuntimeError("Could not initialize ARES due to Theorist LLM failure.") from e

        # Parse Theorist output into strategy table and analysis text
        parsed_response = parse_init_response(full_response)
        self.strategy_table_str = parsed_response["strategy_table"]
        analysis_text = parsed_response["analysis_text"]

        # Save only the strategy table
        self.weights_history_path = f"{self.workspace_dir}/weights_history.txt"
        with open(self.weights_history_path, "w", encoding='utf-8') as f:
            f.write(f"--- Iteration 0 (Initial Beliefs) ---\n\n")
            f.write(self.strategy_table_str + "\n")

        # Log both parts
        logging.info(f"Initial Strategy Table (V0):\n{self.strategy_table_str}")
        logging.info(f"Initial Analysis Text:\n{analysis_text}")

        # Parse strategy table for factor weights
        if self.strategy_table_str:
            try:
                parsed_table = parse_strategy_table(self.strategy_table_str)
                self.history['factor_weights'].append(parsed_table.get('factors', {}))
            except Exception as e:
                logging.error(f"Failed to parse initial strategy table: {e}")

        # -----------------------------------------------------------------
        # Step 2: Evaluate seed function as initial elite
        # -----------------------------------------------------------------
        logging.info("Evaluating seed function as the initial elitist...")
        seed_ind = self.response_to_individual(self.seed_func, 0, role="seed")
        self.population = self.evaluate_population([seed_ind])
        self.update_iter()

        # -----------------------------------------------------------------
        # Step 3: Experimenter generates initial population based on V0
        # -----------------------------------------------------------------
        logging.info("Experimenter is generating the initial population based on Theorist's strategy...")
        initial_anti_patterns = "No anti-patterns have been identified yet."
        prompt = self.experimenter_prompt.format(
            user_generator_context=self.user_generator_context,
            final_strategy_table=self.strategy_table_str,
            proposal=analysis_text,
            updated_problem_characterization="This is the initial population generation. No problem characterization has been developed yet.",
            best_code=self.elitist['code'],
            anti_pattern_lib=initial_anti_patterns,
            func_name=self.func_name
        )

        try:
            responses = self.experimenter_llm.multi_chat_completion(
                messages_list=[[{"role": "user", "content": prompt}]],
                n=self.cfg.init_pop_size
            )
        except Exception as e:
            logging.critical(f"Experimenter failed to generate initial population. Error: {e}")
            raise RuntimeError("Could not continue ARES run due to Experimenter LLM failure.") from e

        new_population = [
            self.response_to_individual(res, i + 1, role="experimenter")
            for i, res in enumerate(responses)
        ]
        evaluated_new_population = self.evaluate_population(new_population)
        self.population.extend(evaluated_new_population)
        self.update_iter()

    def response_to_individual(self, response: str, response_id: Union[int, str], role: str = "experimenter") -> dict:

        # Determine the output directory based on the role
        role_dir = f"{self.workspace_dir}/{role}_outputs"
        # If it is a numeric ID, it belongs to the experimenter.
        if isinstance(response_id, int):
            role_dir = self.experimenter_output_dir
        # If it contains the keyword "critic", it belongs to criti.
        elif "critic" in str(response_id):
            role_dir = self.critic_output_dir

        elif role == "seed":
            role_dir = f"{self.workspace_dir}/seed"
            os.makedirs(role_dir, exist_ok=True)

        file_name_base = f"iter_{self.iteration}_id_{response_id}"
        response_path = f"{role_dir}/{file_name_base}_response.txt"
        code_path = f"{role_dir}/{file_name_base}_code.py"
        stdout_path = f"{role_dir}/{file_name_base}_stdout.txt"

        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(response + '\n')

        code = extract_code_from_generator(response)

        return {
            "stdout_filepath": stdout_path,
            "code_path": code_path,
            "code": code,
            "response_id": response_id,
            "exec_success": False,
            "obj": float('inf'),
            "role": role,
        }

    def evaluate_population(self, population: List[dict]) -> List[dict]:
        """
        Evaluate an entire population in parallel using a process pool.
        This method prepares evaluation tasks, distributes them to worker processes,
        and aggregates the results.
        """
        logging.info(
            f"--- Starting Parallel Evaluation for {len(population)} Individuals in Iteration {self.iteration} ---")

        # 1. Prepare all evaluation tasks. Each task corresponds to a single execution run.
        tasks = []
        for i, ind in enumerate(population):
            for run_id in range(self.eval_runs):
                stdout_path = ind["stdout_filepath"].replace(".txt", f"_run{run_id}.txt")

                # --- [FIX] Add self.root_dir to the arguments ---
                task_args = (
                    i,
                    run_id,
                    ind['code'],
                    self.problem,
                    self.root_dir,  # <-- 将根目录传递进去
                    self.cfg.timeout,
                    self.obj_type,
                    stdout_path
                )
                # --- [END FIX] ---
                tasks.append(task_args)

        # Increment the master function evaluation counter (counts individuals, not runs)
        # self.function_evals += len(population)

        # Prepare a data structure to collect scores for each individual
        results_by_individual = [[] for _ in population]

        # 2. Distribute tasks to the process pool for parallel execution
        # You can configure the number of workers in your config file or it will default to your CPU count.
        num_workers = self.cfg.get("num_eval_workers", os.cpu_count())
        logging.info(f"Distributing {len(tasks)} evaluation runs across {num_workers} worker processes...")

        with multiprocessing.Pool(processes=num_workers) as pool:
            # imap_unordered is efficient as it yields results as soon as they are ready
            for result in pool.imap_unordered(_evaluate_run_worker, tasks):
                ind_index, run_id, score = result

                if score is not None:
                    results_by_individual[ind_index].append(score)
                else:
                    # If a run fails, its score is None. Log the failure for traceability.
                    failed_ind = population[ind_index]
                    stdout_path = failed_ind["stdout_filepath"].replace(".txt", f"_run{run_id}.txt")
                    logging.error(
                        f"Run {run_id + 1}/{self.eval_runs} for Code {failed_ind['response_id']} FAILED. See log: {stdout_path}")

        # 3. Process the aggregated results and update the population
        logging.info(f"--- Evaluation Summary for Iteration {self.iteration} ---")
        for i, ind in enumerate(population):
            scores = results_by_individual[i]

            # An individual is only successful if all its runs completed and returned a score.
            if len(scores) == self.eval_runs:
                ind["obj"] = np.min(scores)
                ind["exec_success"] = True
                obj_val_str = f"{ind['obj']:.4f}"
            else:
                ind["obj"] = float('inf')
                ind["exec_success"] = False
                obj_val_str = "inf"

            logging.info(
                f"Iteration {self.iteration}, response_id {ind['response_id']}: Objective value: {obj_val_str}")

        logging.info("----------------------------------------------------")

        return population

    def select_key_individuals(self) -> Tuple[dict, dict, dict]:
        """Select best, second best, and worst individuals"""
        successful_pop = sorted(
            [ind for ind in self.population if ind.get("exec_success", False)],
            key=lambda x: x["obj"]
        )

        if len(successful_pop) < 3:
            # If not enough successful individuals, copy elite
            if self.elitist:
                return self.elitist, self.elitist, self.elitist
            raise RuntimeError("Not enough successful individuals in the population to proceed.")

        best_ind = successful_pop[0]
        second_best_ind = successful_pop[1] if len(successful_pop) > 1 else best_ind
        worst_ind = successful_pop[-1]

        return best_ind, second_best_ind, worst_ind

    # In ares.py, completely replace the old function with the same name with this version

    def run_theorist_phase1_reflection(self, best_ind: dict, worst_ind: dict) -> Tuple[str, str]:
        """Theorist Phase 1 Reflection"""
        # Task 1: Update strategy table
        prompt1 = self.theorist_phase1_task1_prompt.format(
            user_generator_context=self.user_generator_context,
            prior_strategy_table=self.strategy_table_str,
            best_code=best_ind['code'],
            worst_code=worst_ind['code']
        )

        response_choices1 = self.theorist_llm.chat_completion(
            n=1,
            messages=[{"role": "user", "content": prompt1}],
            temperature=self.theorist_llm.temperature
        )
        hypothetical_table_str = response_choices1[0].message.content if response_choices1 else ""

        # Save updated table
        with open(f"{self.theorist_output_dir}/hypothetical_table_iter{self.iteration}.md", 'w', encoding='utf-8') as f:
            f.write(hypothetical_table_str)

        # Task 2: Generate innovation proposal
        prompt2 = self.theorist_phase1_task2_prompt.format(
            hypothetical_strategy_table=hypothetical_table_str,  # Corrected variable name
            best_code=best_ind['code'],
            worst_code=worst_ind['code']
        )

        response_choices2 = self.theorist_llm.chat_completion(
            n=1,
            messages=[{"role": "user", "content": prompt2}],
            temperature=self.theorist_llm.temperature
        )
        proposal = response_choices2[0].message.content if response_choices2 else ""

        # Save proposal
        with open(f"{self.theorist_output_dir}/proposal_iter{self.iteration}.txt", 'w', encoding='utf-8') as f:
            f.write(proposal)

        return hypothetical_table_str, proposal

    def run_critic_validation(self, best_ind: dict, hypothetical_table: str, proposal: str) -> str:
        """
        Critic Validation:
        1. Call the parser to extract structured entities from the proposal.
        2. Select and populate the corresponding, single-responsibility Prompt based on the extracted type.
        3. Call LLM to generate "challenge codes" and produce the final experimental report.
        """
        logging.info("Critic is designing validation experiments...")

        if not proposal:
            logging.warning("No valid proposal from Theorist. Skipping Critic validation.")
            return {
                "report_markdown": "# Critic's Report\n\nSkipped: No valid proposal.",
                "experiment_details": []
            }

        # --- 1. [Core Fix] Call parser and safely extract all potentially needed entities ---
        innovation_info = self._parse_innovation_proposal(proposal)
        innovation_type = innovation_info.get("type", "Unknown")
        factor_to_ablate = innovation_info.get("factor_to_ablate") or ""
        parameter_to_scan = innovation_info.get("parameter_to_scan") or ""

        # Safely extract parameter scanning information
        param_info = innovation_info.get("parameter_to_scan")
        if isinstance(param_info, dict):
            param_name = param_info.get("parameter_name", "")
            # Convert both values and lists to strings for proper Prompt handling
            proposed_value = str(param_info.get("proposed_value", ""))
            scan_range = str(param_info.get("scan_range", []))
        else:  # If parsing result is not a dictionary, set to empty values
            param_name = str(param_info) if param_info else ""
            proposed_value = ""
            scan_range = "[]"

        logging.info(
            f"Parsed proposal. Type: '{innovation_type}', Ablation: '{factor_to_ablate}', Scan: '{param_name}'")

        # --- 2. [Core Fix] Select and populate the correct Prompt based on parsing results ---
        prompt = ""
        if innovation_type == "Structural Innovation":
            prompt_template = self.critic_ablation_prompt
            prompt = prompt_template.format(best_code=best_ind['code'], strategic_innovation_proposal=proposal,
                                            factor_name_to_ablate=factor_to_ablate)
        elif innovation_type == "Parametric Innovation":
            prompt_template = self.critic_scan_prompt
            prompt = prompt_template.format(best_code=best_ind['code'], strategic_innovation_proposal=proposal,
                                            parameter_name_to_scan=parameter_to_scan)
        elif innovation_type == "Mixed Innovation":
            prompt_template = self.critic_mixed_prompt
            prompt = prompt_template.format(best_code=best_ind['code'], strategic_innovation_proposal=proposal,
                                            factor_name_to_ablate=factor_to_ablate,
                                            parameter_name_to_scan=parameter_to_scan)
        else:
            logging.error(f"Unknown or un-parsable innovation type from proposal. Cannot proceed with criticism.")
            report = f"# Critic's Experimental Data Report - Iteration {self.iteration}\n\n"
            report += f"**Conclusion**: Skipped. Failed to parse a valid innovation type from the Theorist's proposal:\n> {proposal}\n"
            return report
        try:
            response_choices = self.critic_llm.chat_completion(
                n=1,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            response = response_choices[0].message.content if response_choices else ""
        except Exception as e:
            logging.error(f"Critic LLM call failed: {e}")
            response = ""

            # Archive raw output
        output_path = f"{self.critic_output_dir}/mutant_generation_raw_iter_{self.iteration}.txt"
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(response)

        # --- 2. Parse AI solution and prepare experiments ---
        mutant_codes = re.findall(r"```python(.*?)```", response, re.DOTALL)
        if not mutant_codes:
            logging.warning("Critic did not generate any valid mutant code blocks for validation.")
            report = f"# Critic's Experimental Data Report - Iteration {self.iteration}\n\n"
            report += "**Conclusion**: Critic failed to generate valid mutant code for testing.\n"
            # Return structurally consistent empty results even when failed
            return {
                "report_markdown": report,
                "experiment_details": [],
            }

        # --- 3. Execute experiments ---
        mutant_population = []
        for i, code in enumerate(mutant_codes):
            ind = self.response_to_individual(code, f"critic_mutant_{i}", role="critic")

            # Extract "Self-description" from the first line comment of the code.”
            try:
                first_line = code.strip().splitlines()[0]
                if first_line.startswith('#'):
                    ind['description'] = first_line.lstrip('#').strip()
                else:
                    ind['description'] = "No valid description comment found."
            except IndexError:
                ind['description'] = "Code was empty or malformed."

            mutant_population.append(ind)

        logging.info(f"Critic generated {len(mutant_population)} mutants for testing.")
        evaluated_mutants = self.evaluate_population(mutant_population)

        # --- 4. [Core Modification] Compose report and package all evidence ---

        # 4a. Prepare experiment data list containing all details
        experiment_details = [{
            "id": "Baseline",
            "description": "Original Best Code",
            "code": best_ind['code'],
            "score": best_ind['obj']
        }]
        for mutant in evaluated_mutants:
            experiment_details.append({
                "id": mutant['response_id'],
                "description": mutant.get('description', 'N/A'),  # Pass on the description
                "code": mutant['code'],
                "score": mutant['obj'] if mutant.get("exec_success") else 'FAILED'
            })

        # 4b. Compose human-readable Markdown summary report
        innovation_type = detect_innovation_type(proposal)
        report_md = f"# Critic's Experimental Data Report - Iteration {self.iteration}\n\n"
        report_md += "| Test ID | Modification Details | Performance Score | Delta vs. Best | Conclusion |\n"
        report_md += "|---|---|---|---|---|\n"

        # Generate report table from packaged experiment_details
        baseline_score = best_ind["obj"]
        for detail in experiment_details:
            clean_description = detail["description"].replace("Mutation - ", "").replace(":", "").strip()

            if detail["id"] == "Baseline":
                report_md += f'| Baseline (Best Code) | {clean_description} | {detail["score"]:.4f} | 0.0 | Reference |\n'
            else:
                if isinstance(detail["score"], float):
                    delta = detail["score"] - baseline_score
                    conclusion = "Improvement" if delta < 0 else "Degradation" if delta > 0 else "Neutral"
                    report_md += f'| {detail["id"]} | {clean_description} | {detail["score"]:.4f} | {delta:+.4f} | {conclusion} |\n'
                else:  # FAILED
                    report_md += f'| {detail["id"]} | {clean_description} | FAILED | N/A | Execution Error |\n'

        # 4c. Archive the report
        report_path = f"{self.critic_output_dir}/report_iter_{self.iteration}.md"
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(report_md)

        logging.info(f"Critic's report generated and saved to {report_path}")

        # 5d. Package all the information into a dictionary and return it
        return {
            "report_markdown": report_md,
            "experiment_details": experiment_details,
            "evaluated_mutants": evaluated_mutants  # <-- [NEW] Add this to the return dictionary
        }

    def run_theorist_final_reflection(self, phase1_outputs: tuple, critic_results: dict,
                                      best_ind_after_critic: dict) -> str:
        """
        Theorist Final Reflection: Integrates all evidence to form the final guide.

        [MODIFICATION]
        This function's signature has been updated to accept `best_ind_after_critic`.
        This ensures the Theorist's final analysis is always based on the absolute best-known code,
        especially after a potential breakthrough from the Critic's experiments.
        """
        logging.info("Theorist is performing Final Reflection (Conclusion Formation)...")
        hypothetical_table, proposal = phase1_outputs

        # --- 1. Unpack and prepare all necessary evidence from critic_results ---
        critic_report_md = critic_results.get("report_markdown", "Critic's report summary is not available.")

        # --- [MODIFICATION] Prepare detailed experimental code evidence for the LLM ---
        # This block is now shared across all modes that might use it.
        experiment_code_evidence = ""
        # The prompt will now be constructed using the potentially updated `best_ind_after_critic`
        # as the primary baseline for analysis.
        experiment_code_evidence += (
            f"\n--- Experiment ID: Baseline (Current Best After Critic) | "
            f"Final Score: {best_ind_after_critic.get('obj', 'N/A'):.4f} ---\n"
            f"```python\n{best_ind_after_critic.get('code', '# Code not available')}\n```\n"
        )
        for detail in critic_results.get("experiment_details", []):
            if detail.get("id") == "Baseline":
                continue
            score_str = f"{detail.get('score'):.4f}" if isinstance(detail.get('score'), float) else detail.get('score',
                                                                                                               'N/A')
            experiment_code_evidence += (
                f"\n--- Experiment ID: {detail.get('id', 'N/A')} | Final Score: {score_str} ---\n"
                f"```python\n{detail.get('code', '# Code not available')}\n```\n"
            )
        # --- [END MODIFICATION] ---

        # --- 2. Prepare Prompt and call LLM ---
        prompt = self.theorist_phase2_prompt.format(
            hypothetical_strategy_table=hypothetical_table,
            strategic_innovation_proposal=proposal,
            experimental_results_report=critic_report_md,
            parsed_experiment_data="...",
            experimental_code_evidence=experiment_code_evidence
        )
        temperature = self.theorist_llm.temperature

        # --- 3. Decision branch: Select different thinking modes based on system state ---

        # Mode 1: Meta-Reflection Cycle
        if self.stagnation_counter >= self.radical_exploration_trigger:
            logging.warning(
                f"Stagnation detected! Triggering Radical Exploration (Counter: {self.stagnation_counter})...")
            self.create_checkpoint()
            prompt = self.theorist_radical_prompt.format(
                stagnation_counter=self.stagnation_counter,
                stable_strategy_table=self.strategy_table_str,
                # anti_pattern_lib 在新prompt中已从table里读取
            )
            temperature += 0.4
            self.stagnation_counter = 0

        #  1.5Event-driven meta-reflection (Reviewing Major Breakthroughs)
        elif self.history.get("last_radical_was_successful", False):
            logging.info(
                "A successful radical exploration has just concluded. Triggering a post-hoc Meta-Reflection Cycle...")
            # After performing meta - reflection, reset the success flag to avoid repeated triggering.
            self.history["last_radical_was_successful"] = False
            try:
                # Load complete weight history from working directory
                with open(self.weights_history_path, "r", encoding='utf-8') as f:
                    weight_history = f.read()

                # (Optional) Perform programmatic analysis on history to generate trend summary
                trend_analysis_summary = self._analyze_weight_trends(weight_history)

                # Populate meta-reflection Prompt
                prompt = self.theorist_meta_prompt.format(
                    # Pass complete history for LLM's own analysis
                    weight_history=weight_history,
                )
            except Exception as e:
                logging.error(f"Meta-reflection failed during data preparation: {str(e)}")
                return "Meta-reflection could not be completed due to a data preparation error."


        # Mode 2: Radical Exploration
        elif self.iteration > 0 and self.iteration % self.meta_reflection_cycle_period == 0:
            logging.info(f"Triggering Meta-Reflection Cycle (Iteration {self.iteration})...")
            try:
                # Load complete weight history from working directory
                with open(self.weights_history_path, "r", encoding='utf-8') as f:
                    weight_history = f.read()

                # (Optional) Perform programmatic analysis on history to generate trend summary
                trend_analysis_summary = self._analyze_weight_trends(weight_history)

                # Populate meta-reflection Prompt
                prompt = self.theorist_meta_prompt.format(
                    # Pass complete history for LLM's own analysis
                    weight_history=weight_history,
                    # Can also pass programmatic analysis summary
                    # trend_analysis_summary=trend_analysis_summary,
                    # anti_pattern_lib="\n".join([str(p) for p in self.anti_pattern_lib])
                )
            except Exception as e:
                logging.error(f"Meta-reflection failed during data preparation: {str(e)}")
                return "Meta-reflection could not be completed due to a data preparation error."

        # Mode 3: Conventional Reflection
        else:
            prompt = self.theorist_phase2_prompt.format(
                hypothetical_strategy_table=hypothetical_table,
                strategic_innovation_proposal=proposal,
                experimental_results_report=critic_report_md,
                # parsed_experiment_data=str(parsed_report_summary),
                # [Core] Inject complete experimental evidence containing code
                experimental_code_evidence=experiment_code_evidence
            )

        # --- 4. Execute LLM call and process return value ---
        if not prompt:
            logging.error("Theorist final reflection prompt was empty. Aborting.")
            return ""  # Return empty string to avoid downstream errors

        try:
            response_choices = self.theorist_llm.chat_completion(
                n=1,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            # Safely extract text content of first result from returned list of Choice objects
            final_guide = response_choices[0].message.content if response_choices else ""

            # Write final guide to file for debugging and traceability
            final_guide_path = f"{self.theorist_output_dir}/final_guide_iter_{self.iteration}.txt"
            with open(final_guide_path, "w", encoding='utf-8') as f:
                f.write(final_guide)

            return final_guide

        except Exception as e:
            logging.critical(f"A critical error occurred during the Theorist's final reflection: {e}")
            # Return a string containing error information when a critical error occurs
            return f"Error during final reflection: {e}"

    def _analyze_weight_trends(self, weight_history_content: str) -> str:
        """
        Analyze long-term trends in factor weights.
        Accepts the complete text content of `weights_history.txt` as input,
        and returns a structured Markdown-formatted meta-strategy report.
        """
        trend_report = "# Factor Weight Trend Analysis\n\n"
        factor_trends = {}

        # --- 1. Robustly parse historical data ---
        try:
            parsed_history = []
            # Split content blocks by iteration using regular expressions
            iteration_blocks = re.split(r'--- Iteration \d+ .*?---', weight_history_content)

            for block in iteration_blocks:
                if block.strip():
                    # Call helper function to convert each block's Markdown table to dictionary
                    parsed_table = parse_strategy_table(block)
                    if parsed_table and 'factors' in parsed_table:
                        parsed_history.append(parsed_table)

            if len(parsed_history) < 2:  # Need at least two iterations for trend analysis
                return "Trend analysis requires at least two historical data points. Not enough data yet."

        except Exception as e:
            logging.error(f"Error parsing weight history for trend analysis: {e}")
            return f"Trend analysis failed due to a parsing error: {e}"

        # --- 2. Collect and track weight evolution for all factors ---
        all_factors = set()
        for iteration_data in parsed_history:
            all_factors.update(iteration_data.get('factors', {}).keys())

        for factor in sorted(list(all_factors)):
            weights = []
            for iteration_data in parsed_history:
                factor_data = iteration_data.get('factors', {}).get(factor, {})
                # Ensure weight is a float, handle potential parsing errors
                try:
                    weight_value = float(factor_data.get('Weight P(H)', 0.0))
                except (ValueError, TypeError):
                    weight_value = 0.0
                weights.append(weight_value)

            # --- 3. Calculate trend statistics ---
            start_weight = weights[0]
            end_weight = weights[-1]
            avg_weight = np.mean(weights)
            volatility = np.std(weights)  # Volatility (standard deviation)

            # Calculate trend slope using linear regression for greater precision
            if len(weights) > 1:
                x = np.arange(len(weights))
                # np.polyfit with deg=1 returns a tuple of (slope, intercept)
                slope, _ = np.polyfit(x, weights, 1)
            else:
                slope = 0.0

            # --- 4. Categorize each factor based on statistics ---
            trend_category = "Stable"
            if slope > 0.05:
                trend_category = "Strong Upward"
            elif slope > 0.01:
                trend_category = "Moderate Upward"
            elif slope < -0.05:
                trend_category = "Strong Downward"
            elif slope < -0.01:
                trend_category = "Moderate Downward"
            elif volatility > 0.15:
                trend_category = "High Volatility (Chameleon)"
            elif volatility > 0.05:
                trend_category = "Moderate Volatility (Chameleon)"

            factor_trends[factor] = {
                "trend": trend_category,
                "start": start_weight,
                "end": end_weight,
                "avg": avg_weight,
                "volatility": volatility,
                "slope": slope
            }

        # --- 5. Generate structured Markdown report ---
        trend_report += "## Quantitative Trend Summary\n"
        trend_report += "| Factor | Trend Category | Start Weight | End Weight | Avg Weight | Volatility (StdDev) | Slope |\n"
        trend_report += "|---|---|---|---|---|---|---|\n"
        for factor, data in sorted(factor_trends.items()):
            trend_report += (f"| `{factor}` | {data['trend']} | {data['start']:.2f} | {data['end']:.2f} | "
                             f"{data['avg']:.2f} | {data['volatility']:.2f} | {data['slope']:+.3f} |\n")

        # --- 6. Extract strategic recommendations ---
        bedrock_factors = [f"`{f}`" for f, d in factor_trends.items()
                           if ("Upward" in d['trend'] or d['trend'] == "Stable") and d['avg'] > 0.65]
        chameleon_factors = [f"`{f}`" for f, d in factor_trends.items()
                             if "Volatility" in d['trend'] and d['avg'] > 0.4]
        fading_factors = [f"`{f}`" for f, d in factor_trends.items()
                          if "Downward" in d['trend'] and d['end'] < 0.3]

        trend_report += "\n## Strategic Recommendations for the Next Research Phase\n"
        if bedrock_factors:
            trend_report += f"- **Bedrock Factors (Core Principles):** {', '.join(bedrock_factors)}. These are proven to be effective. Future heuristics should continue to build upon them.\n"
        if chameleon_factors:
            trend_report += f"- **Chameleon Factors (Context-Dependent):** {', '.join(chameleon_factors)}. Their oscillating weights suggest their usefulness depends on the situation. Propose designing conditional logic (e.g., based on system load) to harness their power more effectively.\n"
        if fading_factors:
            trend_report += f"- **Fading Factors (Candidates for Deprecation):** {', '.join(fading_factors)}. Their declining importance suggests they are likely redundant or ineffective for this problem. Consider removing them from active exploration to simplify the search space.\n"

        return trend_report

    # def run_experimenter_generation(self, final_guide: str, best_ind: dict, second_best_ind: dict) -> list:
    def run_experimenter_generation(self, final_guide: str, best_ind: dict) -> list:
        """Experimenter: Generate new generation population"""
        logging.info("Experimenter is generating the next generation of code...")

        # --- [FIX] Correctly parse all three sections from the Theorist's final guide ---
        final_table = ""
        proposal = ""
        problem_characterization = "No specific problem characterization provided in this iteration."  # Default value

        # Define all possible headers to look for
        proposal_headers = ["[Final Strategic Proposal]", "[Radical Strategic Proposal]", "[Meta-Strategy Report]"]
        char_header = "[Updated Problem Characterization]"

        # Find which, if any, proposal header is in the guide
        active_proposal_header = next((h for h in proposal_headers if h in final_guide), None)

        # Find the characterization header
        char_header_present = char_header in final_guide

        # Logic to split the string based on the headers found
        if active_proposal_header:
            # A proposal exists, so split by it first
            table_part, rest_part = final_guide.split(active_proposal_header, 1)
            final_table = table_part.strip()

            if char_header_present:
                # If characterization also exists, split the rest by it
                proposal_part, char_part = rest_part.split(char_header, 1)
                proposal = (active_proposal_header + proposal_part).strip()
                problem_characterization = (char_header + char_part).strip()
            else:
                # Only proposal was found, no characterization
                proposal = (active_proposal_header + rest_part).strip()
        elif char_header_present:
            # No proposal, but characterization exists
            table_part, char_part = final_guide.split(char_header, 1)
            final_table = table_part.strip()
            problem_characterization = (char_header + char_part).strip()
            proposal = "No specific proposal provided."  # Set a default for proposal
        else:
            # No headers found, assume the entire guide is the table
            final_table = final_guide.strip()
            proposal = "No specific proposal provided."

        # --- [END FIX] ---

        # Generate prompt with the new placeholder
        prompt = self.experimenter_prompt.format(
            user_generator_context=self.user_generator_context,
            final_strategy_table=final_table,
            proposal=proposal,
            updated_problem_characterization=problem_characterization,  # <-- Pass the newly parsed variable
            best_code=best_ind['code'],
            # second_best_code is commented out in your provided code
            anti_pattern_lib="\n".join([str(p) for p in self.anti_pattern_lib[-3:]]),
            func_name=self.func_name
        )

        # Generate new population
        responses = self.experimenter_llm.multi_chat_completion(
            messages_list=[[{"role": "user", "content": prompt}]],
            n=self.cfg.pop_size
        )
        return [self.response_to_individual(res, i) for i, res in enumerate(responses)]

    def create_checkpoint(self):
        """ Create a checkpoint before radical exploration."""
        logging.info("Creating a checkpoint before radical exploration...")
        self.checkpoint = {
            "elitist": copy.deepcopy(self.elitist),
            "strategy_table_str": copy.deepcopy(self.strategy_table_str),
            "best_obj_overall": self.best_obj_overall,
            "best_code_overall": self.best_code_overall,
            "best_code_path_overall": self.best_code_path_overall,
            "population": copy.deepcopy(self.population),
            "iteration": self.iteration
        }
        #
        self.is_in_radical_exploration = True
        self.radical_exploration_timer = self.radical_exploration_duration
        self.history["last_radical_was_successful"] = False

    def rollback(self):
        """Roll back to checkpoint state"""
        if not self.checkpoint:
            return

        logging.warning("Radical exploration failed. Rolling back to the last stable state.")
        self.elitist = self.checkpoint["elitist"]
        self.strategy_table_str = self.checkpoint["strategy_table_str"]
        self.best_obj_overall = self.checkpoint["best_obj_overall"]
        self.best_code_overall = self.checkpoint["best_code_overall"]
        self.best_code_path_overall = self.checkpoint["best_code_path_overall"]
        self.population = self.checkpoint["population"]
        self.iteration = self.checkpoint["iteration"]
        self.is_in_radical_exploration = False
        self.checkpoint = None

    def update_iter(self) -> None:
        """
        Update system state at the end of an iteration.
        This function handles finding the elite, checking for stagnation,
        managing the radical exploration lifecycle, and saving progress.
        """
        if not self.population:
            self.iteration += 1
            return

        successful_pop = [ind for ind in self.population if ind.get("exec_success")]
        if not successful_pop:
            logging.warning(f"No successful individuals in the population at iteration {self.iteration}.")
            self.iteration += 1
            return

        # Find the best individual and score of the current iteration
        best_individual_this_iter = min(successful_pop, key=lambda x: x["obj"])
        current_best_obj = best_individual_this_iter["obj"]
        avg_obj = np.mean([ind["obj"] for ind in successful_pop])

        # Record historical data for plotting
        self.history['iterations'].append(self.iteration)
        self.history['best_scores'].append(current_best_obj)
        self.history['avg_scores'].append(avg_obj)
        self.history['stagnation'].append(self.stagnation_counter)

        # --- [CORE FIX START] ---

        # Check for global progress and update the elitist
        improvement = self.best_obj_overall - current_best_obj
        if current_best_obj < self.best_obj_overall:
            logging.info(f"New best overall score found! {current_best_obj:.4f} (improvement of {improvement:.4f})")
            self.best_obj_overall = current_best_obj
            self.best_code_overall = best_individual_this_iter["code"]
            self.best_code_path_overall = best_individual_this_iter["code_path"]
            self.elitist = best_individual_this_iter

        # If no elite exists yet (e.g., first iteration), designate the current best as the elite
        if self.elitist is None:
            self.elitist = best_individual_this_iter

        # Handle the lifecycle of Radical Exploration
        if self.is_in_radical_exploration:
            self.radical_exploration_timer -= 1
            logging.info(f"In Radical Exploration mode. Timer: {self.radical_exploration_timer}")

            # Check if the radical exploration period has ended
            if self.radical_exploration_timer <= 0:
                logging.info("Radical exploration period finished. Evaluating success...")

                was_successful = False
                if self.checkpoint and self.elitist:
                    # Success is defined as the new elite being better than the elite at the checkpoint
                    if self.elitist["obj"] < self.checkpoint["elitist"]["obj"]:
                        logging.info("Radical exploration was successful! Committing to the new path.")
                        was_successful = True
                    else:
                        logging.warning("Radical exploration failed. Rolling back to the last stable state.")
                        self.rollback()  # Rollback handles resetting state, including elitist

                # Finalize the state after exploration ends
                if was_successful:
                    self.history["last_radical_was_successful"] = True

                # Whether successful or failed (rolled back), the exploration is over.
                self.checkpoint = None
                self.is_in_radical_exploration = False
                self.stagnation_counter = 0  # Reset stagnation counter to begin a new observation period.

        # Update stagnation counter ONLY if NOT in radical exploration
        else:
            if improvement < self.stagnation_threshold:
                self.stagnation_counter += 1
                logging.warning(
                    f"Performance improvement ({improvement:.4f}) is below stagnation threshold. Stagnation counter: {self.stagnation_counter}")
            else:
                # Any significant improvement resets the counter
                self.stagnation_counter = 0

        # --- [CORE FIX END] ---

        # Persistently save the current iteration's final knowledge (strategy table)
        if self.strategy_table_str:
            with open(self.weights_history_path, "a", encoding='utf-8') as f:
                f.write(f"\n--- Iteration {self.iteration} (FE: {self.function_evals}) ---\n")
                f.write(self.strategy_table_str + "\n")

            # Parse and store factor weights for plotting or metacognition
            try:
                parsed_table = parse_strategy_table(self.strategy_table_str)
                self.history['factor_weights'].append(parsed_table.get('factors', {}))
            except Exception as e:
                logging.error(f"Failed to parse and record strategy table for history: {e}")

        logging.info(
            f"Iteration {self.iteration} finished. Best obj this iter: {current_best_obj:.4f}. Best obj overall: {self.best_obj_overall:.4f}. Stagnation: {self.stagnation_counter}")
        self.iteration += 1

    def save_state(self, filename: str = "ares_state.pkl"):
        """Save current state to file"""
        state = {
            'cfg': self.cfg,
            'strategy_table_str': self.strategy_table_str,
            'elitist': self.elitist,
            'population': self.population,
            'iteration': self.iteration,
            'function_evals': self.function_evals,
            'best_obj_overall': self.best_obj_overall,
            'history': self.history,
            'stagnation_counter': self.stagnation_counter,
            'is_in_radical_exploration': self.is_in_radical_exploration,
            'radical_exploration_timer': self.radical_exploration_timer,
            'anti_pattern_lib': self.anti_pattern_lib
        }

        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"State saved to {filename}")

    def load_state(self, filename: str = "ares_state.pkl"):
        """
        Load state from file to resume operation.
        It loads core state variables but uses the current new working directory
        to store future outputs.
        """
        # Ensure a valid file path is provided
        if not os.path.exists(filename):
            logging.error(f"State file not found: {filename}. Cannot load state.")
            return

        logging.info(f"Loading state from {filename}...")
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)

            # --- Restore core state variables ---
            # self.cfg = state['cfg'] # Generally not recommended to restore config, as new runs may have new configuration overrides
            self.strategy_table_str = state['strategy_table_str']
            self.elitist = state['elitist']
            self.population = state['population']
            self.iteration = state['iteration']
            self.function_evals = state['function_evals']
            self.best_obj_overall = state['best_obj_overall']
            self.history = state['history']
            self.stagnation_counter = state['stagnation_counter']
            self.is_in_radical_exploration = state['is_in_radical_exploration']
            self.radical_exploration_timer = state['radical_exploration_timer']
            self.anti_pattern_lib = state.get('anti_pattern_lib', [])

            # --- [Core Modification] Reinitialize variables related to current run path ---
            # We inherit historical knowledge from the old run, but all future outputs
            # will be written to the new working directory.

            # 1. Set up history file path for new run
            self.weights_history_path = f"{self.workspace_dir}/weights_history.txt"

            # 2. "Migrate" loaded historical knowledge to new path
            #    First, we need to find content of history file from old state.
            #    A robust approach is that save_state should also save content of weights_history.txt.
            #    For simplicity, we assume old history file is in same directory as pkl file.
            old_history_path = os.path.join(os.path.dirname(filename), "weights_history.txt")
            if os.path.exists(old_history_path):
                with open(old_history_path, "r", encoding='utf-8') as f_old:
                    old_history_content = f_old.read()
                with open(self.weights_history_path, "w", encoding='utf-8') as f_new:
                    f_new.write(old_history_content)
                logging.info(f"Copied knowledge from old history file '{old_history_path}' to new path.")
            else:
                logging.warning(f"Old history file not found at '{old_history_path}'. Starting a new history file.")
                # Initialize empty file even if old one can't be found
                with open(self.weights_history_path, "w", encoding='utf-8') as f:
                    f.write(f"--- Resumed from Iteration {self.iteration} ---\n")

            # 3. Ensure file paths in all individuals point to new working directory
            for ind in self.population:
                # Regenerate paths but preserve original ID and iteration number
                role = ind.get("role", "experimenter")
                old_iter_match = re.search(r"iter_(\d+)", ind.get("code_path", ""))
                old_iter = old_iter_match.group(1) if old_iter_match else self.iteration

                role_dir = f"{self.workspace_dir}/{role}_outputs"
                os.makedirs(role_dir, exist_ok=True)

                file_name_base = f"iter_{old_iter}_id_{ind['response_id']}"
                ind["code_path"] = f"{role_dir}/{file_name_base}_code.py"
                ind["stdout_filepath"] = f"{role_dir}/{file_name_base}_stdout.txt"

            logging.info(f"State successfully loaded. Resuming from iteration {self.iteration}.")
            logging.info(f"All future outputs will be saved to: {self.workspace_dir}")

        except Exception as e:
            logging.critical(f"Failed to load state from {filename}. Error: {e}")
            # Recommended to exit on load failure rather than continuing with incomplete state
            raise RuntimeError("State loading failed, cannot continue.") from e

        # in ares.py
        # 请用下面这个函数，完整地、逐字地替换掉您代码中旧的 evolve 函数

    def evolve(self):
        """
        Main evolution loop of the ARES framework.
        It orchestrates the entire scientific discovery process, including reflection,
        experimentation, and evolution, while managing the system's state.
        """
        try:
            while self.function_evals < self.cfg.max_fe:
                logging.info(f"\n{'=' * 25} Starting Evolution: Iteration {self.iteration} {'=' * 25}")

                # -----------------------------------------------------------------
                # Step 1: Select key individuals from the current elite population
                # -----------------------------------------------------------------
                if not self.population:
                    raise RuntimeError("Population is empty. Cannot proceed with evolution.")
                best_ind, second_best_ind, worst_ind = self.select_key_individuals()

                # -----------------------------------------------------------------
                # Step 2: Theorist Phase 1 - Propose a new hypothesis
                # -----------------------------------------------------------------
                hypothetical_table, proposal = self.run_theorist_phase1_reflection(best_ind, worst_ind)

                # -----------------------------------------------------------------
                # Step 3: Critic Validation - Design and run experiments
                # -----------------------------------------------------------------
                critic_results = self.run_critic_validation(best_ind, hypothetical_table, proposal)

                # -----------------------------------------------------------------
                # Step 3.5: [ULTIMATE FIX] Instant Adoption of Critic's Breakthroughs
                # -----------------------------------------------------------------
                logging.info("Checking for breakthroughs from Critic's experiments...")

                # Use a new local variable to track the best individual discovered within this iteration's cycle.
                # It starts as the best from the previous generation.
                best_ind_this_iteration = best_ind

                # Get all evaluated mutants from the critic's experiments.
                all_critic_individuals = critic_results.get("evaluated_mutants", [])
                successful_critic_inds = [ind for ind in all_critic_individuals if ind.get("exec_success")]

                if successful_critic_inds:
                    # Find the best performing individual among the critic's mutants.
                    best_critic_ind = min(successful_critic_inds, key=lambda x: x["obj"])

                    # If the best mutant is better than our current best for this iteration...
                    if best_critic_ind["obj"] < best_ind_this_iteration["obj"]:
                        improvement = best_ind_this_iteration["obj"] - best_critic_ind["obj"]
                        logging.info(f"*** CRITIC'S BREAKTHROUGH! ***")
                        logging.info(
                            f"  - New best code from '{best_critic_ind['response_id']}' has outperformed the previous best.")
                        logging.info(f"  - Score: {best_critic_ind['obj']:.4f} (Improvement of {improvement:.4f})")
                        logging.info(
                            f"  - Instantly promoting this code to be the new 'best_ind' for this iteration.")

                        # 1. Update the local variable for the rest of this iteration's logic.
                        best_ind_this_iteration = best_critic_ind

                        # 2. CRITICAL: Immediately update the official, global ARES state.
                        # This ensures the breakthrough is officially recorded and not lost.
                        self.best_obj_overall = best_ind_this_iteration["obj"]
                        self.best_code_overall = best_ind_this_iteration["code"]
                        self.best_code_path_overall = best_ind_this_iteration["code_path"]
                        self.elitist = best_ind_this_iteration
                        self.stagnation_counter = 0  # A breakthrough immediately resets the stagnation counter.

                # -----------------------------------------------------------------
                # Step 4: Theorist Final Reflection - Formulate the final guide
                # -----------------------------------------------------------------
                final_guide = self.run_theorist_final_reflection(
                    (hypothetical_table, proposal),
                    critic_results,
                    best_ind_this_iteration  # Pass the definitive best individual of this cycle
                )

                # -----------------------------------------------------------------
                # Step 5: Parse and update the central strategy table
                # -----------------------------------------------------------------
                logging.info("Updating the central strategy table with the latest V-final...")
                try:
                    table_match = re.search(r"\[Final Strategy Table\](.*?)(\[|\Z)", final_guide,
                                            re.DOTALL | re.IGNORECASE)
                    if table_match:
                        new_strategy_table = table_match.group(1).strip()
                        if "ID" in new_strategy_table and "| Factor/Strategy |" in new_strategy_table:
                            self.strategy_table_str = new_strategy_table
                            logging.info("Central strategy table has been successfully updated.")
                        else:
                            logging.warning(
                                "Extracted content from [Final Strategy Table] does not appear to be a valid table. State will not be updated.")
                    else:
                        logging.warning(
                            "Could not find a [Final Strategy Table] section in the final guide. The state will not be updated for this iteration.")
                except Exception as e:
                    logging.error(f"An error occurred while updating the strategy table: {e}")

                # -----------------------------------------------------------------
                # Step 6: Experimenter generates the next generation of code (offspring)
                # -----------------------------------------------------------------
                offspring_population = self.run_experimenter_generation(final_guide, best_ind_this_iteration)

                # -----------------------------------------------------------------
                # Step 7: Evaluate the newly generated offspring population
                # -----------------------------------------------------------------
                evaluated_offspring = self.evaluate_population(offspring_population)
                self.function_evals += len(evaluated_offspring)

                # -----------------------------------------------------------------
                # Step 8: Perform survivor selection (Elitism) to build the next generation
                # -----------------------------------------------------------------
                logging.info("Performing survivor selection (Elitism) to form the next generation...")

                # Combine the previous generation's population with the new offspring.
                combined_population = self.population + evaluated_offspring

                # If the best individual of this iteration was a critic mutant, it's not yet in the combined_population.
                # We must add it to ensure it competes for a spot in the next generation's elite pool.
                if best_ind_this_iteration['role'] == 'critic':
                    combined_population.append(best_ind_this_iteration)

                # Deduplicate the combined population based on code, keeping only the best score for each unique code.
                unique_codes = {}
                for ind in combined_population:
                    if ind.get("exec_success"):
                        code = ind['code']
                        if code not in unique_codes or ind['obj'] < unique_codes[code]['obj']:
                            unique_codes[code] = ind

                # Sort the unique, successful individuals by their objective score.
                sorted_unique_population = sorted(unique_codes.values(), key=lambda x: x['obj'])

                # Select the top individuals to form the population for the next generation.
                next_pop_size = min(self.cfg.pop_size, len(sorted_unique_population))
                self.population = sorted_unique_population[:next_pop_size]

                logging.info(f"Next generation formed with {len(self.population)} elite individuals.")

                # -----------------------------------------------------------------
                # Step 9: Update iteration information (history, logging, etc.)
                # -----------------------------------------------------------------
                self.update_iter()

                # Save state periodically
                if self.iteration % 10 == 0:
                    self.save_state(f"ares_state_iter{self.iteration}.pkl")

        except KeyboardInterrupt:
            logging.info("User interrupted. Saving current state...")
            self.save_state("ares_interrupted.pkl")
        except Exception as e:
            logging.critical(f"A critical error occurred in the evolve loop: {e}", exc_info=True)
            self.save_state("ares_error.pkl")
            raise

        return self.best_code_overall, self.best_code_path_overall
