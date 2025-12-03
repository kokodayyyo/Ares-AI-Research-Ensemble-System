***

# ARES System User Guide

ARES (AI Research Ensemble System) is an automated algorithm evolution and discovery system powered by Large Language Models (LLMs). Through multi-role collaboration (Theorist, Critic, Experimenter), ARES automatically generates, evaluates, and optimizes code solutions for specified problems.

This document outlines the system configuration parameters, problem definition methods, and LLM interface settings.

## 1. Core System Configuration (`cfg/config.yaml`)

The system's core behavior is controlled via the `cfg/config.yaml` file. Below are the details for each key parameter:

### Environment & Parallelism
| Parameter | Description                                                                                                                                    |
| :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------|
| **`problem_name`** | The name of the current task (must match the folder name in `problems/`).                                                                      |
| **`description`** | A brief description of the current task.                                                                                                       |
| **`num_eval_workers`** | **CPU Core Count**. Specifies the number of CPU cores used for parallel code evaluation. Increasing this speeds up the evaluation process.     |
| **`timeout`** | **Execution Timeout** (in seconds). The maximum allowed time for a single code run. Execution is considered a failure if it exceeds this time. |
| **`obj_type`** | **Optimization direction**£¬either `min` (minimization) or `max` (maximization).                                                                                                                    |

### Evolutionary Parameters
| Parameter | Description |
| :--- | :--- |
| **`init_pop_size`** | **Initial Population Size**. The number of offspring generated in the very first generation of the evolutionary algorithm. |
| **`pop_size`** | **Population Size**. The number of offspring generated in each subsequent generation during the evolution process. |
| **`max_fe`** | **Max Function Evaluations**. The total number of offspring codes generated during the entire run. This serves as the stopping criterion for the system. |
| **`eval_runs`** | **Evaluation Runs**. For heuristic algorithms with unstable results, this sets how many times each code is run. The system uses the **minimum value** (for minimization problems) or the **average** to assess performance. |
| **`stagnation_threshold`** | **Stagnation Threshold**. The minimum improvement required to consider a new best value as valid (e.g., `1e-4`). Improvements below this are treated as stagnation. |

### Advanced Exploration & Reflection
| Parameter | Description |
| :--- | :--- |
| **`meta_reflection_cycle_period`** | **Meta-Reflection Period**. Sets the number of iterations between triggers of the Meta-Reflection mechanism to summarize historical strategies. |
| **`radical_exploration_trigger`** | **Radical Exploration Trigger**. The number of consecutive stagnant iterations required to trigger the Radical Exploration mode. |
| **`radical_exploration_duration`** | **Radical Exploration Duration**. The number of iterations the system remains in Radical Exploration mode to escape local optima. |

---

## 2. Problem Definition & Initialization

To run a new problem, you must configure the name in `config.yaml` and inject domain knowledge into the Prompt template.

### Key Step
Please edit the following file:
> **`prompts/ares/user_generator_theorist_init.txt`**

At the **very beginning** of this file, you must provide:
1.  **Detailed Problem Description**: Clearly define the target task.
2.  **Function Interface Definition**: List available APIs, data structures, or function signatures that must be implemented.(For widely studied classic problems, this step can be omitted, but for more complex problems, please be sure to provide the function interface definition.)
**The "Theorist" role will use the description provided here to initialize the strategy table and generate the first generation of code.**

---

## 3. LLM Interface Settings

ARES supports multiple LLM backends. API keys and model selections are configured in the following locations:

1.  **Model Specific Config**:
    *   Navigate to the `cfg/llm_client` directory.
    *   Modify the configuration files corresponding to the models you are using (e.g., GPT-4, Claude).

2.  **Global Model Selection**:
    *   In `cfg/config.yaml`, reference the configurations above to specify which model client is used for each role (Theorist, Critic, Experimenter).

---

## 4. Quick Start

1.  Configure your API Keys in `cfg/llm_client`.
2.  Fill in the problem description and interface details in `prompts/ares/user_generator_theorist_init.txt`.
3.  Modify `cfg/config.yaml` to set the `problem_name` and evolutionary parameters (e.g., `pop_size`, `max_fe`).
4.  Run the main program to start ARES.

