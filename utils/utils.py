import logging
import re
import inspect
import os


def file_to_string(filename: str) -> str:
    """Reads a file and returns its content as a string."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        return ""  # 返回空字符串以避免程序崩溃
    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}")
        return ""


def print_hyperlink(path: str, text: str = None) -> str:
    """Prints a hyperlink to a file or folder for convenient navigation in supported terminals."""
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"


def filter_traceback(s: str) -> str:
    """Extracts the traceback from a string, typically stdout."""
    lines = s.split('\n')
    filtered_lines = []
    in_traceback = False
    for line in lines:
        if line.strip().startswith('Traceback'):
            in_traceback = True
        if in_traceback:
            filtered_lines.append(line)
            if "Set the environment variable HYDRA_FULL_ERROR=1" in line:
                break
    return '\n'.join(filtered_lines)


def block_until_running(stdout_filepath: str, log_status: bool = False, iter_num: int = -1, response_id: int = -1):
    """
    Waits until the stdout file for a code run is no longer empty.
    This is a simple way to ensure an external process has started writing output.
    """
    while True:
        if os.path.exists(stdout_filepath) and os.path.getsize(stdout_filepath) > 0:
            if log_status:
                log_content = file_to_string(stdout_filepath)
                if 'Traceback' in log_content:
                    logging.warning(
                        f"Iteration {iter_num}: Code Run {response_id} may have an error! (see {print_hyperlink(stdout_filepath)})"
                    )
                else:
                    logging.info(
                        f"Iteration {iter_num}: Code Run {response_id} started successfully. (see {print_hyperlink(stdout_filepath)})"
                    )
            break


def extract_code_from_generator(content: str, block_name: str = "python") -> str:
    """
    Extracts Python code from a string, typically an LLM response.
    This version is designed to be robust and preserve important comments
    that might be located just before the main code block.
    """
    if not isinstance(content, str):
        return ""

    # --- [ULTIMATE FIX] ---
    # The core idea is to find the main code block and then prepend any
    # relevant lines (imports or comments) that immediately precede it.

    # 1. Define a pattern to find the main code block and any preceding lines.
    # This pattern captures:
    #   - (group 1): Any lines before the block (optional, non-greedy)
    #   - (group 2): The code inside the ```python ... ``` block
    pattern = re.compile(r"(.*?)(?:```python\n(.*?)\n```)", re.DOTALL)
    match = pattern.search(content)

    if match:
        preceding_text = match.group(1).strip()
        code_in_block = match.group(2).strip()

        # 2. Analyze the preceding text to extract relevant lines.
        # Relevant lines are comments (#) and imports (import, from).
        header_lines = []
        for line in preceding_text.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith('#') or \
                    stripped_line.startswith('import ') or \
                    stripped_line.startswith('from '):
                header_lines.append(stripped_line)

        header_str = "\n".join(header_lines)

        # 3. Combine the extracted header with the code from the block.
        # This ensures that both the descriptive comment and any top-level imports are preserved.
        if header_str:
            return f"{header_str}\n\n{code_in_block}"
        else:
            return code_in_block

    # 4. Fallback: If no ```python``` block is found, try a generic one.
    generic_match = re.search(r"```(.*?)```", content, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    # 5. Last resort: If no blocks at all, return the original content if it looks like code.
    if 'def ' in content or 'import ' in content:
        return content.strip()

    return ""  # Return empty if no code-like content is found


def filter_code(code_string: str) -> str:
    """
    Removes function signature, docstrings, and import statements for cleaner display in prompts.
    This is a simplified version and might need to be made more robust.
    """
    if not isinstance(code_string, str): return ""

    lines = code_string.split('\n')
    filtered_lines = []
    in_docstring = False

    # 找到函数定义的起始行
    def_line_found = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('def '):
            def_line_found = True
            continue  # 跳过函数定义行
        if not def_line_found:
            continue

        if '"""' in stripped_line or "'''" in stripped_line:
            in_docstring = not in_docstring
            continue  # 跳过文档字符串的开关行

        if in_docstring:
            continue

        # 跳过导入语句
        if stripped_line.startswith('import ') or stripped_line.startswith('from '):
            continue

        filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def get_heuristic_name(module, possible_names: list[str]):
    """Finds a function by name in a given module from a list of possibilities."""
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name
    return None
