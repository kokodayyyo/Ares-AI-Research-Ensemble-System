
def parse_init_response(response_text: str) -> dict:
    """
    Parses the full output from the init-phase Theorist,
    returning strategy table and the rest as analysis_text.
    """
    import re
    # Regular expression matching for markdown code blocks, non-greedy mode
    table_pattern = r"```markdown(.*?)```"
    match = re.search(table_pattern, response_text, re.DOTALL)

    if match:
        strategy_table = f"```markdown{match.group(1).strip()}\n```"
        analysis_text = re.sub(table_pattern, '', response_text, count=1, flags=re.DOTALL).strip()
    else:
        strategy_table = ""
        analysis_text = response_text.strip()

    return {"analysis_text": analysis_text, "strategy_table": strategy_table}