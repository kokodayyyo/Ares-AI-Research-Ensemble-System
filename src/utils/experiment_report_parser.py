import re


def parse_experiment_report(report_str: str) -> dict:
    """Parse the experimental report into structured data"""
    parsed = {
        "objective": "",
        "innovation_type": "",
        "baseline": {"score": 0.0, "delta": 0.0},
        "mutants": []
    }

    # Analyze the types of innovation
    innovation_match = re.search(r"\*\*Innovation Type\*\*: (.+)", report_str)
    if innovation_match:
        parsed["innovation_type"] = innovation_match.group(1)

    # Analyze the target
    obj_match = re.search(r"Objective: To empirically validate the Theorist's proposal: \"(.+?)\"", report_str)
    if obj_match:
        parsed["objective"] = obj_match.group(1)

    # Parse table data
    table_lines = []
    in_table = False
    for line in report_str.split('\n'):
        if line.startswith('| Test ID'):
            in_table = True
            continue
        if in_table and line.startswith('|'):
            table_lines.append(line)

    # Handle table content
    for line in table_lines[2:]:  # Skip the separator line of the table header
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) < 6:
            continue

        test_data = {
            "id": cells[0],
            "experiment_type": cells[1],
            "factor_tested": cells[2],
            "modification_details": cells[3],
            "performance_score": float(cells[4]) if cells[4] else 0.0,
            "delta": float(cells[5]) if cells[5] else 0.0,
            "conclusion": cells[6] if len(cells) > 6 else ""
        }

        if test_data["id"] == "M0":
            parsed["baseline"] = test_data
        else:
            parsed["mutants"].append(test_data)

    return parsed
