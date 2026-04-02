import re


def parse_strategy_table(table_str: str) -> dict:
    """Parse the Markdown of the strategy table into structured data"""
    parsed = {"factors": {}, "meta": {}}
    lines = table_str.strip().split('\n')

    # Extract the table header
    headers = [h.strip() for h in lines[1].split('|')[1:-1]]

    # Parse data rows
    for line in lines[3:]:
        if not line.startswith('|'):
            continue

        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) != len(headers):
            continue

        factor_id = cells[0]
        parsed["factors"][factor_id] = dict(zip(headers[1:], cells[1:]))


        if 'Weight P(H)' in parsed["factors"][factor_id]:
            try:
                parsed["factors"][factor_id]['Weight P(H)'] = float(
                    parsed["factors"][factor_id]['Weight P(H)'])
            except:
                parsed["factors"][factor_id]['Weight P(H)'] = 0.0

    # Extract metadata
    if "Bedrock Factors" in table_str:
        parsed["meta"]["bedrock"] = re.findall(r"Bedrock Factors: ([\w\s,]+)", table_str)
    if "Chameleon Factors" in table_str:
        parsed["meta"]["chameleon"] = re.findall(r"Chameleon Factors: ([\w\s,]+)", table_str)

    return parsed
