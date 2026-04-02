def detect_innovation_type(proposal: str) -> str:
    """Detect the type of innovation"""
    proposal_lower = proposal.lower()
    structural_keywords = ["structural", "feature", "component", "mechanism"]
    parametric_keywords = ["parametric", "parameter", "weight", "value", "tuning"]
    mixed_keywords = ["hybrid", "mixed", "both", "combination"]

    # Prioritize the detection of hybrid innovation
    if any(kw in proposal_lower for kw in mixed_keywords):
        return "Mixed Innovation"

    structural_count = sum(1 for kw in structural_keywords if kw in proposal_lower)
    parametric_count = sum(1 for kw in parametric_keywords if kw in proposal_lower)

    if structural_count > 0 and parametric_count > 0:
        return "Mixed Innovation"
    elif structural_count > 0:
        return "Structural Innovation"
    elif parametric_count > 0:
        return "Parametric Innovation"
    return "Unknown Innovation Type"


def get_innovation_instruction(innovation_type: str) -> str:
    """Generate experimental design instructions based on the type of innovation"""
    instructions = {
        "Structural Innovation": (
            "**Experimental Design Requirement:**\n"
            "You MUST design a Feature Ablation experiment. "
            "Provide at least 3 versions:\n"
            "1. Original Best Code (as baseline)\n"
            "2. Version with the new structural feature removed\n"
            "3. Version with only the new structural feature\n"
            "4. Version combining the new feature with existing ones"
        ),
        "Parametric Innovation": (
            "**Experimental Design Requirement:**\n"
            "You MUST design a Parameter Sensitivity Scan. "
            "Provide at least 4 versions:\n"
            "1. Original Best Code (as baseline)\n"
            "2. Version with parameter reduced by 50%\n"
            "3. Version with parameter increased by 50%\n"
            "4. Version with parameter set to extreme value (0 or 1)\n"
            "5. Version with optimized parameter combination"
        ),
        "Mixed Innovation": (
            "**Experimental Design Requirement:**\n"
            "You MUST design a Hybrid Experiment. "
            "Provide at least 5 versions:\n"
            "1. Original (baseline)\n"
            "2. Without new structure\n"
            "3. With new structure only\n"
            "4. With new parameters only\n"
            "5. With both new structure and parameters"
        ),
        "default": (
            "**Experimental Design Requirement:**\n"
            "Design comprehensive experiments to validate the proposal. "
            "Include at least 4 mutant versions covering different aspects."
        )
    }
    return instructions.get(innovation_type, instructions["default"])
