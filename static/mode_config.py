# static/mode_config.py
from features_ga import ga_feature_selection
from params_ga import ga_param_search
from structure_ga import ga_structure_search
from optimize_ga import ga_optimize_structure

MODE_CONFIG = {
    "features": {
        "ga_func": ga_feature_selection,
        "extra_args": lambda params, cols: {"cols": cols, "mutation_rate": params.get("MUTATION_RATE")},
        "postprocess": lambda result, cols: {**result, "features": [cols[i] for i, b in enumerate(result["mask"]) if b == 1]},
        "multi": False
    },
    "params": {
        "ga_func": ga_param_search,
        "extra_args": lambda params, cols: {},
        "postprocess": lambda result, cols: result,
        "multi": False
    },
    "structure": {
        "ga_func": ga_structure_search,
        "extra_args": lambda params, cols: {},
        "postprocess": lambda result, cols: result,
        "multi": False
    },
    "opt": {
        "ga_func": ga_optimize_structure,
        "extra_args": lambda params, cols: {},
        "postprocess": lambda results, cols: results,
        "multi": True
    }
}