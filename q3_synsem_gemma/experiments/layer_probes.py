"""
Layer Probing Experiment Definitions.
Defines tasks to run across all 27 layers under baseline unablated conditions.
"""

PROBE_TASKS = [
    {"name": "blimp_battery", "type": "syntax_minimal_pairs"},
    {"name": "stsb_layerwise", "type": "semantic_similarity_layers"},
    {"name": "wic_polysemy", "type": "polysemy_layers"},
    {"name": "hellaswag", "type": "commonsense_plausibility"},
    {"name": "chomsky_quad", "type": "chomskyan_dissociation"},
    {"name": "rsa_interlayer", "type": "interlayer_rsa"},
]
