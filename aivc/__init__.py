# -*- coding: utf-8 -*-
"""
AIVC: Mechanistic Multi-Omics & Grey-Box Temporal Virtual Cell World Model.
Clean public package interface wrapping func_python_scripts/.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FUNC_DIR = os.path.join(_ROOT, "func_python_scripts")
if _FUNC_DIR not in sys.path:
    sys.path.insert(0, _FUNC_DIR)

from aivc_flagship import (
    load_flagship_priors,
    compute_omnipath_causal_footprints,
    compute_ensemble_uq_scores,
    kalman_assimilate_state,
    assess_multi_organ_and_retina_safety,
)
from pkb import build_full_prior_atlas, build_pathway_crosstalk_network
from pgnn import build_gene_pathway_bipartite_tensors
from chem_interface import (
    compute_admet_druglikeness,
    encode_smiles_multimodal,
    map_molecules_to_targets_and_pathways,
)

__version__ = "1.1.0"
__all__ = [
    "load_flagship_priors",
    "compute_omnipath_causal_footprints",
    "compute_ensemble_uq_scores",
    "kalman_assimilate_state",
    "assess_multi_organ_and_retina_safety",
    "build_full_prior_atlas",
    "build_pathway_crosstalk_network",
    "build_gene_pathway_bipartite_tensors",
    "compute_admet_druglikeness",
    "encode_smiles_multimodal",
    "map_molecules_to_targets_and_pathways",
]
