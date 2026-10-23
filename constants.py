"""
Project-wide constants shared across all ProteinMPNN modules.

Centralising the alphabet here ensures a single source of truth: every module
that needs the amino-acid encoding imports from this file instead of
duplicating the string literal.
"""

# Model encoding alphabet: 20 standard amino acids + 'X' for unknown/masked residues.
# This order is baked into the model weights and must not be changed.
PROTEIN_ALPHABET: str = 'ACDEFGHIKLMNPQRSTVWYX'

# Alphabet with gap marker '-' appended, used for sequence validation in datasets.
PROTEIN_ALPHABET_WITH_GAP: str = PROTEIN_ALPHABET + '-'

# Full protein backbone extracted when ca_only is False. Order matters: tied_featurize
# stacks these coordinates along the atom axis and indexes them positionally, so the
# default parser and the downstream stack must always agree. A tuple keeps it immutable.
FULL_BACKBONE_ATOM_NAMES: tuple[str, ...] = ('N', 'CA', 'C', 'O')

# Number of numeric chain IDs probed after the alphabetic ones, i.e. '0'..'299'.
# Large complexes and those from mmCIF sources sometimes use numeric chain labels.
NUMERIC_CHAIN_ID_COUNT: int = 300
