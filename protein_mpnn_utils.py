from __future__ import print_function
import json, time, os, sys, glob
import shutil
import string
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from collections.abc import Sequence
from typing import Any, NamedTuple
from constants import (
    FULL_BACKBONE_ATOM_NAMES,
    NUMERIC_CHAIN_ID_COUNT,
    PROTEIN_ALPHABET,
    PROTEIN_ALPHABET_WITH_GAP,
)

#A number of functions/classes are adopted from: https://github.com/jingraham/neurips19-graph-protein-design

def parse_fasta(filename, limit=-1, omit=[]):
    header = []
    sequence = []
    lines = open(filename, "r")
    for line in lines:
        line = line.rstrip()
        if line[0] == ">":
            if len(header) == limit:
                break
            header.append(line[1:])
            sequence.append([])
        else:
            if omit:
                line = [item for item in line if item not in omit]
                line = ''.join(line)
            line = ''.join(line)
            sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    return np.array(header), np.array(sequence)

def _scores(
    integer_encoded_sequence: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability of the observed residue, averaged over masked positions.

    For every sequence, look up the model's log-probability of the residue actually
    present in ``integer_encoded_sequence`` at each position, keep only the positions
    selected by ``mask``, and average them into one score per sequence. Lower values mean
    the model assigns a higher probability to the observed sequence.

    Args:
        integer_encoded_sequence (torch.Tensor): integer residue-type indices of the
            observed sequence, shape = [..., L], dtype int64. Values index into
            ``PROTEIN_ALPHABET``. Describes either the native sequence or a sampled one.
        log_probs (torch.Tensor): per-position log-softmax probabilities, shape =
            [..., L, num_amino_acid_types] with last dim = 21 (alphabet size), dtype
            float32. Must already be log-probabilities (see Notes).
        mask (torch.Tensor): per-position mask, shape = [..., L], dtype float32.
            1.0 for positions included in the score (valid, designable residues);
            0.0 for padding or excluded positions.

    Returns:
        torch.Tensor: one score per leading index of ``integer_encoded_sequence``,
            shape = [...], dtype float32: the mean per-position negative log-likelihood
            over masked positions, i.e.
            sum(nll * mask, dim=-1) / sum(mask, dim=-1), where ``nll`` is the per-position
            value from ``torch.nn.NLLLoss(reduction='none')``. Lower is better: a smaller
            score means the model assigns a higher probability to the observed sequence.

    Shape:
        In this codebase the leading axis is the batch size B, so a typical call uses
        integer_encoded_sequence shape = [B, L], log_probs shape = [B, L, 21] and mask
        shape = [B, L], returning shape = [B]. An arbitrary number of leading axes is
        supported: they are flattened before scoring and restored afterwards.

    Notes:
        ``torch.nn.NLLLoss`` expects log-probabilities, so ``log_probs`` must come from
        a log-softmax (as produced by ProteinMPNN's forward pass), not raw logits. Every
        ``mask`` slice must sum to a positive value, otherwise the division yields NaN.

    Examples:
        >>> integer_encoded_sequence = torch.tensor([[0, 1, 2]])
        >>> log_probs = torch.log_softmax(torch.randn(1, 3, 21), dim=-1)
        >>> mask = torch.ones(1, 3)
        >>> _scores(integer_encoded_sequence, log_probs, mask).shape
        torch.Size([1])
    """
    # Per-position loss with no reduction: one NLL value per (sequence, position) pair
    criterion: torch.nn.NLLLoss = torch.nn.NLLLoss(reduction='none')
    # Flatten the leading axes onto a single batch axis so NLLLoss consumes the whole
    # batch in one call: log_probs [..., L, 21] -> [-1, 21] and
    # integer_encoded_sequence [..., L] -> [-1], then restore the per-position losses
    # to the sequence's shape [..., L]
    loss: torch.Tensor = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        integer_encoded_sequence.contiguous().view(-1)
    ).view(integer_encoded_sequence.size())
    # Zero out the masked-off positions, then average over the selected ones along L:
    # loss and mask are [..., L] -> scores is [...]
    scores: torch.Tensor = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    # One negative log probability per sequence, shape = [...]
    return scores

def _S_to_seq(integer_encoded_sequence: torch.Tensor, mask: torch.Tensor) -> str:
    """Convert integer-encoded residue indices into a one-letter amino-acid sequence.

    Positions whose ``mask`` value is 0 are skipped, so only the residues belonging to
    the extracted sequence are kept. This is how a full multi-chain sequence is split
    back into per-chain strings, e.g. for FASTA output.

    Args:
        integer_encoded_sequence (torch.Tensor): integer residue-type indices of one
            sequence, shape = [L], dtype int64. Values index into ``PROTEIN_ALPHABET``.
        mask (torch.Tensor): per-position selector, shape = [L], dtype float32. Positions
            with a value > 0 are kept; 0.0 positions (padding, or residues of other
            chains) are dropped.

    Returns:
        str: one-letter amino-acid sequence whose length equals the number of positions
            with ``mask > 0``.

    Examples:
        >>> integer_encoded_sequence = torch.tensor([0, 1, 2])
        >>> mask = torch.tensor([1.0, 0.0, 1.0])
        >>> _S_to_seq(integer_encoded_sequence, mask)
        'AD'
    """
    # Single source of truth for the residue alphabet: positional index -> one-letter code
    alphabet: str = PROTEIN_ALPHABET
    # Keep only the positions selected by the mask (value > 0), map each residue-type index
    # to its one-letter code, and join the letters into the output string
    seq: str = ''.join([
        alphabet[residue_type_index]
        for residue_type_index, position_mask in zip(
            integer_encoded_sequence.tolist(), mask.tolist()
        )
        if position_mask > 0
    ])
    # One-letter sequence of the selected residues
    return seq

def parse_PDB_biounits(
    pdb_path: str,
    backbone_atom_names: Sequence[str] = FULL_BACKBONE_ATOM_NAMES,
    chain_id: str | None = None,
) -> tuple[np.ndarray, str] | tuple[None, None]:
    '''
    Parse one chain from a PDB file and extract atom coordinates plus the sequence.

    Args:
        pdb_path: path to the PDB file (text format).
        backbone_atom_names: atom names to extract, in output order.
                        Default FULL_BACKBONE_ATOM_NAMES, i.e. ['N','CA','C','O'] (full protein backbone, required by the model).
        chain_id: single chain ID to extract (e.g. 'A'). If None, atoms from ALL chains are returned.

    Returns:
        coords_array: numpy array of shape (n_residues, len(backbone_atom_names), 3), 0-based residue indexing.
                    Residues/atoms missing from the file are filled with NaN.
        seq: the amino-acid sequence in one-letter codes,
                    with '-' for missing/unknown residues.

        If no ATOM record matches the requested chain, returns (None, None).

    Notes:
        - Residue numbers are converted from 1-based PDB numbering to 0-based.
        - Insertion codes (e.g. residue "100A") are kept as separate sub-entries, sorted after the base residue.
        - Selenomethionine (MSE, a HETATM record) is converted to standard MET and included.
    '''

    # single source of truth: 1-letter code -> 3-letter residue name.
    # '-'/GAP marks gaps and unknown residues.
    # Only key lookups are performed on this mapping, so entry order does not
    # affect correctness; the ARNDC... order is kept for readability and to
    # match the upstream ProteinMPNN convention. The model's actual encoding
    # alphabet is defined in constants.PROTEIN_ALPHABET, not here.
    amino_acid_one_to_three = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', '-': 'GAP',
    }

    # inverse mapping: 3-letter residue name -> 1-letter code (e.g. 'ALA'->'A', 'GAP'->'-')
    amino_acid_three_to_one: dict[str, str] = {three: one for one, three in amino_acid_one_to_three.items()}

    # residue_number -> insertion_code -> atom_name -> xyz; 0-based residue numbering, e.g.
    #   {
    #       0: {
    #           '': {
    #               'N': np.array([33.142, 21.390, 56.357]),    # shape (3,)
    #               'CA': np.array([33.003, 22.335, 57.475]),   # shape (3,)
    #               ...
    #           },
    #       },
    #   }
    atom_coords_dict: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    # residue_number -> insertion_code -> 3-letter residue name, e.g.
    #   {
    #       0: {'': 'PRO'},   # residue 0, no insertion code
    #   }
    residue_names_dict: dict[int, dict[str, str]] = {}

    # lowest/highest residue numbers seen; sentinels make the first min/max take effect
    min_residue_number: int = 10**6
    max_residue_number: int = -(10**6)
    
    # read the PDB file line by line in binary mode
    with open(pdb_path, "rb") as pdb_file:
        for line in pdb_file:
            # decode bytes to text (ignoring undecodable characters) and strip the trailing newline
            line = line.decode("utf-8", "ignore").rstrip()

            # selenomethionine (MSE) is a MET-like residue recorded as HETATM; rewrite it as standard ATOM/MET
            if line[:6] == "HETATM" and line[17:17+3] == "MSE":
                # change the record type so the ATOM branch below accepts this line
                line = line.replace("HETATM", "ATOM  ")
                # rename the residue to standard methionine
                line = line.replace("MSE", "MET")

            # only coordinate records of type ATOM are parsed
            if line[:4] == "ATOM":
                # chain ID occupies column 22 (0-based index 21)
                ch = line[21:22]
                # process the line only if it belongs to the requested chain (or no chain filter is set)
                if ch == chain_id or chain_id is None:
                    # atom name occupies columns 13-16, stripped of padding spaces
                    atom_name = line[12:12+4].strip()
                    # residue name (3 letters) occupies columns 18-20
                    residue_name = line[17:17+3]
                    # residue sequence number + optional insertion code occupy columns 23-27
                    residue_number = line[22:22+5].strip()
                    # x, y, z are fixed-width 8-character fields starting at columns 31, 39, 47
                    x, y, z = [float(line[i:(i+8)]) for i in [30, 38, 46]]

                    # insertion codes (e.g. residue "100A") appear as a trailing letter on the residue field
                    if residue_number[-1].isalpha():
                        # split into insertion code + 0-based residue number
                        insertion_code, residue_number = residue_number[-1], int(residue_number[:-1])-1
                    else:
                        # no insertion code; convert the 1-based PDB numbering to 0-based
                        insertion_code, residue_number = "", int(residue_number)-1
                    # update the lowest residue number seen
                    min_residue_number = min(min_residue_number, residue_number)
                    # update the highest residue number seen
                    max_residue_number = max(max_residue_number, residue_number)
                    # create the per-residue dict on first encounter
                    if residue_number not in atom_coords_dict:
                        atom_coords_dict[residue_number] = {}
                    # insertion-code sub-level (handles residues that share the same number)
                    if insertion_code not in atom_coords_dict[residue_number]:
                        atom_coords_dict[residue_number][insertion_code] = {}
                    # mirror structure for the sequence dict
                    if residue_number not in residue_names_dict:
                        residue_names_dict[residue_number] = {}
                    # record the residue's 3-letter name once per (residue number, insertion code) pair
                    if insertion_code not in residue_names_dict[residue_number]:
                        residue_names_dict[residue_number][insertion_code] = residue_name

                    # keep only the first occurrence of each atom_name (handles duplicate/alternate locations)
                    if atom_name not in atom_coords_dict[residue_number][insertion_code]:
                        # store the atom_name's (x, y, z) as a small numpy array
                        atom_coords_dict[residue_number][insertion_code][atom_name] = np.array([x, y, z])

    # no ATOM record matched the requested chain: the sentinels are never updated,
    # so min_residue_number stays above max_residue_number
    if min_residue_number > max_residue_number:
        # signal "chain not found" to the caller
        return None, None

    # build the output: collect one-letter amino acids and flat atom coordinates
    # one character per residue, e.g. ['H', 'M', 'P', 'E', ...] -> 'HMPE...'
    one_letter_amino_acids: list[str] = []
    # xyz arrays in (residue, backbone_atom_names) order, flattened so that
    # reshape(-1, len(backbone_atom_names), 3) rebuilds [n_residues, n_atoms, 3], e.g.
    #   [np.array([33.142, 21.390, 56.357]),   # residue 0, 'N'
    #    np.array([33.003, 22.335, 57.475]),   # residue 0, 'CA'
    #    np.array([32.424, 21.668, 58.723]),   # residue 0, 'C'
    #    np.array([31.745, 20.640, 58.632]),   # residue 0, 'O'
    #    np.array([nan, nan, nan]), ...]       # missing atoms/residues are NaN-filled
    flat_atom_coordinates: list[np.ndarray] = []
    # walk every residue number from lowest to highest (gaps get filled below)
    for residue_number in range(min_residue_number, max_residue_number+1):
        # residue present: convert 3-letter name directly to 1-letter code
        if residue_number in residue_names_dict:
            for insertion_code in sorted(residue_names_dict[residue_number]):
                three_letter = residue_names_dict[residue_number][insertion_code]
                one_letter_amino_acids.append(amino_acid_three_to_one.get(three_letter, '-'))
        # residue number missing entirely: use '-' for gap
        else:
            one_letter_amino_acids.append('-')
        # residue has coordinates: emit them in the requested atom_name order
        if residue_number in atom_coords_dict:
            for insertion_code in sorted(atom_coords_dict[residue_number]):
                for atom_name in backbone_atom_names:
                    # atom_name present: use its stored coordinates
                    if atom_name in atom_coords_dict[residue_number][insertion_code]:
                        flat_atom_coordinates.append(atom_coords_dict[residue_number][insertion_code][atom_name])
                    # atom_name absent: fill with NaN so the output shape stays consistent
                    else:
                        flat_atom_coordinates.append(np.full(shape=3, fill_value=np.nan))
        # residue missing from the coordinate data: fill all requested backbone_atom_names with NaN
        else:
            for atom_name in backbone_atom_names:
                flat_atom_coordinates.append(np.full(shape=3, fill_value=np.nan))
    # reshape flat_atom_coordinates to (n_residues, n_atoms, 3); join chars into one sequence string
    return np.array(flat_atom_coordinates).reshape(-1, len(backbone_atom_names), 3), "".join(one_letter_amino_acids)


def parse_PDB(
    path_to_pdb: str,
    input_chain_list: list[str] | None = None,
    ca_only: bool = False,
) -> list[dict[str, Any]]:
    '''
    Parse a single PDB file into the ``seq_chain_X`` / ``coords_chain_X`` dict format.

    Candidate chains are probed one by one with ``parse_PDB_biounits`` and only those that
    actually contain matching ATOM records are kept, so a file with unused chain IDs does
    not produce empty entries. Coordinates are stored as plain Python lists rather than
    NumPy arrays, which keeps the returned dicts JSON-serialisable; ``tied_featurize``
    converts them back with ``np.array``.

    Args:
        path_to_pdb: path to the PDB file (text format).
        input_chain_list: chain IDs to parse, e.g. ``['A', 'B']``. Duplicate IDs are
            ignored. When None (default),
            every candidate name is probed exhaustively: ``A``-``Z``, ``a``-``z``, then the
            numeric names ``'0'``-``'299'``.
        ca_only: if True, store only the C-alpha atom per residue; otherwise the full
            backbone ``['N', 'CA', 'C', 'O']`` is extracted.

    Returns:
        list[dict[str, Any]]: one dict per parsed file; the outer loop currently runs over a
        single file, so the list normally holds exactly one entry. Keys are inserted chain by
        chain, in the order the chains were probed. For 6EHB parsed with chains A and C and
        ``ca_only=False`` the entry looks like (sequences and coordinate lists truncated)::

            [
                {
                    'seq_chain_A': 'DGINQSGD...AIGLRYDF',                 # one-letter seq, '-' = gap
                    'coords_chain_A': {
                        'N_chain_A': [[-19.25, 70.285, 67.383], ..., [-21.355, 58.508, 65.054]],
                        'CA_chain_A': [[-20.692, 70.114, 67.754], ..., [-21.536, 57.311, 65.859]],
                        'C_chain_A': [[-21.367, 71.52, 67.682], ..., [-21.986, 57.626, 67.287]],
                        'O_chain_A': [[-20.7, 72.627, 67.739], ..., [-21.707, 58.713, 67.813]],
                    },                                                    # each array [n_res, 3]
                    'seq_chain_C': 'DGINQSGD...AIGLRYDF',                 # same block for chain C
                    'coords_chain_C': {
                        'N_chain_C': [[-21.542, 35.964, 55.216], ..., [-17.101, 38.772, 66.224]],
                        'CA_chain_C': [[-22.53, 35.35, 56.164], ..., [-17.514, 39.701, 67.273]],
                        'C_chain_C': [[-23.242, 34.222, 55.387], ..., [-18.975, 40.035, 67.034]],
                        'O_chain_C': [[-23.35, 34.196, 54.056], ..., [-19.63, 39.996, 65.956]],
                    },
                    'name': '6EHB',                                       # file name minus '.pdb'
                    'num_of_chains': 2,                                   # chains parsed from file
                    'seq': 'DGINQSGD...AIGLRYDFDGINQSGD...AIGLRYDF',      # display only, see Notes
                },
            ]

        With ``ca_only=True``, ``'coords_chain_<ID>'`` holds only ``'CA_chain_<ID>'``. Atoms
        missing from the file are stored as ``[nan, nan, nan]``; chains not present in the
        file are omitted entirely, so their ``'seq_chain_<ID>'`` / ``'coords_chain_<ID>'``
        keys are absent too.

    Notes:
        ``'name'`` is the bare file name with the ``'.pdb'`` extension removed, taken with
        ``os.path.basename`` so the separator convention of the running platform applies,
        e.g. ``'inputs/PDB_monomers/pdbs/5L33.pdb'`` -> ``'5L33'``.

        ``'seq'`` is only a flattened concatenation of all ``'seq_chain_<ID>'`` values, kept
        for display, logging, and length bookkeeping. It is never decoded position by
        position: ``tied_featurize`` rebuilds the per-residue sequence tensor ``S`` by
        walking the chains itself and reading ``pdb_dict['seq_chain_<ID>']`` (masked chains
        first, then visible ones), applying its own gap-to-``X`` substitution on the way.
        From ``'seq'`` the model only consumes its length, once, to derive each structure's
        total residue count and the padded ``L`` axis. ``StructureDatasetPDB`` reads it the
        same way, for alphabet validation, ``max_length`` filtering, and verbose printing.
        A ``'seq'`` whose chain order merely mirrors the probe order therefore cannot
        influence the design: it may differ from the internal chain layout without
        consequence, as long as its length stays equal to the sum of the chain lengths.

    Examples:
        >>> parse_PDB('inputs/PDB_monomers/pdbs/5L33.pdb', input_chain_list=['A'], ca_only=True)
        [{'seq_chain_A': 'HMPEEEKAARLF...NGRIKRVTITHV',
          'coords_chain_A': {'CA_chain_A': [[36.936, 18.773, 53.168], ..., [3.735, 10.392, 57.071]]},
          'name': '5L33',
          'num_of_chains': 1,
          'seq': 'HMPEEEKAARLF...NGRIKRVTITHV'}]
    '''
    # num_parsed_structures: counter of successfully parsed structures, reported just before returning
    num_parsed_structures: int = 0
    # list that holds the parsed dict for each PDB file; this is the return value
    pdb_dict_list: list[dict[str, Any]] = []
    # candidate chain IDs: uppercase A-Z followed by lowercase a-z (52 single characters)
    alphabetic_chain_ids: list[str] = list(string.ascii_uppercase + string.ascii_lowercase)
    # second layer: numeric chain IDs '0'..'299' for complexes that use numeric labels
    numeric_chain_ids: list[str] = [str(item) for item in range(NUMERIC_CHAIN_ID_COUNT)]
    # combined candidate list (352 entries); each ID is probed to see if the chain exists
    candidate_chain_ids: list[str] = alphabetic_chain_ids + numeric_chain_ids
     
    # if the user explicitly supplies chains, use only those (faster than exhaustive search)
    if input_chain_list:
        # drop duplicate chain IDs (keeps the caller's order) and copy the list so callers are never mutated
        candidate_chain_ids = list(dict.fromkeys(input_chain_list))
 
    # list of files to parse (single file here; loop structure supports multi-biological-unit files)
    biounit_names: list[str] = [path_to_pdb]
    for biounit in biounit_names:
        # dict holding all parsed info for the current PDB file
        pdb_dict: dict[str, Any] = {}
        # number of chains successfully parsed from this file
        num_parsed_chains: int = 0
        # concatenated sequence string from all chains in the file
        concat_seq: str = ''
        # choose which atom types to extract based on the ca_only flag (independent of the chain)
        if ca_only:
            backbone_atom_names: Sequence[str] = ('CA',)
        else:
            backbone_atom_names: Sequence[str] = FULL_BACKBONE_ATOM_NAMES
        # iterate over candidate chain IDs and try to extract coordinates for each
        for chain_id in candidate_chain_ids:
            # call the low-level parser: filter the PDB text by chain id and atom names
            # returns coords_array: shape (n_residues, n_atoms, 3) coordinate array; seq: amino-acid sequence string
            # if the chain does not exist, returns (None, None)
            # both are None together when the chain is absent, otherwise (coords, sequence)
            coords_array: np.ndarray | None
            seq: str | None
            coords_array, seq = parse_PDB_biounits(biounit, backbone_atom_names=backbone_atom_names, chain_id=chain_id)
            # only write to the dict when coords_array is not None, i.e. the chain actually exists
            if coords_array is not None:
                # append this chain's sequence to the concatenated full sequence
                concat_seq += seq
                # store this chain's sequence under a key like 'seq_chain_A'
                pdb_dict[f'seq_chain_{chain_id}'] = seq
                # initialize the coordinate sub-dict for this chain
                coords_dict_chain: dict[str, list[list[float]]] = {}
                # one key per extracted atom type, e.g. 'CA_chain_A' -> shape (n_res, 3)
                # (with ca_only only 'CA_chain_A' is produced)
                for atom_index, atom_name in enumerate(backbone_atom_names):
                    coords_dict_chain[f'{atom_name}_chain_{chain_id}'] = coords_array[:, atom_index, :].tolist()
                # attach this chain's coordinate sub-dict to the main dict under 'coords_chain_A' etc.
                pdb_dict[f'coords_chain_{chain_id}'] = coords_dict_chain
                # increment the parsed-chain counter
                num_parsed_chains += 1
        # strip the directory prefix using the platform's own path rules: os.path.basename
        # handles both separator styles on Windows and the POSIX separator on Linux
        pdb_file_name = os.path.basename(biounit)
        # remove the ".pdb" extension to get the structure name
        pdb_dict['name'] = os.path.splitext(pdb_file_name)[0]
        # record the total number of chains found in this file
        pdb_dict['num_of_chains'] = num_parsed_chains
        # store the concatenated sequence of all chains
        pdb_dict['seq'] = concat_seq
        # add this file's parsed dict to the return list
        pdb_dict_list.append(pdb_dict)
        # increment the structure counter reported at the end of the function
        num_parsed_structures += 1
    # report how many structures were parsed (num_parsed_structures is otherwise unused)
    print(f'parse_PDB: parsed {num_parsed_structures} structure(s) from {path_to_pdb}')
    # return the list of parsed dicts
    return pdb_dict_list


class TiedFeatures(NamedTuple):
    """Packed batch returned by :func:`tied_featurize`.

    Field order is the historical positional order of that function, so callers may
    either read fields by name (``features.chain_M``) or keep unpacking the bundle
    positionally. Field names follow the public names documented in
    :func:`tied_featurize`; where the function builds a value under a more
    descriptive local name, the mapping is noted on the corresponding field.
    """

    # Padded backbone coordinates, [B, L, 4, 3] for N/CA/C/O or [B, L, 3] in CA-only mode
    X: torch.Tensor
    # Integer-encoded amino-acid sequence, [B, L] (index into PROTEIN_ALPHABET)
    S: torch.Tensor
    # Per-residue validity mask, [B, L] (1.0 when all backbone atoms are present)
    mask: torch.Tensor
    # Total residue count per entry, [B] (numpy int32, intentionally not a tensor)
    sequence_lengths: np.ndarray
    # Chain-level design mask, [B, L]; internal name is chain_design_mask
    chain_M: torch.Tensor
    # Integer chain encoding number per residue, [B, L]
    chain_encoding_all: torch.Tensor
    # Chain IDs of every chain, one inner list per structure
    all_chain_ids_per_entry: list[list[str]]
    # Visible (fixed context) chain IDs, one inner list per structure
    visible_chain_ids_per_entry: list[list[str]]
    # Masked (designed) chain IDs, one inner list per structure
    masked_chain_ids_per_entry: list[list[str]]
    # Residue count of each masked chain, one inner list per structure
    masked_chain_lengths_per_entry: list[list[int]]
    # Per-position design mask, [B, L]; internal name is free_residue_mask
    chain_M_pos: torch.Tensor
    # Omitted amino-acid mask, [B, L, num_amino_acid_types]
    omit_AA_mask: torch.Tensor
    # Per-residue numbering, [B, L] (-100 marks padding)
    residue_idx: torch.Tensor
    # Phi/psi/omega validity mask, [B, L, 3]
    dihedral_mask: torch.Tensor
    # Tied-position groups per entry; innermost list holds the 0-based residues tied together
    tied_position_groups_per_entry: list[list[list[int]]]
    # PSSM mixing coefficient, [B, L]
    pssm_coef_all: torch.Tensor
    # PSSM per-position bias, [B, L, num_amino_acid_types]
    pssm_bias_all: torch.Tensor
    # PSSM log-odds, [B, L, num_amino_acid_types]
    pssm_log_odds_all: torch.Tensor
    # Per-residue amino-acid bias, [B, L, num_amino_acid_types]
    bias_by_res_all: torch.Tensor
    # Tied-position weighting, [B, L] (1.0 = uniform weight)
    tied_position_weight: torch.Tensor


def tied_featurize(
    pdb_dict_list: list[dict[str, Any]],
    device: torch.device,
    chain_id_dict: dict[str, tuple[list[str], list[str]]] | None,
    fixed_position_dict: dict[str, dict[str, list[int]]] | None = None,
    omit_AA_dict: dict[str, dict[str, list[list[list[int] | list[str]]]]] | None = None,
    tied_positions_dict: dict[str, list[dict[str, Any]]] | None = None,
    pssm_dict: dict[str, Any] | None = None,
    bias_by_res_dict: dict[str, dict[str, np.ndarray]] | None = None,
    ca_only: bool = False
) -> TiedFeatures:
    '''
    Pack and pad a batch of parsed PDB entries into torch tensors.

    Args:
        pdb_dict_list: the batch to featurize, i.e. a list of parsed PDB entry
            dicts (output of ``parse_PDB``), each holding ``seq_chain_X`` and
            ``coords_chain_X`` keys for every chain ``X``. E.g. one entry for
            ``1abc`` with chains A, B and C (coordinates abbreviated with ``...``):
                {
                    'name': '1abc',
                    'seq': 'AAAA...BBBB...CCCC...',               # all chains concatenated
                    'seq_chain_A': 'AAAA...',                     # [chain_length]
                    'coords_chain_A': {                           # one key per backbone atom
                        'N_chain_A': [[-13.9, 7.2, 0.7], ...],    # [chain_length, 3]
                        'CA_chain_A': [[-13.3, 7.6, 1.1], ...],   # [chain_length, 3]
                        'C_chain_A': [[-13.6, 6.5, 2.0], ...],    # [chain_length, 3]
                        'O_chain_A': [[-12.6, 5.9, 2.1], ...],    # [chain_length, 3]
                    },
                    'seq_chain_B': 'BBBB...',
                    'coords_chain_B': {...},
                    'seq_chain_C': 'CCCC...',
                    'coords_chain_C': {...},
                }
        device: torch device on which the returned tensors are placed, e.g.
            ``torch.device('cuda:0')``.
        chain_id_dict: optional mapping of structure name to a
            ``(masked_chain_ids, visible_chain_ids)`` pair; masked chains are
            (re)designed, visible chains are kept fixed as context (both lists are
            sorted internally, so their input order does not matter). An omitted dict
            (None) treats every parsed chain as masked, i.e. designs everything. E.g.
            for entry ``1abc``, design chains A and B while keeping C as context:
                {
                    '1abc': (['A', 'B'], ['C']),
                }
        fixed_position_dict: optional mapping of structure name to chain ID to
            the list of 1-based, chain-local residue positions that must keep their
            native amino acid. Positions are counted from 1 along that chain's own
            sequence. Only masked chains are consulted, but every masked chain must be
            listed (an empty list ``[]`` leaves that chain fully free); an omitted
            dict (None) pins nothing. E.g. pin residues 1, 2 and 3 of chain A while
            leaving chain B free:
                {
                    '1abc': {
                        'A': [1, 2, 3],
                        'B': [],
                    },
                }
        omit_AA_dict: optional per-chain amino-acid omission constraints. Maps PDB
            name to chain ID to a list of ``(positions, one letter amino acid codes)`` rules;
            each rule pairs a list of 1-based residue positions with the list
            of amino-acid codes forbidden at those positions. Each pair is a
            2-element list, exactly as decoded from JSON, and every masked chain must
            be listed (an empty list omits nothing); e.g. entry ``1abc``, forbidding
            A and W at residues 1, 3 and 5 and C at residue 10 of chain A:
                {
                    '1abc': {
                        'A': [
                            [[1, 3, 5], ['A', 'W']],  # forbid A and W at residues 1, 3, 5
                            [[10], ['C']],           # forbid C at residue 10
                        ],
                        'B': [],
                    },
                }
        tied_positions_dict: optional tied-position groups forcing identical amino
            acids across chains (homomer design). Maps structure name to a list of
            groups; each group maps a chain ID to its tied 1-based, chain-local
            positions (numbered as in ``fixed_position_dict``, and a group may span two
            or more chains). All positions of one group are decoded as a single step
            and must receive the exact same residue. A chain value is either a plain
            list of positions (weight 1.0 everywhere) or a ``[positions, weights]``
            pair, whose weights are written into the matching slots of
            ``tied_position_weight``. Unlike the dicts above it is keyed by structure
            name only, so chain IDs can be listed per group. E.g. tie A1/A2/A3 to
            B1/B2/B3, and in a second group tie A5/A6 to B5/B6 down-weighting residue 6
            to 0.5:
                {
                    '1abc': [
                        {'A': [1, 2, 3], 'B': [1, 2, 3]},      # plain form, weight 1.0
                        {'A': [[5, 6], [1.0, 0.5]],            # weighted form
                         'B': [[5, 6], [1.0, 0.5]]},
                    ],
                }
        pssm_dict: optional per-chain PSSM data (coef/bias/log-odds). Maps structure
            name to chain ID to a dict holding the keys ``'pssm_coef'``
            (shape ``[chain_length]``), ``'pssm_bias'`` and ``'pssm_log_odds'``
            (shape ``[chain_length, num_amino_acid_types]``, i.e. ``[chain_length, 21]``).
            Only masked chains are consulted, but every masked chain must be listed;
            a chain mapped to a falsy value (e.g. an empty dict) keeps the neutral
            defaults, namely a zero coefficient, a zero bias and log-odds 10000.0
            (unconstrained). E.g. chain A of entry ``1abc`` is 10 residues long and
            stays unconstrained, and chain B is listed as unconstrained too:
                {
                    '1abc': {
                        'A': {
                            'pssm_coef': np.zeros(10),                    # [10]
                            'pssm_bias': np.zeros((10, 21)),              # [10, 21]
                            'pssm_log_odds': 10000.0 * np.ones((10, 21)), # [10, 21]
                        },
                        'B': {},
                    },
                }
        bias_by_res_dict: optional mapping of structure name to chain ID to a
            per-residue amino-acid bias array of shape
            ``[chain_length, num_amino_acid_types]`` (i.e. ``[chain_length, 21]``) that
            is added to the model logits. Only masked chains are consulted, but every
            masked chain must be listed, and an omitted dict (None) leaves every chain
            unbiased. E.g. both masked chains of ``1abc`` (10 residues each) unbiased:
                {
                    '1abc': {
                        'A': np.zeros((10, 21)),  # [10, 21], 0.0 = no bias
                        'B': np.zeros((10, 21)),
                    },
                }
        ca_only: if True, featurize CA atoms only instead of N/CA/C/O.

    Returns:
        TiedFeatures, a ``NamedTuple`` holding the 20 fields listed below. Fields can
        be read by name (e.g. ``features.chain_M``); the field order also matches the
        historical positional order, so tuple-style unpacking still works. Shape
        notation below refers to the batch, which is the input ``pdb_dict_list``
        (a list of parsed PDB entries):

        - ``B``: batch axis, the number of structures in the batch, i.e.
          ``len(pdb_dict_list)``; note that the chain
          ID ``'B'`` used elsewhere in this docstring is a different thing
        - ``L``: length axis, the maximum total number of residues in an entry
          across the batch (summed over all chains of each entry), to which every entry is
          padded; the unpadded length of the entry being
          processed is the local variable ``entry_total_residue_count``
        - ``num_amino_acid_types``: size of the amino-acid alphabet, i.e.
          ``len(PROTEIN_ALPHABET)`` = 21 (20 standard residues + ``X``); it is
          the last-axis size of every per-amino-acid tensor below

        1. ``X`` (torch.Tensor, float32): padded backbone atom coordinates
           [B, L, 4, 3], where 4 = the N/CA/C/O atoms and 3 = the (x, y, z)
           components; in ``ca_only`` mode the atom axis is squeezed, giving
           [B, L, 3] (CA only)
        2. ``S`` (torch.Tensor, int64, local variable ``residue_type_indices``):
           integer-encoded amino-acid sequence [B, L]; "integer" here means the
           0-based index of the residue's one-letter code in ``PROTEIN_ALPHABET``
           (``'ACDEFGHIKLMNPQRSTVWYX'``), e.g.
           ``A`` -> 0 and ``X`` -> 20; padded positions beyond an entry's length
           stay 0. The encoding alphabet contains no gap symbol: raw ``'-'``
           characters in the input sequence are converted to ``X`` before
           encoding
        3. ``mask`` (torch.Tensor, float32, local variable ``valid_residue_mask``):
           per-residue validity mask [B, L].
           Binary 0.0/1.0 only: 1.0 for a residue whose backbone coordinates are
           all present; 0.0 for a padded position or a residue with missing
           coordinates (both were NaN and detected via ``np.isfinite``). The
           model uses it to ignore invalid positions when averaging per-residue
           scores and when attending across residues
        4. ``sequence_lengths`` (numpy.ndarray, int32, not a tensor): total residue
           sequence length of each entry [B], summed over all chains (internal
           chain gaps are counted as positions)
        5. ``chain_M`` (torch.Tensor, float32, local variable
           ``chain_design_mask``): chain-level design mask [B, L].
           Binary 0.0/1.0: 1.0 for every position of a chain selected for
           (re)design, 0.0 for positions of a visible chain kept as fixed
           context and for padding. This is a chain-granularity switch only:
           it does not exclude user-fixed positions (see ``chain_M_pos``).
           Internally the model builds ``chain_M * chain_M_pos * mask``:
           sampling writes residues only where that product is 1.0, and scores
           are averaged over the same positions
        6. ``chain_encoding_all`` (torch.Tensor, int64): integer chain encoding number per
           residue [B, L] (1 for the first chain, 2 for the second, ...)
        7. ``all_chain_ids_per_entry`` (list[list[str]], plain Python list): chain
           IDs of every chain per structure. The outer list has one element
           per PDB structure (length B); the inner list holds that structure's
           chain IDs in processing order: masked chains first, then visible
           chains, with each bucket sorted ascending
           (``masked_chain_ids.sort()`` / ``visible_chain_ids.sort()``), so the
           order is deterministic. It is ragged, so it has no fixed [B, L] shape
        8. ``visible_chain_ids_per_entry`` (list[list[str]], plain Python list): visible
           (fixed context) chain IDs per structure, one inner list per PDB
           structure, sorted ascending
        9. ``masked_chain_ids_per_entry`` (list[list[str]], plain Python list): masked
           (designed) chain IDs per structure, one inner list per PDB
           structure, sorted ascending
        10. ``masked_chain_lengths_per_entry`` (list[list[int]], plain Python
            list): residue count of each masked chain, one inner list per PDB
            structure. The inner lists are aligned element by element with
            ``masked_chain_ids_per_entry``: ``masked_chain_lengths_per_entry[i][k]``
            is the residue count of the chain named by
            ``masked_chain_ids_per_entry[i][k]``, because both are built by
            iterating the same sorted ``masked_chain_ids``; e.g. for entry 0
                print(features.masked_chain_ids_per_entry[0])      # ['A', 'C']
                print(features.masked_chain_lengths_per_entry[0])  # [10, 7]
            means chain A has 10 residues and chain C has 7
        11. ``chain_M_pos`` (torch.Tensor, float32, local variable
            ``free_residue_mask``): per-position free-to-design mask
            [B, L]. Binary 0.0/1.0: 0.0 for a position the user fixed via
            ``fixed_position_dict`` (and for padding), 1.0 for every other
            position, including all positions of visible chains. On its own it
            does not say which chains are designed, so it is always used
            multiplied by ``chain_M`` (see item 5)
        12. ``omit_AA_mask`` (torch.Tensor, float32): omitted amino-acid mask
            [B, L, num_amino_acid_types]; 1.0 marks an amino acid type that may not be placed at
            that position. Sampling multiplies the predicted probabilities by
            ``1 - omit_AA_mask`` and renormalizes before drawing, so omitted
            types get exactly 0 probability and are never sampled
        13. ``residue_idx`` (torch.Tensor, int64): per-residue numbering [B, L].
            Each chain gets a disjoint band of width 100: for the chain with
            1-based id ``chain_encoding_number`` (the same value stored in
            ``chain_encoding_all`` -- not the PDB chain ID) the residue at
            0-based position ``k`` in the entry's concatenated chain layout is
            numbered ``100 * (chain_encoding_number - 1) + k``; -100 marks padding. The
            100-wide bands keep numbering of different chains non-consecutive,
            so a difference of exactly 1 means two residues are adjacent within
            the same chain
        14. ``dihedral_mask`` (torch.Tensor, float32): phi/psi/omega validity
            mask [B, L, 3]. Column 0 is phi, column 1 psi, column 2 omega; 1.0
            means the neighbouring residue needed by that dihedral exists in the
            same chain, 0.0 means it does not (chain boundary or padding). For an
            entry whose first two chains have 3 and 2 residues:
                print(features.dihedral_mask[0, :5].tolist())
                [[0.0, 1.0, 1.0],   # chain 1 residue 0: phi has no predecessor
                 [1.0, 1.0, 1.0],
                 [1.0, 0.0, 0.0],   # chain 1 residue 2: psi and omega have no successor
                 [0.0, 1.0, 1.0],   # chain 2 residue 0: phi does not cross the chain boundary
                 [1.0, 0.0, 0.0]]   # chain 2 residue 1: last residue of that chain
            It is derived from ``residue_idx`` (item 13): the internal array
            ``jumps`` is 1.0 where ``residue_idx`` increases by exactly 1, which
            the 100-wide chain bands make equivalent to "same chain and adjacent
            residue"; a chain boundary jumps by 101 and padding is -100, so both
            give 0.0. The mask only tracks adjacency. 
        15. ``tied_position_groups_per_entry`` (list[list[list[int]]], plain
            Python list): tied-position groups per entry; the innermost list
            holds the absolute 0-based global residue indices tied together, i.e. the
            global index in the entry's concatenated chain layout (chain-local
            1-based positions are converted by adding the chain's start offset
            and subtracting 1). For a homodimer whose chains A and B both have
            10 residues, with every position tied:
                print(features.tied_position_groups_per_entry)
                [
                    [[0, 10], [1, 11], [2, 12], ...],  # entry 0: 10 groups, A1+B1 ... A10+B10
                ]
            An entry without tied positions prints as [], so the outer list has
            shape [B]
        16. ``pssm_coef_all`` (torch.Tensor, float32): PSSM mixing coefficient
            [B, L]
        17. ``pssm_bias_all`` (torch.Tensor, float32): PSSM per-position bias
            [B, L, num_amino_acid_types]
        18. ``pssm_log_odds_all`` (torch.Tensor, float32): PSSM log-odds
            [B, L, num_amino_acid_types]
        19. ``bias_by_res_all`` (torch.Tensor, float32): per-residue amino-acid
            bias [B, L, num_amino_acid_types]
        20. ``tied_position_weight`` (torch.Tensor, float32): tied-position weighting
            [B, L] (1.0 = uniform weight)
    '''
    # Shape notation used below:
    #   B = number of PDB structures packed into one batch (batch axis); this is
    #       unrelated to the chain ID 'B' mentioned in the examples above
    #   L = maximum total sequence length across the batch (padded residue axis)
    #   C = chain_length (residue count of a single chain)
    #   num_amino_acid_types = len(alphabet) = 21 (amino-acid axis)

    # Amino-acid alphabet used to convert one-letter codes into integer labels
    alphabet: str = PROTEIN_ALPHABET
    # Number of amino-acid classes: 20 standard residues plus the unknown code X.
    num_amino_acid_types: int = len(alphabet)
    # Number of PDB structures packed into the output tensors (batch axis)
    B: int = len(pdb_dict_list)
    # Total sequence length (sum over chains) of each PDB structure, shape [B]
    sequence_lengths: np.ndarray = np.array([len(pdb_dict['seq']) for pdb_dict in pdb_dict_list], dtype=np.int32)
    # Longest sequence across the packed structures; all tensors are padded to it
    L: int = max([len(pdb_dict['seq']) for pdb_dict in pdb_dict_list])
    if ca_only:
        # CA-only layout [B, L, 1, 3]
        X: np.ndarray = np.zeros([B, L, 1, 3])
    else:
        # Full backbone layout [B, L, 4, 3] holding N, CA, C, O
        X = np.zeros([B, L, 4, 3])
    # Residue numbering per position [B, L]; -100 marks padding and chain breaks
    residue_idx: np.ndarray = -100 * np.ones([B, L], dtype=np.int32)
    # Chain-level design switch, shape [B, L].
    # All residues of one chain share a single value: 
    # 1.0 when the chain is selected for design (a "masked" chain), 
    # 0.0 when the chain is kept as fixed context(a "visible" chain). 
    # It answers only "which chains", never "which residues".
    chain_design_mask: np.ndarray = np.zeros([B, L], dtype=np.int32)
    # PSSM mixing coefficient per position [B, L]
    pssm_coef_all: np.ndarray = np.zeros([B, L], dtype=np.float32)
    # PSSM per-position amino-acid bias [B, L, num_amino_acid_types]
    pssm_bias_all: np.ndarray = np.zeros([B, L, num_amino_acid_types], dtype=np.float32)
    # PSSM log-odds [B, L, num_amino_acid_types];
    # a large positive default leaves positions unconstrained
    pssm_log_odds_all: np.ndarray = 10000.0 * np.ones([B, L, num_amino_acid_types], dtype=np.float32)
    # Residue-level free mask, shape [B, L].
    # Unlike chain_design_mask (one value per chain), this varies at residue granularity:
    # 1.0 for a position the user leaves free to be redesigned, 
    # 0.0 for a position the user pins via the fixed-residue specification.
    # Residues of visible chains are 1.0 here, but they are never sampled because
    # chain_design_mask is 0.0 along those chains. The model samples only where
    # chain_design_mask * free_residue_mask * valid_residue_mask == 1.
    free_residue_mask: np.ndarray = np.zeros([B, L], dtype=np.int32)
    # Per-position amino-acid bias added to the model logits [B, L, num_amino_acid_types]
    bias_by_res_all: np.ndarray = np.zeros([B, L, num_amino_acid_types], dtype=np.float32)
    # Integer chain encoding number per position [B, L] (1 for the first chain, 2 for the second, ...)
    chain_encoding_all: np.ndarray = np.zeros([B, L], dtype=np.int32)
    # Amino-acid sequence encoded as integer residue-type indices, shape [B, L].
    # "Integer" means a 0-based index
    # into the alphabet PROTEIN_ALPHABET ('ACDEFGHIKLMNPQRSTVWYX'): each element is
    # the alphabet position of that residue's one-letter code, e.g. A -> 0, C -> 1
    # and X -> 20 (raw '-' gap characters were converted to X earlier). Slots
    # beyond an entry's real length stay 0 and are ignored via ``mask``.
    residue_type_indices: np.ndarray = np.zeros([B, L], dtype=np.int32)
    # Per-position mask of amino-acid types to omit during sampling [B, L, num_amino_acid_types]
    omit_AA_mask: np.ndarray = np.zeros([B, L, num_amino_acid_types], dtype=np.int32)
    
    # ------------------------------------------------------------------
    # Per-entry chain ID bookkeeping. The four lists below are ragged Python
    # lists of length B (one element per PDB structure), filled inside the
    # entry loop, so they are not padded and have no fixed inner width.
    # ------------------------------------------------------------------
    # Chain IDs of every chain of each structure, in processing order: masked
    # (designed) chains first, then visible (fixed context) chains, with each
    # bucket sorted ascending, i.e. it is always the exact concatenation
    #   all_chain_ids_per_entry[i] = masked_chain_ids_per_entry[i] + visible_chain_ids_per_entry[i]
    # (same as the local ``all_chain_ids`` used inside the entry loop). This is
    # the order used to concatenate chains along the residue axis, e.g. for 1X2I
    # with both chains designed
    #     print(all_chain_ids_per_entry)  # [['A', 'B']]
    all_chain_ids_per_entry: list[list[str]] = []
    # Visible (fixed context) chain IDs of each structure, sorted ascending; an
    # empty inner list means the structure has no fixed chain, e.g. for 1X2I
    # used as a homodimer with nothing fixed
    #     print(visible_chain_ids_per_entry)  # [[]]
    # and for a design of chain A only, with chain B kept as context
    #     print(visible_chain_ids_per_entry)  # [['B']]
    visible_chain_ids_per_entry: list[list[str]] = []
    # Masked (designed) chain IDs of each structure, sorted ascending, e.g. for
    # 1X2I with both chains designed
    #     print(masked_chain_ids_per_entry)  # [['A', 'B']]
    masked_chain_ids_per_entry: list[list[str]] = []
    # Residue count of each masked chain. The inner
    # lists are aligned element by element with masked_chain_ids_per_entry:
    # masked_chain_lengths_per_entry[i][k] is the residue count of the chain
    # named by masked_chain_ids_per_entry[i][k], e.g. for 1X2I (68 residues per
    # chain, both chains designed)
    #     print(masked_chain_lengths_per_entry)  # [[68, 68]]
    masked_chain_lengths_per_entry: list[list[int]] = []

    # Per-entry tied-position groups; ragged, length B, inner lists holding the
    # absolute 0-based global residue indices of one tied group.
    # Example, for a batch of a homodimer (chains A and B, 10 residues each) and
    # a homotrimer (chains C, D and E, 20 residues each):
    #   print(features.tied_position_groups_per_entry)
    #   [
    #       [[0, 1, 2, 10, 11, 12], [4, 14], ...],  # entry 0 (homodimer A/B)
    #       [[0, 20, 40], ...]                      # entry 1 (homotrimer C/D/E)
    #   ]
    # The first inner list belongs to entry 0 and ties A1/A2/A3 to B1/B2/B3,
    # then A5 to B5; any further groups of that entry would follow the "...".
    # The second inner list belongs to entry 1 and ties C1, D1 and E1. Indices
    # are global 0-based positions in the concatenated chain layout, so the
    # second chain of a 10-residue dimer starts at 10. An entry without tied
    # positions prints as [].
    tied_position_groups_per_entry: list[list[list[int]]] = []
    # Per-position tied weighting, shape [B, L]
    # (1.0 = uniform weight). Kept batch-shaped so heterogeneous entries can carry
    # their own tied weights
    tied_position_weight: np.ndarray = np.ones([B, L], dtype=np.float32)
    # Featurize each entry
    for entry_index, pdb_dict in enumerate(pdb_dict_list):
        # Resolve which chains are masked (designed) vs visible (fixed context)
        # for this entry; resolved per entry so a heterogeneous batch stays correct
        if chain_id_dict is not None:
            # User-supplied (masked_chain_ids, visible_chain_ids) pair for
            # this entry. Example, for a batch of two entries:
            #   print(chain_id_dict)
            #   {'1abc': (['A', 'B'], ['C']), '2def': (['D'], ['E', 'F']), ...}
            # For '1abc' chains A and B are designed while C is fixed context; 
            # for '2def' chain D is designed while E and F are fixed context. 
            # Chains listed in neither bucket are dropped from that entry entirely.
            # Copy both lists so sorting them below never reorders the caller-owned
            # chain_id_dict, and so returned metadata does not alias it
            masked_chain_ids: list[str] = list(chain_id_dict[pdb_dict['name']][0])
            visible_chain_ids: list[str] = list(chain_id_dict[pdb_dict['name']][1])
        else:
            # Otherwise treat every parsed chain as masked (design everything).
            # Iterating the dict yields its keys, e.g. 'seq_chain_A'; the text after
            # the prefix is the chain ID itself
            masked_chain_ids = [key.removeprefix('seq_chain_') for key in pdb_dict if key.startswith('seq_chain_')]
            visible_chain_ids = []
        # Sort chain IDs for deterministic ordering
        masked_chain_ids.sort()
        visible_chain_ids.sort()
        # Processing order for this entry: masked chains first, then visible
        # chains. For every PDB structure, all chain-wise concatenations along
        # the residue axis follow exactly this order. The per-entry metadata stored 
        # in the returned TiedFeatures uses the same order
        all_chain_ids: list[str] = masked_chain_ids + visible_chain_ids
        # Per-chain backbone coordinates for this entry; one array of shape
        # [C, 4, 3] per chain ([C, 1, 3] in ca_only mode)
        chain_backbone_coordinates_list: list[np.ndarray] = []
        # Per-chain chain-level design mask; one 1-D array of shape [C] per chain,
        # appended in the order of all_chain_ids (masked chains first, visible
        # chains after). Every element of one array holds the same value because
        # the decision is made per chain, not per residue: 1.0 across a masked
        # (designed) chain, 0.0 across a visible (fixed-context) chain.
        # Example, for an entry built from a 10-residue masked chain A followed by
        # a 6-residue visible chain B:
        #   print([chain_mask.tolist() for chain_mask in chain_mask_list])
        #   [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # chain A: designed
        #    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]                     # chain B: fixed context
        # The arrays are concatenated into all_chain_mask [entry_total_residue_count], then padded to
        # [B, L] and returned as ``chain_M``; padding stays 0.0. Positions missing
        # backbone atoms are not zeroed here, the model applies ``chain_M * mask``.
        chain_mask_list: list[np.ndarray] = []
        # Per-chain one-letter sequences for this entry; one string of length C per
        # chain, in the same order as all_chain_ids. Gaps '-' are already
        # replaced by the unknown-residue code 'X', so every character is a member
        # of PROTEIN_ALPHABET.
        # Example, for the 10-residue chain A and 6-residue chain B used above:
        #   print(chain_seq_list)
        #   ['MKTAYIAKQR', 'GSSHHH']   # chain A, chain B
        chain_seq_list: list[str] = []
        # Per-chain integer chain-encoding arrays; one array of shape [C] per chain,
        # constant within a chain and equal to chain_encoding_number (1-based, in
        # packing order, unrelated to the PDB chain ID).
        # Example, for the same two chains:
        #   print([chain_encoding.tolist() for chain_encoding in chain_encoding_list])
        #   [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # chain A -> chain_encoding_number 1
        #    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]                     # chain B -> chain_encoding_number 2
        chain_encoding_list: list[np.ndarray] = []
        # 1-based running chain encoding number (not the PDB chain ID),
        # incremented after each chain is packed
        chain_encoding_number: int = 1
        # Start offset of each chain in the concatenated layout, indexed by the
        # chain's position in all_chain_ids; the first chain starts at 0. Length is
        # num_chains + 1, so the last element also marks the exclusive end of the
        # final concatenated chain (equal to the entry's total residue count)
        global_chain_start_offsets: list[int] = [0]
        # Length of each masked chain; Python ints, one per masked chain
        masked_chain_lengths: list[int] = []
        # Per-residue free-to-design masks, one array of shape [C] per chain:
        # 1.0 at positions the user leaves free to design, 0.0 at positions the
        # user pinned via fixed_position_dict (all 1.0 when that structure has
        # no fixed-position entry). This is residue-level and applies to every
        # chain; whether a whole chain is designed is decided separately by the
        # chain-level chain_design_mask
        free_residue_mask_list: list[np.ndarray] = []
        # Per-chain omitted-amino-acid masks; one array of shape
        # [C, num_amino_acid_types] per chain
        omit_AA_mask_list: list[np.ndarray] = []
        # Per-chain PSSM coefficients; one array of shape [C] per chain
        pssm_coef_list: list[np.ndarray] = []
        # Per-chain PSSM biases; one array of shape [C, num_amino_acid_types] per chain
        pssm_bias_list: list[np.ndarray] = []
        # Per-chain PSSM log-odds; one array of shape [C, num_amino_acid_types] per chain
        pssm_log_odds_list: list[np.ndarray] = []
        # Per-chain per-residue bias arrays; one array of shape
        # [C, num_amino_acid_types] per chain
        bias_by_res_list: list[np.ndarray] = []
        # Start offset of the current chain in the concatenated layout (inclusive)
        current_chain_start: int = 0
        # End offset of the current chain in the concatenated layout (exclusive)
        current_chain_end: int = 0
        # Iterate over chains in processing order (masked first, then visible)
        for chain_id in all_chain_ids:
            # Visible chains provide fixed structural context
            if chain_id in visible_chain_ids:
                # Native sequence of this chain
                chain_seq: str = pdb_dict[f'seq_chain_{chain_id}']
                # Replace gap markers with the unknown-residue code 'X'
                chain_seq = chain_seq.replace('-', 'X')
                # Number of residues in this chain
                chain_length: int = len(chain_seq)
                # Record where the next chain starts in the concatenated layout
                global_chain_start_offsets.append(global_chain_start_offsets[-1] + chain_length)
                # Per-chain backbone atom coordinates, keyed as '{atom}_chain_{chain_id}'
                # with atom one of N, CA, C, O. Each value is a plain Python list of
                # shape [chain_length, 3] holding one [x, y, z] row per residue, e.g.
                # chain_coords['CA_chain_A'][2] is residue 3's CA atom coordinate in chain A
                # Sample for a 2-residue chain A:
                # {
                #     'N_chain_A':  [[-13.1, 10.4, 3.2], [-11.9, 8.7, 1.5]],
                #     'CA_chain_A': [[-13.3, 7.6, 1.1], [-11.7, 6.9, -1.2]],
                #     'C_chain_A':  [[-12.1, 6.7, -0.7], [-11.6, 5.8, -2.3]],
                #     'O_chain_A':  [[-11.1, 7.0, -0.8], [-12.2, 5.3, -3.0]],
                # }
                chain_coords: dict[str, Any] = pdb_dict[f'coords_chain_{chain_id}']
                # Visible chains are not predicted: design mask is all zeros, shape [C]
                chain_mask: np.ndarray = np.zeros(chain_length) #0.0 for visible chains
                if ca_only:
                    # Keep only CA coordinates, initial shape [C, 3], final shape [C, 1, 3]
                    chain_backbone_coordinates: np.ndarray = np.array(chain_coords[f'CA_chain_{chain_id}'])
                    # Add the missing atom axis when the array is 2-D
                    if len(chain_backbone_coordinates.shape) == 2:
                        chain_backbone_coordinates = chain_backbone_coordinates[:, None, :]
                else:
                    # Stack N, CA, C, O coordinates into [C, 4, 3]
                    chain_backbone_coordinates: np.ndarray = np.stack([np.array(chain_coords[f'{atom_name}_chain_{chain_id}']) for atom_name in FULL_BACKBONE_ATOM_NAMES], axis=1) 
                # Collect the per-chain arrays
                chain_backbone_coordinates_list.append(chain_backbone_coordinates)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                # Tag every residue of this chain with the current chain encoding number
                chain_encoding_list.append(chain_encoding_number * np.ones(chain_length))
                # Advance the end offset
                current_chain_end += chain_length
                # Residue numbering restarts per chain with a 100-step offset between chains
                residue_idx[entry_index, current_chain_start:current_chain_end] = 100 * (chain_encoding_number - 1) + np.arange(current_chain_start, current_chain_end)
                # Advance the start offset to the end of this chain
                current_chain_start = current_chain_end
                # Move on to the next chain encoding number
                chain_encoding_number += 1
                # No residue of a visible chain is user-fixed, so every position stays 1.0 here.
                # This mask only records user-fixed positions (1.0 = free, 0.0 = pinned) and
                # says nothing about chain-level designability: visible chains are kept out of
                # sampling by chain_design_mask = 0.0, not by this mask.
                chain_free_residue_mask: np.ndarray = np.ones(chain_length)  # [C]
                free_residue_mask_list.append(chain_free_residue_mask)
                # Visible chains have no omitted amino acids: mask stays all zeros,
                # shape [C, num_amino_acid_types]
                omit_AA_mask_temp: np.ndarray = np.zeros([chain_length, num_amino_acid_types], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                # Neutral PSSM defaults for visible chains
                pssm_coef: np.ndarray = np.zeros(chain_length)  # [C]
                pssm_bias: np.ndarray = np.zeros([chain_length, num_amino_acid_types])  # [C, num_amino_acid_types]
                pssm_log_odds: np.ndarray = 10000.0 * np.ones([chain_length, num_amino_acid_types])  # [C, num_amino_acid_types]
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                # No per-residue bias for visible chains: [C, num_amino_acid_types] zeros
                bias_by_res_list.append(np.zeros([chain_length, num_amino_acid_types]))
            # Masked chains are the design targets
            if chain_id in masked_chain_ids:
                # Native sequence of this chain
                chain_seq = pdb_dict[f'seq_chain_{chain_id}']
                # Replace gap markers with the unknown-residue code X
                chain_seq = chain_seq.replace('-', 'X')
                # Number of residues in this chain
                chain_length = len(chain_seq)
                # Record where the next chain starts in the concatenated layout
                global_chain_start_offsets.append(global_chain_start_offsets[-1] + chain_length)
                # Track masked-chain lengths for later chain splitting
                masked_chain_lengths.append(chain_length)
                # Same per-chain coordinate dict as in the visible branch above: keyed as
                # '{atom}_chain_{chain_id}' for atom in {N, CA, C, O}, each value a plain
                # Python list of shape [chain_length, 3] with one [x, y, z] per residue
                chain_coords: dict[str, Any] = pdb_dict[f'coords_chain_{chain_id}']
                # Masked chains are predicted: design mask is all ones, shape [C]
                chain_mask = np.ones(chain_length) #1.0 for masked
                if ca_only:
                    # Keep only CA coordinates, shape [C, 1, 3]
                    chain_backbone_coordinates = np.array(chain_coords[f'CA_chain_{chain_id}'])
                    # Add the missing atom axis when the array is 2-D
                    if len(chain_backbone_coordinates.shape) == 2:
                        chain_backbone_coordinates = chain_backbone_coordinates[:, None, :]
                else:
                    # Stack N, CA, C, O coordinates into [C, 4, 3]
                    chain_backbone_coordinates = np.stack([np.array(chain_coords[f'{atom_name}_chain_{chain_id}']) for atom_name in FULL_BACKBONE_ATOM_NAMES], axis=1) 
                # Collect the per-chain arrays
                chain_backbone_coordinates_list.append(chain_backbone_coordinates)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                # Tag every residue of this chain with the current chain encoding number
                chain_encoding_list.append(chain_encoding_number * np.ones(chain_length))
                # Advance the end offset
                current_chain_end += chain_length
                # Residue numbering restarts per chain with a 100-step offset between chains
                residue_idx[entry_index, current_chain_start:current_chain_end] = 100 * (chain_encoding_number - 1) + np.arange(current_chain_start, current_chain_end)
                # Advance the start offset to the end of this chain
                current_chain_start = current_chain_end
                # Move on to the next chain encoding number
                chain_encoding_number += 1
                # Start with every position free to be designed, shape [C]
                chain_free_residue_mask = np.ones(chain_length)
                # Apply user-specified fixed positions when provided
                # fixed_position_dict maps structure name -> chain ID -> 1-based,
                # chain-local residue positions that keep their native amino acid.
                # Only masked chains are looked up, and every masked chain must have
                # an entry ([] leaves that chain fully free). Sample for entry '1abc'
                # with masked chains A and B, pinning residues 1-3 of A only:
                # {
                #     '1abc': {
                #         'A': [1, 2, 3],
                #         'B': [],
                #     },
                # }
                if fixed_position_dict is not None:
                    # 1-based positions to keep fixed on this chain; Python list
                    fixed_pos_list: list[int] = fixed_position_dict[pdb_dict['name']][chain_id]
                    if fixed_pos_list:
                        # Mark fixed positions as 0.0 (converting to 0-based indices)
                        chain_free_residue_mask[np.array(fixed_pos_list)-1] = 0.0
                free_residue_mask_list.append(chain_free_residue_mask)
                # Start with no omitted amino acids, shape [C, num_amino_acid_types]
                omit_AA_mask_temp = np.zeros([chain_length, num_amino_acid_types], np.int32)
                # Apply user-specified amino-acid omissions when provided
                if omit_AA_dict is not None:
                    # Each item pairs 1-based positions with the letters to omit there.
                    # Sample for entry '1abc', chain 'A'; each pair is a 2-element array,
                    # exactly as decoded from JSON:
                    # {
                    #     '1abc': {
                    #         'A': [
                    #             [[1, 3, 5], ['A', 'W']],  # forbid A and W at residues 1, 3, 5
                    #             [[10], ['C']],           # forbid C at residue 10
                    #         ],
                    #     },
                    # }
                    for item in omit_AA_dict[pdb_dict['name']][chain_id]:
                        # for example, item is [[1, 3, 5], ['A', 'W']]
                        # 0-based residue positions, Python list
                        idx_AA: list[int] = [position - 1 for position in item[0]]
                        # alphabet indices of the omitted amino acids, Python list
                        AA_idx: list[int] = [alphabet.index(AA) for AA in item[1]]
                        # Mark every (residue index, amino acid type) combination as forbidden.
                        # np.ix_ converts the two 1-D index lists into an outer
                        # indexing pair, so the assignment hits all
                        # len(idx_AA) x len(AA_idx) cells at once instead of just
                        # the diagonal: with idx_AA = [0, 2, 4] and AA_idx = [0, 20]
                        # (A and W) it sets rows 0, 2 and 4 of columns 0 and 20 to 1.
                        # A 1 means "this amino acid type may not be sampled as this residue"; 
                        # every untouched cell keeps 0 (allowed).
                        omit_AA_mask_temp[np.ix_(idx_AA, AA_idx)] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                # Neutral PSSM defaults for this chain
                pssm_coef = np.zeros(chain_length)  # [C]
                pssm_bias = np.zeros([chain_length, num_amino_acid_types])  # [C, num_amino_acid_types]
                pssm_log_odds = 10000.0 * np.ones([chain_length, num_amino_acid_types])  # [C, num_amino_acid_types]
                # Override with the user-supplied PSSM when available
                # pssm_dict maps structure name -> chain ID -> dict holding
                # 'pssm_coef' [C], 'pssm_bias' [C, num_amino_acid_types] and
                # 'pssm_log_odds' [C, num_amino_acid_types]. Only masked chains are
                # consulted, but every masked chain must be listed; a chain mapped to
                # a falsy value ({}) keeps the neutral defaults set above, i.e. a zero
                # coefficient, a zero bias and log-odds 10000.0 (unconstrained).
                # Sample for entry '1abc' with 10-residue masked chains A and B,
                # both left unconstrained:
                # {
                #     '1abc': {
                #         'A': {
                #             'pssm_coef': np.zeros(10),  # [10]
                #             'pssm_bias': np.zeros((10, num_amino_acid_types)),  # [10, 21]
                #             'pssm_log_odds': 10000.0 * np.ones((10, num_amino_acid_types)),  # [10, 21]
                #         },
                #         'B': {},
                #     },
                # }
                if pssm_dict:
                    if pssm_dict[pdb_dict['name']][chain_id]:
                        pssm_coef = pssm_dict[pdb_dict['name']][chain_id]['pssm_coef']
                        pssm_bias = pssm_dict[pdb_dict['name']][chain_id]['pssm_bias']
                        pssm_log_odds = pssm_dict[pdb_dict['name']][chain_id]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                # Per-residue amino-acid bias added to the model logits
                # bias_by_res_dict maps structure name -> chain ID -> array of
                # shape [C, num_amino_acid_types] (i.e. [C, 21]); 0.0 in a cell means
                # "no bias for this residue/amino-acid pair". Only masked chains are
                # consulted, and every masked chain must be listed: the lookup is
                # direct, so a missing chain raises KeyError instead of falling back
                # to zeros. An omitted dict (None or any empty mapping) gives every
                # chain an all-zero bias. Sample for entry '1abc' with 10-residue
                # masked chains A and B, both unbiased:
                # {
                #     '1abc': {
                #         'A': np.zeros((10, num_amino_acid_types)),  # [10, 21]
                #         'B': np.zeros((10, num_amino_acid_types)),
                #     },
                # }
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[pdb_dict['name']][chain_id])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, num_amino_acid_types]))

        # Tied-position groups for this entry: one inner list per tied group, each
        # holding the absolute 0-based global residue indices tied together. Example: chains
        # A and B (each length 10, layout A then B) with
        # tied_positions_dict['1abc'] = [{'A': [1, 2, 3], 'B': [1, 2, 3]},
        #                                {'A': [5], 'B': [5]}]
        # gives [[0, 1, 2, 10, 11, 12], [4, 14]]; positions in the same inner
        # list are decoded as one step and share the exact same sampled residue
        # Ragged Python list: one inner list of absolute 0-based global indices per tied group
        tied_position_groups: list[list[int]] = []
        # Convert tied positions from per-chain 1-based to global 0-based indices
        if tied_positions_dict is not None:
            # Tied-position groups for this structure; Python list of group dicts
            tied_pos_list: list[dict[str, Any]] = tied_positions_dict[pdb_dict['name']]
            if tied_pos_list:
                # tied_pos_list for entry '1abc': a list of tied groups, each
                # mapping a chain ID to 1-based positions, plain (Step 1)
                # or paired with per-position weights (Step 2):
                # [
                #     {'A': [1, 2, 3], 'B': [1, 2, 3]},   # Step 1
                #     {'A': [[5, 6], [1.0, 0.5]],         # Step 2
                #      'B': [[5, 6], [1.0, 0.5]]},
                # ]
                # All positions in one group (e.g. A 1/2/3 and B 1/2/3 above) are
                # decoded as a single step and must get the exact same residue

                # Process each tied group independently
                for tied_item in tied_pos_list:
                    # Absolute 0-based indices of the tied positions in this group;
                    # Python list, order follows the group's chain iteration
                    tied_position_group: list[int] = []
                    # Each entry maps a chain ID to its tied residue positions
                    for chain_id, v in tied_item.items():
                        # Start offset of this chain in the concatenated layout; scalar int
                        start_offset: int = global_chain_start_offsets[all_chain_ids.index(chain_id)]
                        # v holds [positions, weights] when per-position weights are given,
                        # i.e. two parallel Python lists of equal length
                        if isinstance(v[0], list):
                            # v: [[5, 6], [1.0, 0.5]]
                            # Convert each weighted position and record its weight
                            for position, weight in zip(v[0], v[1]):
                                # Convert 1-based chain-local index to a 0-based global index
                                tied_position_group.append(start_offset + position - 1)
                                # Store the tied weight for this position
                                tied_position_weight[entry_index, start_offset + position - 1] = weight
                        else:
                            # v is a plain list of 1-based positions
                            # v: [1, 2, 3]
                            for position in v:
                                # Convert 1-based chain-local index to a 0-based global index
                                tied_position_group.append(start_offset + position - 1)
                    tied_position_groups.append(tied_position_group)
        tied_position_groups_per_entry.append(tied_position_groups)

        # Coordinates of all chains of this entry concatenated along the residue
        # axis: shape [entry_total_residue_count, 4, 3] ([entry_total_residue_count, 1, 3]
        # in ca_only mode), where entry_total_residue_count is this entry's unpadded
        # residue count summed over all its chains
        all_chain_backbone_coordinates: np.ndarray = np.concatenate(chain_backbone_coordinates_list, 0) #[entry_total_residue_count, 4, 3]
        # Concatenated sequence of all chains in processing order; string of length entry_total_residue_count
        all_chain_sequence: str = "".join(chain_seq_list)
        # Concatenated chain-level design mask [entry_total_residue_count]; 1.0 for positions that need to be predicted
        all_chain_mask: np.ndarray = np.concatenate(chain_mask_list, 0)
        # Concatenated integer chain encoding number [entry_total_residue_count]
        all_chain_encoding: np.ndarray = np.concatenate(chain_encoding_list, 0)
        # Concatenated free-residue mask [entry_total_residue_count]; 1.0 for free (designable) positions, 0.0 for user-fixed positions
        all_chain_free_residue_mask: np.ndarray = np.concatenate(free_residue_mask_list, 0)

        # Concatenated PSSM coefficient [entry_total_residue_count]
        pssm_coef_: np.ndarray = np.concatenate(pssm_coef_list, 0) 
        # Concatenated PSSM bias [entry_total_residue_count, num_amino_acid_types]
        pssm_bias_: np.ndarray = np.concatenate(pssm_bias_list, 0)
        # Concatenated PSSM log-odds [entry_total_residue_count, num_amino_acid_types]
        pssm_log_odds_: np.ndarray = np.concatenate(pssm_log_odds_list, 0) 

        # Concatenated per-residue bias [entry_total_residue_count, num_amino_acid_types]; 0.0 where no tweaking is needed
        bias_by_res_: np.ndarray = np.concatenate(bias_by_res_list, 0)

        # Total (unpadded) length of this entry
        entry_total_residue_count: int = len(all_chain_sequence)
        # Pad coordinates up to L with NaN to mark missing padding
        # rows; from [entry_total_residue_count, 4, 3] ([entry_total_residue_count, 1, 3]) to [L, 4, 3] ([L, 1, 3])
        padded_backbone_coordinates: np.ndarray = np.pad(all_chain_backbone_coordinates, [[0, L - entry_total_residue_count], [0, 0], [0, 0]], 'constant', constant_values=(np.nan, ))
        # Store the padded coordinates in the coordinate tensor
        X[entry_index, :, :, :] = padded_backbone_coordinates

        # Pad the chain-level design switch with zeros (1.0 for designed chains): [entry_total_residue_count] -> [L]
        padded_all_chain_mask: np.ndarray = np.pad(all_chain_mask, [[0, L - entry_total_residue_count]], 'constant', constant_values=(0.0, ))
        # Pad the free-residue mask with zeros (1.0 for free positions): [entry_total_residue_count] -> [L]
        padded_all_chain_free_residue_mask: np.ndarray = np.pad(all_chain_free_residue_mask, [[0, L - entry_total_residue_count]], 'constant', constant_values=(0.0, ))
        # Pad the omitted-amino-acid mask with zeros along the residue axis only,
        # i.e. [entry_total_residue_count, num_amino_acid_types] -> [L, num_amino_acid_types];
        padded_omit_AA_mask: np.ndarray = np.pad(np.concatenate(omit_AA_mask_list, 0), [[0, L - entry_total_residue_count], [0, 0]], 'constant', constant_values=(0.0, ))
        # Store the padded masks in the pdb_dict_list arrays
        chain_design_mask[entry_index, :] = padded_all_chain_mask
        free_residue_mask[entry_index, :] = padded_all_chain_free_residue_mask
        omit_AA_mask[entry_index, :, :] = padded_omit_AA_mask

        # Pad the chain-encoding array with zeros: [entry_total_residue_count] -> [L]
        padded_all_chain_encoding: np.ndarray = np.pad(all_chain_encoding, [[0, L - entry_total_residue_count]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[entry_index, :] = padded_all_chain_encoding

        # Pad the PSSM arrays with zeros. Note the padding value 0.0 is not the
        # neutral default (10000.0 for log-odds): padding rows rely on mask = 0
        # downstream, so their PSSM values are never read
        pssm_coef_pad: np.ndarray = np.pad(pssm_coef_, [[0, L - entry_total_residue_count]], 'constant', constant_values=(0.0, ))  # [L]
        pssm_bias_pad: np.ndarray = np.pad(pssm_bias_, [[0, L - entry_total_residue_count], [0, 0]], 'constant', constant_values=(0.0, ))  # [L, num_amino_acid_types]
        pssm_log_odds_pad: np.ndarray = np.pad(pssm_log_odds_, [[0, L - entry_total_residue_count], [0, 0]], 'constant', constant_values=(0.0, ))  # [L, num_amino_acid_types]

        # Store the padded PSSM arrays in the pdb_dict_list arrays
        pssm_coef_all[entry_index, :] = pssm_coef_pad
        pssm_bias_all[entry_index, :, :] = pssm_bias_pad
        pssm_log_odds_all[entry_index, :, :] = pssm_log_odds_pad

        # Pad the per-residue bias with zeros: [entry_total_residue_count, num_amino_acid_types] -> [L, num_amino_acid_types]
        bias_by_res_pad: np.ndarray = np.pad(bias_by_res_, [[0, L - entry_total_residue_count], [0, 0]], 'constant', constant_values=(0.0, ))
        bias_by_res_all[entry_index, :, :] = bias_by_res_pad

        # Convert to labels
        # Integer-encode the concatenated sequence using the alphabet; [entry_total_residue_count]
        indices: np.ndarray = np.asarray([alphabet.index(a) for a in all_chain_sequence], dtype=np.int32)
        # Store the encoded sequence in the pdb_dict_list array
        residue_type_indices[entry_index, :entry_total_residue_count] = indices
        # Record the per-entry chain metadata
        all_chain_ids_per_entry.append(all_chain_ids)
        visible_chain_ids_per_entry.append(visible_chain_ids)
        masked_chain_ids_per_entry.append(masked_chain_ids)
        masked_chain_lengths_per_entry.append(masked_chain_lengths)

    # Boolean array of padding (NaN) coordinate entries, shape [B, L, 4, 3]
    # ([B, L, 1, 3] in ca_only mode)
    isnan: np.ndarray = np.isnan(X)
    # Per-residue validity mask, shape [B, L];
    # 1.0 when all of a residue's backbone atom coordinates are finite, 0.0 for padded
    # positions and residues with missing (NaN) coordinates
    valid_residue_mask: np.ndarray = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    # Replace NaN padding coordinates with zeros (masked out downstream)
    X[isnan] = 0.

    # Conversion
    # Move the PSSM arrays to torch tensors on the target device
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)  # [B, L]
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)  # [B, L, num_amino_acid_types]
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device)  # [B, L, num_amino_acid_types]

    # Move the tied weighting to a torch tensor [B, L]
    tied_position_weight = torch.from_numpy(tied_position_weight).to(dtype=torch.float32, device=device)

    # Per-residue amino-acid bias [B, L, num_amino_acid_types]
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)

    # 1.0 where consecutive residues belong to the same chain (numbering step 1).
    # For a residue at global column t, residue_idx[:, t] = 100 * (chain_encoding_number - 1) + t,
    # so a raw step of exactly +1 marks a same-chain adjacent pair; a chain boundary
    # jumps by 101 and padding is -100, so both yield 0.0.
    # Shape [B, L-1]: one flag per adjacent pair along the L axis, float32 0.0/1.0
    jumps: np.ndarray = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    # Phi angle uses the previous residue: mask shifted right by one, shape [B, L]
    phi_mask: np.ndarray = np.pad(jumps, [[0, 0], [1, 0]])
    # Psi angle uses the next residue: mask shifted left by one, shape [B, L]
    psi_mask: np.ndarray = np.pad(jumps, [[0, 0], [0, 1]])
    # Omega angle uses the next residue: mask shifted left by one, shape [B, L]
    omega_mask: np.ndarray = np.pad(jumps, [[0, 0], [0, 1]])
    # Stack the three dihedral masks into [B, L, 3]
    dihedral_mask: np.ndarray = np.stack([phi_mask, psi_mask, omega_mask], axis=-1)
    
    # Move all remaining arrays to torch tensors on the target device; the shapes
    # are unchanged by the conversion
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)  # [B, L, 3]
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)  # [B, L]
    residue_type_indices = torch.from_numpy(residue_type_indices).to(dtype=torch.long, device=device)  # [B, L]
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)  # [B, L, 4, 3] or [B, L, 1, 3]
    valid_residue_mask = torch.from_numpy(valid_residue_mask).to(dtype=torch.float32, device=device)  # [B, L]
    chain_design_mask = torch.from_numpy(chain_design_mask).to(dtype=torch.float32, device=device)  # [B, L]
    free_residue_mask = torch.from_numpy(free_residue_mask).to(dtype=torch.float32, device=device)  # [B, L]
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)  # [B, L, num_amino_acid_types]
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)  # [B, L]
    # Drop the atom axis in CA-only mode
    if ca_only:
        X = X[:, :, 0, :]  # [B, L, 3]
    # Bundle the packed batch; keyword construction keeps the field-to-variable
    # mapping explicit instead of relying on the positional order below
    return TiedFeatures(
        X=X,  # [B, L, 4, 3]; [B, L, 3] when ca_only
        S=residue_type_indices,  # [B, L] int64; local name is residue_type_indices
        mask=valid_residue_mask,  # [B, L] float32; local name is valid_residue_mask
        sequence_lengths=sequence_lengths,  # [B] int32
        chain_M=chain_design_mask,  # [B, L] float32; local name is chain_design_mask
        chain_encoding_all=chain_encoding_all,  # [B, L] int64
        all_chain_ids_per_entry=all_chain_ids_per_entry,  # length B, one list[str] per entry
        visible_chain_ids_per_entry=visible_chain_ids_per_entry,  # length B, one list[str] per entry
        masked_chain_ids_per_entry=masked_chain_ids_per_entry,  # length B, one list[str] per entry
        masked_chain_lengths_per_entry=masked_chain_lengths_per_entry,  # length B, one list[int] per entry
        chain_M_pos=free_residue_mask,  # [B, L] float32; local name is free_residue_mask
        omit_AA_mask=omit_AA_mask,  # [B, L, num_amino_acid_types] float32
        residue_idx=residue_idx,  # [B, L] int64
        dihedral_mask=dihedral_mask,  # [B, L, 3] float32
        tied_position_groups_per_entry=tied_position_groups_per_entry,  # length B, one list[list[int]] per entry
        pssm_coef_all=pssm_coef_all,  # [B, L] float32
        pssm_bias_all=pssm_bias_all,  # [B, L, num_amino_acid_types] float32
        pssm_log_odds_all=pssm_log_odds_all,  # [B, L, num_amino_acid_types] float32
        bias_by_res_all=bias_by_res_all,  # [B, L, num_amino_acid_types] float32
        tied_position_weight=tied_position_weight,  # [B, L] float32
    )


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

class StructureDataset():
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
        alphabet=PROTEIN_ALPHABET_WITH_GAP):
        alphabet_set = set(alphabet)
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq'] 
                name = entry['name']

                # Convert raw coords to np arrays
                #for key, val in entry['coords'].items():
                #    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set(seq).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count['bad_seq_length'] += 1
                    else:
                        discard_count['too_long'] += 1
                else:
                    if verbose:
                        print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print(f'{len(self.data)} valid entries, {i+1} entries checked, {elapsed:.1f} s consumed')
            if verbose:
                print('discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class StructureDatasetPDB():
    '''
    A lightweight in-memory dataset wrapping pre-parsed PDB entries.

    Each entry is a PDB dict produced by ``parse_PDB`` (keys include ``'seq'``,
    ``'name'``, and ``'coords_chain_<ID>'``).  The constructor filters entries
    by alphabet validity and sequence length, storing only the valid entry
    in ``self.data``.

    '''
    def __init__(self, 
                 pdb_dict_list: list[dict[str, Any]], 
                 verbose: bool = True,
                 truncate: int | None = None, 
                 max_length: int = 10000,
                 alphabet: str = PROTEIN_ALPHABET_WITH_GAP) -> None:
        """
        Args:
            pdb_dict_list: list of parsed PDB entry dicts (output of ``parse_PDB``).
            verbose: if True, print progress every 1000 entries.
            truncate: if set, stop collecting once this many valid entries are gathered.
            max_length: maximum allowed total residue count; longer entries are discarded.
            alphabet: the one-letter amino-acid alphabet used for validation.
        """
        # build a set of valid amino acid letters for fast membership testing
        alphabet_set: set[str] = set(alphabet)
        # track how many entries were discarded and for what reason
        discard_count: dict[str, int] = {
            'bad_chars': 0,
            'too_long': 0,
        }

        # list of entries that passed all filters
        self.data: list[dict[str, Any]] = []

        # record the start time for progress reporting
        start: float = time.time()
        # iterate over each parsed PDB entry with its index
        for i, entry in enumerate(pdb_dict_list):
            # extract the one-letter amino acid sequence from the entry
            seq: str = entry['seq']

            # find any characters in the sequence that are not in the alphabet
            bad_chars: set[str] = set(seq).difference(alphabet_set)
            # if the sequence contains only valid characters
            if len(bad_chars) == 0:
                # check whether the sequence length is within the allowed limit
                if len(seq) <= max_length:
                    # entry passed all filters; add it to the dataset
                    self.data.append(entry)
                else:
                    # sequence exceeds max_length; record the discard reason
                    discard_count['too_long'] += 1
            else:
                # sequence contains invalid characters; record the discard reason
                discard_count['bad_chars'] += 1

            # early exit once the requested number of entries has been collected
            if truncate is not None and len(self.data) == truncate:
                return

            # print progress every 1000 entries when verbose is enabled
            if verbose and (i + 1) % 1000 == 0:
                # compute elapsed time since the loop started
                elapsed: float = time.time() - start
                print(f'{len(self.data)} valid entries, {i+1} entries checked, {elapsed:.1f} seconds consumed')

        # print discard summary after the loop finishes
        if verbose:
            print(f'discarded: {discard_count}')

    def __len__(self) -> int:
        '''Return the number of valid entries stored in the dataset.'''
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        '''
        Return the dataset entry at the given index.

        Args:
            idx: zero-based index into ``self.data``.

        Returns:
            The PDB entry dict at position ``idx``.
        '''
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x: x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
            
            
            
# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V 



class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E



class CA_ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(CA_ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 3, num_positional_embeddings + num_rbf*9 + 7
        self.node_embedding = nn.Linear(node_in,  node_features, bias=False) #NOT USED
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)


    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        dX_mask = (3.6<dX_norm) & (dX_norm<4.0) #exclude CA-CA jumps
        dX = dX*dX_mask[:, :, None]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1+eps, 1-eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2, dim=-1)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)
        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)
        return AD_features, O_features



    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :])**2, -1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Ca, mask, residue_idx, chain_labels):
        """ Featurize coordinates as an attributed graph """
        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_2 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_0[:, 1:, :] = Ca[:, :-1, :]
        Ca_1 = Ca
        Ca_2[:, :-1, :] = Ca[:, 1:, :]

        V, O_features = self._orientations_coarse(Ca, E_idx)
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx)) 
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))


        RBF_all = torch.cat(tuple(RBF_all), dim=-1)


        offset = residue_idx[:, :, None]-residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :])==0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all, O_features), -1)
        

        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        return E, E_idx 




class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :])**2, -1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None]-residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx 



class ProteinMPNN(nn.Module):
    def __init__(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CA_ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, use_input_decoding_order=False, decoding_order=None):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
        if not use_input_decoding_order:
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1-torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs



    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1-torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        #chain_mask_combined = chain_mask*chain_M_pos 
        omit_AA_mask_flag = omit_AA_mask != None


        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:, t_] #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None]) #[B]
            mask_gathered = torch.gather(mask, 1, t[:, None]) #[B]
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :] #[B, 21]
            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l+1].scatter_(1, t[:, None, None].repeat(1, 1, h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits-constant[None, :]*1e8+constant_bias[None, :]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(pssm_bias, 1, t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]))[:, 0]
                    probs = (1-pssm_multi*pssm_coef_gathered[:, None])*probs + pssm_multi*pssm_coef_gathered[:, None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(pssm_log_odds_mask, 1, t[:, None, None].repeat(1, 1, pssm_log_odds_mask.shape[-1]))[:, 0] #[B, 21]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]))[:, 0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21), (chain_mask_gathered[:, :, None,]*probs[:, None, :]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def tied_sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None, chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None, pssm_bias_flag=None, tied_pos=None, tied_position_weight=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(list(itertools.chain(*new_decoding_order)), device=device)[None,].repeat(X.shape[0], 1)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1-torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (mask[:, t]==0).all():
                    S_t = S_true[:, t]
                    for t in t_list:
                        h_S[:, t, :] = self.W_s(S_t)
                        S[:, t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:, t:t+1, :]
                    h_E_t = h_E[:, t:t+1, :, :]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, t:t+1, :, :]
                    mask_t = mask[:, t:t+1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:, t:t+1, :]
                        h_ESV_t = mask_bw[:, t:t+1, :, :] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l+1][:, t, :] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
                    h_V_t = h_V_stack[-1][:, t, :]
                    logit_list.append((self.W_out(h_V_t) / temperature)/len(t_list))
                    logits += tied_position_weight[:, t].unsqueeze(-1)*(self.W_out(h_V_t) / temperature)/len(t_list)
            if done_flag:
                pass
            else:
                bias_by_res_gathered = bias_by_res[:, t, :] #[B, 21]
                probs = F.softmax(logits-constant[None, :]*1e8+constant_bias[None, :]/temperature+bias_by_res_gathered/temperature, dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:, t]
                    pssm_bias_gathered = pssm_bias[:, t]
                    probs = (1-pssm_multi*pssm_coef_gathered[:, None])*probs + pssm_multi*pssm_coef_gathered[:, None]*pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, t]
                    probs_masked = probs*pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:, t]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                S_t_repeat = (chain_mask[:, t]*S_t_repeat + (1-chain_mask[:, t])*S_true[:, t]).long() #hard pick fixed positions
                for t in t_list:
                    h_S[:, t, :] = self.W_s(S_t_repeat)
                    S[:, t] = S_t_repeat
                    all_probs[:, t, :] = probs.float()
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict


    def conditional_probs(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, backbone_only=False):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
  
        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :]==1)[:, 0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 21], device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.
            decoding_order = torch.argsort((order_mask[None,]+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1-torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]
        return log_conditional_probs


    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]], device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

