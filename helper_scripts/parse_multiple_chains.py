import argparse

def main(args):

    import numpy as np
    import os, time, gzip, json
    import glob
    import string
    import sys

    # Add the repository root to sys.path so this standalone script can reuse the shared
    # constants without importing the torch-dependent protein_mpnn_utils module.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from constants import FULL_BACKBONE_ATOM_NAMES, NUMERIC_CHAIN_ID_COUNT
    
    folder_with_pdbs_path = args.input_path
    save_path = args.output_path
    ca_only = args.ca_only
    
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']
    
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    
    def N_to_AA(x):
      # [[0,1,2,3]] -> ["ARND"]
      x = np.array(x);
      if x.ndim == 1: x = x[None]
      return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]
    
    
    def parse_PDB_biounits(x, atoms=FULL_BACKBONE_ATOM_NAMES, chain=None):
      '''
      input:  x = PDB filename
              atoms = atoms to extract (optional)
      output: (length, atoms, coords=(x,y,z)), sequence
      '''
      xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
      for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()
    
        if line[:6] == "HETATM" and line[17:17+3] == "MSE":
          line = line.replace("HETATM", "ATOM  ")
          line = line.replace("MSE", "MET")
    
        if line[:4] == "ATOM":
          ch = line[21:22]
          if ch == chain or chain is None:
            atom = line[12:12+4].strip()
            resi = line[17:17+3]
            resn = line[22:22+5].strip()
            x, y, z = [float(line[i:(i+8)]) for i in [30, 38, 46]]
    
            if resn[-1].isalpha(): 
                resa, resn = resn[-1], int(resn[:-1])-1
            else: 
                resa, resn = "", int(resn)-1
    #         resn = int(resn)
            if resn < min_resn: 
                min_resn = resn
            if resn > max_resn: 
                max_resn = resn
            if resn not in xyz: 
                xyz[resn] = {}
            if resa not in xyz[resn]: 
                xyz[resn][resa] = {}
            if resn not in seq: 
                seq[resn] = {}
            if resa not in seq[resn]: 
                seq[resn][resa] = resi
    
            if atom not in xyz[resn][resa]:
              xyz[resn][resa][atom] = np.array([x, y, z])
    
      # convert to numpy arrays, fill in missing values
      seq_, xyz_ = [], []
      try:
          for resn in range(min_resn, max_resn+1):
            if resn in seq:
              for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k], states - 1))
            else: seq_.append(states - 1)
            if resn in xyz:
              for k in sorted(xyz[resn]):
                for atom in atoms:
                  if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
                  else: xyz_.append(np.full(3, np.nan))
            else:
              for atom in atoms: xyz_.append(np.full(3, np.nan))
          return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
      except TypeError:
          return None, None
    
    
    
    pdb_dict_list = []
    
    alphabetic_chain_ids = list(string.ascii_uppercase + string.ascii_lowercase)
    numeric_chain_ids = [str(item) for item in range(NUMERIC_CHAIN_ID_COUNT)]
    candidate_chain_ids = alphabetic_chain_ids + numeric_chain_ids
    
    biounit_names = glob.glob(os.path.join(folder_with_pdbs_path, '*.pdb'))
    for biounit in biounit_names:
        pdb_dict = {}
        num_parsed_chains = 0
        concat_seq = ''
        for chain_id in candidate_chain_ids:
            if ca_only:
                backbone_atom_names = ('CA',)
            else:
                backbone_atom_names = FULL_BACKBONE_ATOM_NAMES
            xyz, seq = parse_PDB_biounits(biounit, atoms=backbone_atom_names, chain=chain_id)
            if xyz is not None:
                # parse_PDB_biounits returns a one-element list of sequence strings
                concat_seq += seq[0]
                pdb_dict[f'seq_chain_{chain_id}'] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain[f'CA_chain_{chain_id}'] = xyz[:, 0, :].tolist()
                else:
                    coords_dict_chain[f'N_chain_{chain_id}'] = xyz[:, 0, :].tolist()
                    coords_dict_chain[f'CA_chain_{chain_id}'] = xyz[:, 1, :].tolist()
                    coords_dict_chain[f'C_chain_{chain_id}'] = xyz[:, 2, :].tolist()
                    coords_dict_chain[f'O_chain_{chain_id}'] = xyz[:, 3, :].tolist()
                pdb_dict[f'coords_chain_{chain_id}'] = coords_dict_chain
                num_parsed_chains += 1
        pdb_file_name = os.path.basename(biounit)
        pdb_dict['name'] = os.path.splitext(pdb_file_name)[0]
        pdb_dict['num_of_chains'] = num_parsed_chains
        pdb_dict['seq'] = concat_seq
        pdb_dict_list.append(pdb_dict)
            
            
    with open(save_path, 'w') as f:
        for entry in pdb_dict_list:
            f.write(json.dumps(entry) + '\n')
           

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
    argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary of parsed pdbs")
    argparser.add_argument("--ca_only", action="store_true", default=False, help="parse a backbone-only structure (default: false)")

    args = argparser.parse_args()
    main(args)
