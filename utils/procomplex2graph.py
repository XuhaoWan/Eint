from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem
from ase import Atoms


Total_features = {
    'Atomic_number_list': list(range(1, 119)) + ['other'],
    'Bond_length_list': list(range(0, 22)),
    'Angle_list': list(range(0, 181)),
    'Chirality_list': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'Degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'other'],
    'Formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'other'],
    'Valence_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'other'],
    'NumberH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'other'],
    'Radicale_list': [0, 1, 2, 3, 4, 'other'],
    'Hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'other'],
    'Aromatic_list': [False, True],
    'Ring_list': [False, True],
    'Bond_type_list': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'other'],
    'Bond_stereo_list': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY', 'other'],
    'Conjugated_list': [False, True],
}


def get_index(lst, ele):
    try:
        return lst.index(ele)
    except Exception:
        return len(lst) - 1


def Dis_Bondlength(L, min_v, max_v, step=0.025):
    # exactly as mol2graph.py (0..21)
    if L < min_v:
        Ldis = 0
    elif L > max_v:
        Ldis = 21
    else:
        Ldis = (((L - min_v) * 1000) // (step * 1000)) + 1
    return int(Ldis)


def Get_Bondlength(mola: Atoms, i: int, j: int):
    BL = float(mola.get_distance(i, j))
    return Dis_Bondlength(BL, 1.20, 1.70)


def Get_Angle(mola: Atoms, i: int, j: int):
    # consistent with mol2graph.py behavior
    if i == 0 or j == 0:
        Angle = 0
    else:
        Angle = float(mola.get_angle(j, i, 0))
    return int(round(Angle))


def Atom2Feature(atom: Chem.Atom):
    Atom_feature = [
        get_index(Total_features['Atomic_number_list'], atom.GetAtomicNum()),
        Total_features['Chirality_list'].index(str(atom.GetChiralTag())),
        get_index(Total_features['Degree_list'], atom.GetTotalDegree()),
        get_index(Total_features['Formal_charge_list'], atom.GetFormalCharge()),
        get_index(Total_features['Valence_list'], atom.GetTotalValence()),
        get_index(Total_features['NumberH_list'], atom.GetTotalNumHs()),
        get_index(Total_features['Radicale_list'], atom.GetNumRadicalElectrons()),
        get_index(Total_features['Hybridization_list'], str(atom.GetHybridization())),
        Total_features['Aromatic_list'].index(atom.GetIsAromatic()),
        Total_features['Ring_list'].index(atom.IsInRing()),
    ]
    return Atom_feature


def Bond2Feature(bond: Chem.Bond, mola: Atoms, i: int, j: int):
    Bond_feature = [
        get_index(Total_features['Bond_type_list'], str(bond.GetBondType())),
        Total_features['Bond_length_list'].index(Get_Bondlength(mola, i, j)),
        Total_features['Angle_list'].index(Get_Angle(mola, i, j)),
        Total_features['Aromatic_list'].index(bond.GetIsAromatic()),
        Total_features['Bond_stereo_list'].index(str(bond.GetStereo())),
        Total_features['Conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return Bond_feature

@dataclass
class PDBAtom:
    serial: int
    name: str
    resname: str
    chain: str
    resid: int
    element: str
    xyz: np.ndarray
    is_het: bool


def _parse_pdb_atom_line(line: str) -> Optional[PDBAtom]:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    serial = int(line[6:11])
    name = line[12:16].strip()
    resname = line[17:20].strip()
    chain = (line[21].strip() or "?")
    resid = int(line[22:26])
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    element = line[76:78].strip()
    if not element:
        # fallback from atom name
        element = ''.join([c for c in name if c.isalpha()]).upper()
        element = element[:2] if len(element) >= 2 else element[:1]
    is_het = line.startswith("HETATM")
    return PDBAtom(serial, name, resname, chain, resid, element.upper(), np.array([x, y, z], dtype=float), is_het)


def read_pdb_atoms_and_conect(pdb_path: str) -> Tuple[List[PDBAtom], Dict[int, List[int]]]:
    atoms: List[PDBAtom] = []
    conect: Dict[int, List[int]] = {}
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                a = _parse_pdb_atom_line(line)
                if a is not None:
                    atoms.append(a)
            elif line.startswith("CONECT"):
                parts = line.split()
                if len(parts) >= 3:
                    src = int(parts[1])
                    dsts = [int(x) for x in parts[2:] if x.isdigit()]
                    conect.setdefault(src, [])
                    conect[src].extend(dsts)
    # make symmetric
    for src, dsts in list(conect.items()):
        for dst in dsts:
            conect.setdefault(dst, [])
            if src not in conect[dst]:
                conect[dst].append(src)
    return atoms, conect


def select_pocket_atoms(
    pdb_atoms: List[PDBAtom],
    pocket_atom_serials: Optional[Sequence[int]] = None,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
    pocket_residues_full: Optional[Sequence[Tuple[str, int, str]]] = None,
    keep_het: bool = True,
) -> List[PDBAtom]:
    """
    Provide:
      - pocket_residues_full: list of (chain, resid, resname)
    """
    serial_set = set(int(x) for x in pocket_atom_serials) if pocket_atom_serials is not None else None
    res_set = set((c, int(r)) for (c, r) in pocket_residues) if pocket_residues is not None else None
    res3_set = set((c, int(r), rn.upper()) for (c, r, rn) in pocket_residues_full) if pocket_residues_full is not None else None

    out: List[PDBAtom] = []
    for a in pdb_atoms:
        if not keep_het and a.is_het:
            continue
        ok = False
        if serial_set is not None:
            ok = (a.serial in serial_set)
        elif res3_set is not None:
            ok = ((a.chain, a.resid, a.resname.upper()) in res3_set)
        elif res_set is not None:
            ok = ((a.chain, a.resid) in res_set)
        else:
            raise ValueError("You must provide pocket_atom_serials or pocket_residues(_full).")

        if ok:
            out.append(a)
    if not out:
        raise ValueError("Pocket selection returned 0 atoms. Check your prior indices.")
    return out



_ATOMIC_NUM = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "BR": 35, "I": 53, "MG": 12, "ZN": 30, "CA": 20, "NA": 11, "K": 19,
    "FE": 26, "MN": 25, "CU": 29, "SE": 34,
}

_AROMATIC_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
}

_NEG_RES = {"ASP", "GLU"}
_POS_RES = {"LYS", "ARG"}  


def infer_formal_charge(resname: str, atom_name: str) -> int:
    r = (resname or "UNK").upper()
    a = (atom_name or "").upper().strip()

    # very safe rules only
    if r == "ASP" and a.startswith("OD"):
        return -1
    if r == "GLU" and a.startswith("OE"):
        return -1
    if r == "LYS" and a == "NZ":
        return +1
    if r == "ARG" and a in {"NH1", "NH2", "NE"}:
        return +1
    return 0


def infer_aromatic_and_ring(resname: str, atom_name: str) -> Tuple[bool, bool]:
    r = (resname or "UNK").upper()
    a = (atom_name or "").upper().strip()
    if r in _AROMATIC_ATOMS and a in _AROMATIC_ATOMS[r]:
        return True, True
    return False, False


def infer_hybridization(element: str, aromatic: bool, degree: Optional[int]) -> str:
    ele = (element or "C").upper().strip()
    if aromatic:
        return "SP2"
    if ele in {"C", "N", "O", "S", "P"} and degree is not None:
        return "SP2" if degree <= 2 else "SP3"
    return "other"


_COV_RAD = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
            "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39, "SE": 1.20, "FE": 1.24, "ZN": 1.22,
            "MG": 1.30, "CA": 1.74, "NA": 1.66, "K": 2.03}


def infer_pocket_covalent_bonds_by_distance(
    atoms: List[PDBAtom],
    keep_heavy_only: bool = True,
    scale: float = 1.25,
    min_dist: float = 0.4,
    max_dist: float = 2.2,
) -> List[Tuple[int, int]]:
    coords = np.stack([a.xyz for a in atoms], axis=0)
    elems = [a.element for a in atoms]

    n = len(atoms)
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        ei = elems[i]
        if keep_heavy_only and ei == "H":
            continue
        ri = _COV_RAD.get(ei, 0.77)
        for j in range(i + 1, n):
            ej = elems[j]
            if keep_heavy_only and ej == "H":
                continue
            rj = _COV_RAD.get(ej, 0.77)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < min_dist or d > max_dist:
                continue
            thresh = scale * (ri + rj)
            if d <= thresh:
                bonds.append((i, j))
    return bonds


def pocket_degrees_and_hcount(
    atoms: List[PDBAtom],
    bonds: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(atoms)
    deg = np.zeros(n, dtype=int)
    hcount = np.zeros(n, dtype=int)
    for i, j in bonds:
        deg[i] += 1
        deg[j] += 1

        if atoms[i].element == "H" and atoms[j].element != "H":
            hcount[j] += 1
        if atoms[j].element == "H" and atoms[i].element != "H":
            hcount[i] += 1

    return deg, hcount


def protein_atom_to_feature_rich(
    element: str,
    resname: str,
    atom_name: str,
    degree: Optional[int],
    num_h: Optional[int],
) -> List[int]:
    ele = (element or "C").upper().strip()
    z = _ATOMIC_NUM.get(ele, "other")
    atomic_num_idx = get_index(Total_features["Atomic_number_list"], z if isinstance(z, int) else "other")

    chirality_idx = Total_features["Chirality_list"].index("CHI_UNSPECIFIED")

    deg_val = degree if degree is not None else "other"
    degree_idx = get_index(Total_features["Degree_list"], int(deg_val) if deg_val != "other" else "other")

    fc = infer_formal_charge(resname, atom_name)
    formal_charge_idx = get_index(Total_features["Formal_charge_list"], fc)

    valence_idx = get_index(Total_features["Valence_list"], int(deg_val) if deg_val != "other" else "other")

    nh = int(num_h) if num_h is not None else 0
    num_h_idx = get_index(Total_features["NumberH_list"], nh)

    radical_idx = get_index(Total_features["Radicale_list"], 0)

    aromatic, ring = infer_aromatic_and_ring(resname, atom_name)
    hybrid = infer_hybridization(ele, aromatic, int(deg_val) if deg_val != "other" else None)
    hybrid_idx = get_index(Total_features["Hybridization_list"], hybrid)

    aromatic_idx = Total_features["Aromatic_list"].index(bool(aromatic))
    ring_idx = Total_features["Ring_list"].index(bool(ring))

    return [
        atomic_num_idx,
        chirality_idx,
        degree_idx,
        formal_charge_idx,
        valence_idx,
        num_h_idx,
        radical_idx,
        hybrid_idx,
        aromatic_idx,
        ring_idx,
    ]



@dataclass
class ContactConfig:
    cutoff_A: float = 4.5
    min_A: float = 2.0
    max_A: float = 4.5  

def discretize_contact_distance(dist_A: float, ccfg: ContactConfig) -> int:
    d = float(dist_A)
    if d <= ccfg.min_A:
        return 0
    if d >= ccfg.max_A:
        return 21
    t = (d - ccfg.min_A) / (ccfg.max_A - ccfg.min_A)
    return int(round(t * 21))


def contact_label_hbond_salt(
    lig_atom: Chem.Atom,
    pocket_atom: PDBAtom,
    dist_A: float,
) -> int:
    """
    Heuristic label:
      0 none
      1 hbond (distance-only)
      2 salt bridge (distance-only + residue/atom-name)
      3 both
    """
    # hbond (distance-only)
    hbond = 0
    if dist_A <= 3.5:
        le = lig_atom.GetSymbol().upper()
        pe = pocket_atom.element.upper()
        if le in {"N", "O", "S"} and pe in {"N", "O", "S"}:
            hbond = 1

    # salt bridge
    salt = 0
    if dist_A <= 4.0:
        r = pocket_atom.resname.upper()
        an = pocket_atom.name.upper().strip()
        le = lig_atom.GetSymbol().upper()


        if r in _NEG_RES and an.startswith(("OD", "OE")) and le == "N":
            salt = 1

        if r in _POS_RES and (an in {"NZ", "NH1", "NH2", "NE"}) and le == "O":
            salt = 1

    return (1 if hbond else 0) + (2 if salt else 0)


def contact_edge_feature(dist_A: float, label: int, ccfg: ContactConfig) -> List[int]:

    bond_type_idx = get_index(Total_features["Bond_type_list"], "other")
    bond_len_bin = discretize_contact_distance(dist_A, ccfg)  # 0..21
    angle_idx = 0
    aromatic_idx = Total_features["Aromatic_list"].index(False)

    stereo_map = {0: "STEREONONE", 1: "STEREOZ", 2: "STEREOE", 3: "STEREOCIS"}
    stereo = stereo_map.get(int(label), "STEREOANY")
    stereo_idx = Total_features["Bond_stereo_list"].index(stereo)

    conj_idx = Total_features["Conjugated_list"].index(False)
    return [bond_type_idx, int(bond_len_bin), int(angle_idx), int(aromatic_idx), int(stereo_idx), int(conj_idx)]


@dataclass
class BuildConfig:
    contact_cutoff_A: float = 4.5
    keep_pocket_het: bool = True
    drop_h_nodes: bool = True  
    infer_pocket_bonds_if_no_conect: bool = True


def complex_frame_to_graph_prior(
    lig_mol: Chem.Mol,
    lig_pos_A: np.ndarray,                   
    pocket_pdb_path: str,
    pocket_atom_serials: Optional[Sequence[int]] = None,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
    pocket_residues_full: Optional[Sequence[Tuple[str, int, str]]] = None,
    global_u: Optional[np.ndarray] = None,  
    node_charges: Optional[np.ndarray] = None,  
    cfg: Optional[BuildConfig] = None,
) -> Data:
    cfg = cfg or BuildConfig()
    lig_pos_A = np.asarray(lig_pos_A, dtype=float)
    assert lig_pos_A.ndim == 2 and lig_pos_A.shape[1] == 3


    pdb_atoms, conect = read_pdb_atoms_and_conect(pocket_pdb_path)
    pocket_atoms_all = select_pocket_atoms(
        pdb_atoms,
        pocket_atom_serials=pocket_atom_serials,
        pocket_residues=pocket_residues,
        pocket_residues_full=pocket_residues_full,
        keep_het=cfg.keep_pocket_het,
    )


    serial_to_local = {a.serial: i for i, a in enumerate(pocket_atoms_all)}

    pocket_bonds_local: List[Tuple[int, int]] = []
    if conect:
        for s, dsts in conect.items():
            if s not in serial_to_local:
                continue
            i = serial_to_local[s]
            for d in dsts:
                if d not in serial_to_local:
                    continue
                j = serial_to_local[d]
                if i < j:
                    pocket_bonds_local.append((i, j))

    if (not pocket_bonds_local) and cfg.infer_pocket_bonds_if_no_conect:
        pocket_bonds_local = infer_pocket_covalent_bonds_by_distance(pocket_atoms_all, keep_heavy_only=False)


    deg_all, hcount_all = pocket_degrees_and_hcount(pocket_atoms_all, pocket_bonds_local)

    if cfg.drop_h_nodes:
        keep_idx = [i for i, a in enumerate(pocket_atoms_all) if a.element != "H"]
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        pocket_atoms = [pocket_atoms_all[i] for i in keep_idx]
        deg = np.array([int(deg_all[i]) for i in keep_idx], dtype=int)
        hcount = np.array([int(hcount_all[i]) for i in keep_idx], dtype=int)

        pocket_bonds = []
        for i, j in pocket_bonds_local:
            if i in old_to_new and j in old_to_new:
                pocket_bonds.append((old_to_new[i], old_to_new[j]))
    else:
        pocket_atoms = pocket_atoms_all
        deg = deg_all.astype(int)
        hcount = hcount_all.astype(int)
        pocket_bonds = pocket_bonds_local

    pocket_pos_A = np.stack([a.xyz for a in pocket_atoms], axis=0) if pocket_atoms else np.zeros((0, 3), dtype=float)

    n_lig = lig_mol.GetNumAtoms()
    assert lig_pos_A.shape[0] == n_lig, "lig_pos_A must match RDKit atom order."

    n_pocket = len(pocket_atoms)
    n_total = n_lig + n_pocket


    x_lig = np.array([Atom2Feature(a) for a in lig_mol.GetAtoms()], dtype=np.int64)

    x_pocket = np.array([
        protein_atom_to_feature_rich(
            element=pocket_atoms[k].element,
            resname=pocket_atoms[k].resname,
            atom_name=pocket_atoms[k].name,
            degree=int(deg[k]) if k < len(deg) else None,
            num_h=int(hcount[k]) if k < len(hcount) else None,
        )
        for k in range(n_pocket)
    ], dtype=np.int64)

    x = torch.from_numpy(np.vstack([x_lig, x_pocket])).to(torch.long)

    pos = torch.from_numpy(np.vstack([lig_pos_A, pocket_pos_A]).astype(np.float32))

    lig_mask = torch.zeros((n_total,), dtype=torch.bool)
    lig_mask[:n_lig] = True

    symbols = [lig_mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_lig)] + [a.element for a in pocket_atoms]
    mola = Atoms(symbols=symbols, positions=pos.numpy())

    # ---- edges
    edges: List[Tuple[int, int]] = []
    eattrs: List[List[int]] = []

    for bond in lig_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = Bond2Feature(bond, mola, i, j)
        edges.append((i, j)); eattrs.append(bf)
        edges.append((j, i)); eattrs.append(bf)

    base = n_lig
    for (pi, pj) in pocket_bonds:
        i = base + int(pi)
        j = base + int(pj)

        dist = float(mola.get_distance(i, j))

        bl_bin = Dis_Bondlength(dist, 1.20, 1.70)
        ang = int(np.clip(Get_Angle(mola, i, j), 0, 180))
        bf = [
            Total_features["Bond_type_list"].index("SINGLE"),
            Total_features["Bond_length_list"].index(int(bl_bin)),
            Total_features["Angle_list"].index(int(ang)),
            Total_features["Aromatic_list"].index(False),
            Total_features["Bond_stereo_list"].index("STEREONONE"),
            Total_features["Conjugated_list"].index(False),
        ]
        edges.append((i, j)); eattrs.append(bf)
        edges.append((j, i)); eattrs.append(bf)


    ccfg = ContactConfig(cutoff_A=float(cfg.contact_cutoff_A), min_A=2.0, max_A=float(cfg.contact_cutoff_A))

    if n_pocket > 0:

        d = lig_pos_A[:, None, :] - pocket_pos_A[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1)) 
        pairs = np.argwhere(dist <= cfg.contact_cutoff_A)
        for (i_l, i_p) in pairs:
            i_l = int(i_l)
            i_p = int(i_p)
            dij = float(dist[i_l, i_p])
            label = contact_label_hbond_salt(lig_mol.GetAtomWithIdx(i_l), pocket_atoms[i_p], dij)
            bf = contact_edge_feature(dij, label, ccfg)
            i = i_l
            j = base + i_p
            edges.append((i, j)); eattrs.append(bf)
            edges.append((j, i)); eattrs.append(bf)

    edge_index = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(np.array(eattrs, dtype=np.int64), dtype=torch.long) if eattrs else torch.empty((0, 6), dtype=torch.long)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    data.lig_mask = lig_mask

    if global_u is not None:
        u = np.asarray(global_u, dtype=np.float32).reshape(1, -1)
        data.u = torch.from_numpy(u)

    if node_charges is not None:
        q = np.asarray(node_charges, dtype=np.float32).reshape(-1)
        assert q.shape[0] == n_total, "node_charges must have length N(lig+pocket)."
        data.n_properties_1 = torch.from_numpy(q)

    return data



def rdkit_conformer_positions_A(mol: Chem.Mol, conf_id: int = -1) -> np.ndarray:
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    pos = np.zeros((n, 3), dtype=float)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        pos[i] = [p.x, p.y, p.z]
    return pos


if __name__ == "__main__":

    lig_sdf = "dex.sdf"
    pdb = "a2b.pdb"
    pocket_serials = []  # <-- your prior

    mol = Chem.SDMolSupplier(lig_sdf, removeHs=False)[0]
    lig_pos = rdkit_conformer_positions_A(mol)

    g = complex_frame_to_graph_prior(
        lig_mol=mol,
        lig_pos_A=lig_pos,
        pocket_pdb_path=pdb,
        pocket_atom_serials=pocket_serials,
        global_u=None,
    )

    print(g)
