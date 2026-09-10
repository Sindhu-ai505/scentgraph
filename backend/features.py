"""
ScentGraph - Molecular Featurization Pipeline
Converts SMILES strings into PyTorch Geometric Graph Data objects using RDKit.
Computes comprehensive atom (node) and bond (edge) features.
Also provides 2D structure rendering and chemical property calculation.
"""

import io
import base64
from typing import Optional, Tuple, Dict, Any, List
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# Atomic number one-hot choices: C, N, O, F, P, S, Cl, Br, I + other
ALLOWED_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
# Degree choices: 0, 1, 2, 3, 4, 5
ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5]
# Formal charge choices: -2, -1, 0, 1, 2
ALLOWED_CHARGES = [-2, -1, 0, 1, 2]
# Hybridization choices: SP, SP2, SP3, SP3D, SP3D2, OTHER
ALLOWED_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
# Total H count choices: 0, 1, 2, 3, 4
ALLOWED_H_COUNTS = [0, 1, 2, 3, 4]

# Bond types: SINGLE, DOUBLE, TRIPLE, AROMATIC
ALLOWED_BONDS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

NODE_FEATURE_DIM = (
    len(ALLOWED_ATOMS) + 1 +
    len(ALLOWED_DEGREES) + 1 +
    len(ALLOWED_CHARGES) + 1 +
    len(ALLOWED_HYBRIDIZATIONS) + 1 +
    1 + # aromaticity
    1 + # is in ring
    len(ALLOWED_H_COUNTS) + 1
)

EDGE_FEATURE_DIM = (
    len(ALLOWED_BONDS) + 1 +
    1 + # conjugated
    1   # is in ring
)


def _one_hot(val: Any, allowed: List[Any]) -> List[float]:
    """Helper to generate one-hot vector with an extra 'other' bucket."""
    encoding = [0.0] * (len(allowed) + 1)
    if val in allowed:
        encoding[allowed.index(val)] = 1.0
    else:
        encoding[-1] = 1.0
    return encoding


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """Extract 38-dim one-hot feature vector for an RDKit Atom."""
    features = []
    # 1. Atomic Number
    features.extend(_one_hot(atom.GetAtomicNum(), ALLOWED_ATOMS))
    # 2. Degree
    features.extend(_one_hot(atom.GetTotalDegree(), ALLOWED_DEGREES))
    # 3. Formal Charge
    features.extend(_one_hot(atom.GetFormalCharge(), ALLOWED_CHARGES))
    # 4. Hybridization
    features.extend(_one_hot(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS))
    # 5. Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    # 6. Is in ring
    features.append(1.0 if atom.IsInRing() else 0.0)
    # 7. Total H count
    features.extend(_one_hot(atom.GetTotalNumHs(), ALLOWED_H_COUNTS))
    
    return features


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """Extract 7-dim feature vector for an RDKit Bond."""
    features = []
    # 1. Bond Type
    features.extend(_one_hot(bond.GetBondType(), ALLOWED_BONDS))
    # 2. Conjugation
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    # 3. Is in ring
    features.append(1.0 if bond.IsInRing() else 0.0)
    
    return features


def smiles_to_graph(smiles: str, labels: Optional[List[float]] = None) -> Optional[Data]:
    """
    Parses SMILES into a PyTorch Geometric Data object with node and edge attributes.
    Returns None if SMILES is unparseable or has 0 atoms.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        # Atom features
        atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_feats:
            return None
        x = torch.tensor(atom_feats, dtype=torch.float)

        # Bond edges & features (bidirectional)
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            b_feat = get_bond_features(bond)
            
            # (i, j)
            edge_indices.append([i, j])
            edge_attrs.append(b_feat)
            # (j, i)
            edge_indices.append([j, i])
            edge_attrs.append(b_feat)
            
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            # Single atom molecule without bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float)
            
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = x.size(0)
        
        if labels is not None:
            data.y = torch.tensor([labels], dtype=torch.float)
            
        return data
    except Exception as e:
        return None


def get_molecule_properties(smiles: str) -> Optional[Dict[str, Any]]:
    """Calculates chemical properties for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        return {
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "h_bond_donors": Descriptors.NumHDonors(mol),
            "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True)
        }
    except Exception:
        return None


def render_molecule_png(smiles: str, width: int = 400, height: int = 300) -> Optional[str]:
    """
    Renders 2D molecule structure with dark-mode aesthetic styling.
    Returns base64-encoded PNG data URI.
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
            
        AllChem.Compute2DCoords(mol)
        
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False  # Transparent background
        opts.bondLineWidth = 2.4
        opts.scaleBondWidth = True
        opts.padding = 0.12
        
        # High contrast atom colors suitable for dark background
        opts.setSymbolColour((0.92, 0.94, 0.98))
        
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        png_data = drawer.GetDrawingText()
        b64 = base64.b64encode(png_data).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        return None
