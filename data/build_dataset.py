"""
ScentGraph - Molecular Odor Dataset Builder
Fetches real molecular odor datasets from pyrfume-data:
- leffingwell (~3,500 molecules)
- goodscents (~3,500 molecules)
- flavornet (~700 molecules)
- aromadb (~1,500 molecules)

Cleans, canonicalizes via RDKit, merges across sources,
filters sparse labels (< 30 occurrences), and saves to parquet.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import requests
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit verbose logs during batch parsing
RDLogger.DisableLog('rdApp.*')

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
MIN_LABEL_OCCURRENCE = 30

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"

# Mapping of datasets to required files
DATASET_FILES = {
    "leffingwell": ["molecules.csv", "behavior.csv"],
    "goodscents": ["molecules.csv", "stimuli.csv", "behavior.csv"],
    "flavornet": ["molecules.csv", "behavior.csv"],
    "aromadb": ["molecules.csv", "behavior.csv"]
}

# Stopwords & noise tokens to filter out of descriptor strings
DESCRIPTOR_STOPWORDS = {
    "odor", "odour", "scent", "smell", "note", "notes", "aroma", "taste",
    "flavor", "flavour", "like", "type", "weak", "strong", "very", "slight",
    "slightly", "trace", "characteristic", "diffusive", "nuance", "nuances",
    "heavy", "light", "pure", "quality", "rich", "sharp", "soft", "warmth",
    "natural", "artificial", "synthetic", "freshly", "faint", "faintly",
    "dryout", "undertone", "undertones", "topnote", "topnotes", "fragrance"
}

# Synonym normalization map
SYNONYM_MAP = {
    "fruits": "fruity",
    "fruit": "fruity",
    "flowers": "floral",
    "flower": "floral",
    "woods": "woody",
    "wood": "woody",
    "sweetness": "sweet",
    "citrusy": "citrus",
    "greenish": "green",
    "spices": "spicy",
    "spice": "spicy",
    "herbs": "herbal",
    "herb": "herbal",
    "meats": "meaty",
    "meat": "meaty",
    "fishes": "fishy",
    "fish": "fishy",
    "soapy": "soap",
    "mint": "minty",
    "peppermint": "minty",
    "spearmint": "minty",
    "balsam": "balsamic",
    "roses": "rose",
    "almonds": "almond",
    "apples": "apple",
    "onions": "onion",
    "garlics": "garlic",
    "caramellic": "caramel",
    "vanillic": "vanilla",
    "vegetative": "vegetable",
    "vegetables": "vegetable",
    "earth": "earthy",
    "musks": "musk",
    "musky": "musk",
    "phenols": "phenolic",
    "fat": "fatty",
    "wax": "waxy",
    "sulphur": "sulfur",
    "sulfurous": "sulfur",
    "sulphurous": "sulfur",
    "sulfer": "sulfur",
    "alliaceous": "garlic",
    "anisic": "anise",
    "camphoraceous": "camphor",
    "cortex": "bark",
    "dairy": "milky",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a remote file if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    print(f"  Downloading: {url} -> {dest}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(dest, "wb") as f:
                f.write(resp.content)
            return True
        else:
            print(f"  [WARN] Failed to download {url} (status: {resp.status_code})")
            return False
    except Exception as e:
        print(f"  [ERROR] Download failed for {url}: {e}")
        return False


def fetch_all_raw_datasets():
    """Fetch raw CSV files from pyrfume-data repository."""
    print("=== Fetching Raw Datasets from pyrfume-data ===")
    for ds_name, files in DATASET_FILES.items():
        ds_dir = RAW_DATA_DIR / ds_name
        print(f"Dataset '{ds_name}':")
        for fname in files:
            url = f"{GITHUB_RAW_BASE}/{ds_name}/{fname}"
            dest = ds_dir / fname
            download_file(url, dest)


def clean_descriptor(desc: str) -> List[str]:
    """Clean and normalize a descriptor string into a list of normalized labels."""
    if not isinstance(desc, str):
        return []
    
    # Split by delimiters: semicolon, comma, slash, pipe, or newline
    tokens = re.split(r"[,;/|\n]+", desc.lower())
    results = []
    
    for token in tokens:
        token = token.strip()
        # Remove parentheses and non-alpha chars except hyphens/spaces
        token = re.sub(r"[()\[\]{}'\"?*+]", "", token).strip()
        # If token has multiple words, check if it contains a known compound or split
        if not token or len(token) < 2:
            continue
            
        # Replace hyphens with spaces
        token = token.replace("-", " ").strip()
        subwords = token.split()
        
        # If all subwords are stopwords, skip
        filtered_subwords = [w for w in subwords if w not in DESCRIPTOR_STOPWORDS]
        if not filtered_subwords:
            continue
            
        # Map common terms or subterms
        normalized = " ".join(filtered_subwords)
        normalized = SYNONYM_MAP.get(normalized, normalized)
        
        # If the phrase is composite like "sweet floral", split into individual descriptors
        if len(filtered_subwords) > 1 and normalized not in {"orange blossom", "fresh cut grass"}:
            for subw in filtered_subwords:
                subw = SYNONYM_MAP.get(subw, subw)
                if subw not in DESCRIPTOR_STOPWORDS and len(subw) > 2:
                    results.append(subw)
        else:
            if normalized not in DESCRIPTOR_STOPWORDS and len(normalized) > 2:
                results.append(normalized)
                
    return results


def canonicalize_smiles(smiles: str) -> Tuple[str, str]:
    """
    Validates and canonicalizes a SMILES string.
    Returns (clean_smiles, canonical_smiles) or (None, None) if invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, None
        
        # Take largest fragment (remove salts/solvents like counterions)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None, None
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        
        # Require at least 2 heavy atoms
        if largest_frag.GetNumHeavyAtoms() < 2:
            return None, None
            
        canonical = Chem.MolToSmiles(largest_frag, canonical=True, isomericSmiles=False)
        clean = Chem.MolToSmiles(largest_frag, canonical=True, isomericSmiles=True)
        return clean, canonical
    except Exception:
        return None, None


def parse_leffingwell() -> List[Dict]:
    """Parse Leffingwell odor dataset."""
    m_path = RAW_DATA_DIR / "leffingwell" / "molecules.csv"
    b_path = RAW_DATA_DIR / "leffingwell" / "behavior.csv"
    if not (m_path.exists() and b_path.exists()):
        print("  [SKIP] Leffingwell files not found.")
        return []

    mol_df = pd.read_csv(m_path)
    beh_df = pd.read_csv(b_path)
    
    # In leffingwell, behavior.csv has Stimulus == CID, and binary columns
    # First column is Stimulus, remaining are descriptor names
    descriptor_cols = [c for c in beh_df.columns if c != "Stimulus"]
    
    records = []
    beh_dict = {}
    for _, row in beh_df.iterrows():
        stim = row["Stimulus"]
        labels = []
        for col in descriptor_cols:
            if row[col] == 1 or row[col] is True:
                norm_labels = clean_descriptor(col)
                labels.extend(norm_labels)
        if labels:
            beh_dict[stim] = list(set(labels))
            
    for _, row in mol_df.iterrows():
        cid = row["CID"]
        smiles = row.get("IsomericSMILES")
        if cid in beh_dict and smiles:
            records.append({
                "source": "leffingwell",
                "smiles": smiles,
                "labels": beh_dict[cid]
            })
            
    print(f"  Parsed Leffingwell: {len(records)} entries")
    return records


def parse_goodscents() -> List[Dict]:
    """Parse GoodScents dataset."""
    m_path = RAW_DATA_DIR / "goodscents" / "molecules.csv"
    s_path = RAW_DATA_DIR / "goodscents" / "stimuli.csv"
    b_path = RAW_DATA_DIR / "goodscents" / "behavior.csv"
    if not (m_path.exists() and s_path.exists() and b_path.exists()):
        print("  [SKIP] GoodScents files not found.")
        return []

    mol_df = pd.read_csv(m_path)
    stim_df = pd.read_csv(s_path)
    beh_df = pd.read_csv(b_path)
    
    # Map Stimulus -> CID
    stim_to_cid = dict(zip(stim_df["Stimulus"], stim_df["CID"]))
    # Map CID -> IsomericSMILES
    cid_to_smiles = dict(zip(mol_df["CID"], mol_df["IsomericSMILES"]))
    
    records = []
    for _, row in beh_df.iterrows():
        stim = row.get("Stimulus")
        desc_str = row.get("Descriptors")
        cid = stim_to_cid.get(stim)
        smiles = cid_to_smiles.get(cid)
        if smiles and isinstance(desc_str, str):
            labels = clean_descriptor(desc_str)
            if labels:
                records.append({
                    "source": "goodscents",
                    "smiles": smiles,
                    "labels": list(set(labels))
                })
                
    print(f"  Parsed GoodScents: {len(records)} entries")
    return records


def parse_flavornet() -> List[Dict]:
    """Parse Flavornet dataset."""
    m_path = RAW_DATA_DIR / "flavornet" / "molecules.csv"
    b_path = RAW_DATA_DIR / "flavornet" / "behavior.csv"
    if not (m_path.exists() and b_path.exists()):
        print("  [SKIP] Flavornet files not found.")
        return []

    mol_df = pd.read_csv(m_path)
    beh_df = pd.read_csv(b_path)
    
    cid_to_smiles = dict(zip(mol_df["CID"], mol_df["IsomericSMILES"]))
    records = []
    for _, row in beh_df.iterrows():
        cid = row.get("Stimulus")
        desc_str = row.get("Descriptors")
        smiles = cid_to_smiles.get(cid)
        if smiles and isinstance(desc_str, str):
            labels = clean_descriptor(desc_str)
            if labels:
                records.append({
                    "source": "flavornet",
                    "smiles": smiles,
                    "labels": list(set(labels))
                })
                
    print(f"  Parsed Flavornet: {len(records)} entries")
    return records


def parse_aromadb() -> List[Dict]:
    """Parse AromaDB dataset."""
    m_path = RAW_DATA_DIR / "aromadb" / "molecules.csv"
    b_path = RAW_DATA_DIR / "aromadb" / "behavior.csv"
    if not (m_path.exists() and b_path.exists()):
        print("  [SKIP] AromaDB files not found.")
        return []

    mol_df = pd.read_csv(m_path)
    beh_df = pd.read_csv(b_path)
    
    cid_to_smiles = dict(zip(mol_df["CID"], mol_df["IsomericSMILES"]))
    records = []
    for _, row in beh_df.iterrows():
        cid = row.get("Stimulus")
        desc_str = row.get("Filtered Descriptors")
        if not isinstance(desc_str, str):
            desc_str = row.get("Raw Descriptors")
            
        smiles = cid_to_smiles.get(cid)
        if smiles and isinstance(desc_str, str):
            labels = clean_descriptor(desc_str)
            if labels:
                records.append({
                    "source": "aromadb",
                    "smiles": smiles,
                    "labels": list(set(labels))
                })
                
    print(f"  Parsed AromaDB: {len(records)} entries")
    return records


def build_and_clean_dataset():
    """Main pipeline execution."""
    fetch_all_raw_datasets()
    
    print("\n=== Parsing Raw Datasets ===")
    all_records = []
    all_records.extend(parse_leffingwell())
    all_records.extend(parse_goodscents())
    all_records.extend(parse_flavornet())
    all_records.extend(parse_aromadb())
    
    print(f"\nTotal raw records extracted: {len(all_records)}")
    
    print("\n=== Canonicalizing SMILES & Merging Duplicates ===")
    merged_data: Dict[str, Dict] = {}
    invalid_smiles_count = 0
    
    for rec in all_records:
        raw_smiles = rec["smiles"]
        clean_smi, canon_smi = canonicalize_smiles(raw_smiles)
        if not canon_smi:
            invalid_smiles_count += 1
            continue
            
        if canon_smi not in merged_data:
            merged_data[canon_smi] = {
                "smiles": clean_smi,
                "canonical_smiles": canon_smi,
                "labels": set(rec["labels"]),
                "sources": {rec["source"]}
            }
        else:
            merged_data[canon_smi]["labels"].update(rec["labels"])
            merged_data[canon_smi]["sources"].add(rec["source"])
            
    print(f"Dropped invalid/unparseable SMILES: {invalid_smiles_count}")
    print(f"Unique canonical molecules before label filtering: {len(merged_data)}")
    
    # Calculate initial label frequencies
    raw_label_counts: Dict[str, int] = {}
    for entry in merged_data.values():
        for lbl in entry["labels"]:
            raw_label_counts[lbl] = raw_label_counts.get(lbl, 0) + 1
            
    print(f"Total raw distinct labels found: {len(raw_label_counts)}")
    
    # Filter labels appearing >= MIN_LABEL_OCCURRENCE (30)
    valid_labels = {lbl for lbl, count in raw_label_counts.items() if count >= MIN_LABEL_OCCURRENCE}
    print(f"Labels with >= {MIN_LABEL_OCCURRENCE} occurrences: {len(valid_labels)}")
    
    # Filter molecules: keep only valid labels, drop molecules that end up with 0 labels
    cleaned_rows = []
    final_label_counts: Dict[str, int] = {lbl: 0 for lbl in valid_labels}
    
    for canon_smi, item in merged_data.items():
        filtered_labels = sorted(list(item["labels"].intersection(valid_labels)))
        if filtered_labels:
            for l in filtered_labels:
                final_label_counts[l] += 1
            cleaned_rows.append({
                "smiles": item["smiles"],
                "canonical_smiles": item["canonical_smiles"],
                "labels": filtered_labels,
                "num_labels": len(filtered_labels),
                "sources": ",".join(sorted(list(item["sources"])))
            })
            
    df = pd.DataFrame(cleaned_rows)
    print(f"\nFinal cleaned dataset size: {len(df)} molecules with {len(valid_labels)} odor labels.")
    
    # Save outputs
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_DIR / "molecules_odors.parquet"
    csv_path = PROCESSED_DATA_DIR / "molecules_odors.csv"
    stats_path = PROCESSED_DATA_DIR / "dataset_stats.json"
    vocab_path = PROCESSED_DATA_DIR / "label_vocab.json"
    
    df.to_parquet(parquet_path, index=False)
    # For CSV, serialize labels list to semicolon string
    csv_df = df.copy()
    csv_df["labels"] = csv_df["labels"].apply(lambda x: ";".join(x))
    csv_df.to_csv(csv_path, index=False)
    
    # Sort label counts descending
    sorted_label_counts = dict(sorted(final_label_counts.items(), key=lambda x: x[1], reverse=True))
    
    stats = {
        "total_molecules": len(df),
        "total_unique_labels": len(valid_labels),
        "min_label_occurrence_threshold": MIN_LABEL_OCCURRENCE,
        "avg_labels_per_molecule": round(float(df["num_labels"].mean()), 2),
        "median_labels_per_molecule": int(df["num_labels"].median()),
        "max_labels_per_molecule": int(df["num_labels"].max()),
        "label_frequencies": sorted_label_counts,
        "top_25_labels": list(sorted_label_counts.items())[:25],
        "source_distribution": df["sources"].value_counts().to_dict()
    }
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(list(sorted_label_counts.keys()), f, indent=2)
        
    print("\n" + "="*50)
    print("        SCENTGRAPH DATASET SUMMARY")
    print("="*50)
    print(f"Total Cleaned Molecules: {stats['total_molecules']}")
    print(f"Total Unique Odor Labels: {stats['total_unique_labels']}")
    print(f"Avg Labels / Molecule:    {stats['avg_labels_per_molecule']}")
    print(f"Max Labels / Molecule:    {stats['max_labels_per_molecule']}")
    print("\nTop 25 Most Frequent Odor Descriptors:")
    for rank, (lbl, count) in enumerate(stats["top_25_labels"], 1):
        pct = (count / stats['total_molecules']) * 100
        print(f"  {rank:2d}. {lbl:<18} : {count:4d} ({pct:5.1f}%)")
    print("="*50)
    print(f"Saved parquet to: {parquet_path}")
    print(f"Saved csv to:     {csv_path}")
    print(f"Saved stats to:   {stats_path}")
    print(f"Saved vocab to:   {vocab_path}")
    print("Dataset build complete successfully.\n")


if __name__ == "__main__":
    build_and_clean_dataset()
