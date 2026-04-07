from __future__ import annotations

"""
Unified implementation module for scaffold-based vs pharmacophore-based comparison.

This file now contains the real implementation used by the comparison notebook:
- metric-family comparison
- overlap/category analysis
- UMAP generation and visualization
- cross-modality miss analysis

Legacy modules can be kept temporarily for reference, but notebooks should import
only this file.
"""

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MakeScaffoldGeneric
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.DataStructs import TanimotoSimilarity
from collections import defaultdict
from multiprocessing import Pool, cpu_count, current_process
from functools import partial
import os
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


from scipy.spatial.distance import jaccard
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
import io
import base64
from rdkit import RDConfig

DEFAULT_THRESHOLDS = {
    "dis": 0.7,
    "sim": 0.8,
}

DEFAULT_UMAP_ACTIVE_CATEGORY_SAMPLE = 5000
DEFAULT_UMAP_NON_ACTIVE_SAMPLE = 5000


def _data_root(base_path: str) -> str:
    root = os.fspath(base_path)
    if os.path.basename(os.path.normpath(root)) == "data":
        return root
    return os.path.join(root, "data")


def _overlap_dir(base_path: str, receptor: str, *parts: str) -> str:
    return os.path.join(_data_root(base_path), "comparison_outputs", "overlap", receptor, *parts)


def _umap_dir(base_path: str, receptor: str, *parts: str) -> str:
    return os.path.join(_data_root(base_path), "comparison_outputs", "umap", receptor, *parts)


def _figure_dir(base_path: str, *parts: str) -> str:
    return os.path.join(_data_root(base_path), "comparison_outputs", "figures", *parts)


def _sample_category_rows(
    df: pd.DataFrame,
    category: str,
    sample_size: Optional[int],
    random_state: int = 42,
) -> pd.DataFrame:
    subset = df[df["activity_category"] == category][["smiles", "activity_category"]].copy()
    if subset.empty:
        return subset
    if sample_size is None or sample_size <= 0 or len(subset) <= sample_size:
        return subset.reset_index(drop=True)
    return subset.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _build_umap_input_dataframe(
    input_set_df: pd.DataFrame,
    recall_set_df: pd.DataFrame,
    output_set_df: pd.DataFrame,
    active_categories: Iterable[str] = ("both_active", "only_scaf", "only_fp"),
    per_active_category_samples: int = DEFAULT_UMAP_ACTIVE_CATEGORY_SAMPLE,
    include_non_active: bool = True,
    non_active_samples: int = DEFAULT_UMAP_NON_ACTIVE_SAMPLE,
    use_stratified_sampling: bool = True,
    max_output_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    datasets = []

    is_df = input_set_df[["smiles"]].copy()
    is_df["activity_category"] = "IS"
    datasets.append(is_df)

    rs_df = recall_set_df[["smiles"]].copy()
    rs_df["activity_category"] = "RS"
    datasets.append(rs_df)

    if use_stratified_sampling:
        for category in active_categories:
            sampled = _sample_category_rows(
                output_set_df,
                category,
                per_active_category_samples,
                random_state=random_state,
            )
            if not sampled.empty:
                datasets.append(sampled)

        if include_non_active:
            sampled_non_active = _sample_category_rows(
                output_set_df,
                "non_active",
                non_active_samples,
                random_state=random_state,
            )
            if not sampled_non_active.empty:
                datasets.append(sampled_non_active)
    else:
        os_df = output_set_df.copy()
        if max_output_samples is not None and len(os_df) > max_output_samples:
            print(f"Sampling {max_output_samples} from {len(os_df)} output compounds...")
            os_df = os_df.sample(n=max_output_samples, random_state=random_state).reset_index(drop=True)

        categories = list(active_categories)
        if include_non_active:
            categories.append("non_active")
        for category in categories:
            subset = os_df[os_df["activity_category"] == category][["smiles", "activity_category"]]
            if not subset.empty:
                datasets.append(subset)

    umap_input = pd.concat(datasets, ignore_index=True)
    counts = umap_input["activity_category"].value_counts().to_dict()
    print("UMAP category counts:")
    for label in ["IS", "RS", "both_active", "only_scaf", "only_fp", "non_active"]:
        if label in counts:
            print(f"  {label}: {counts[label]}")
    return umap_input


def smiles_to_morgan(smiles: str, radius: int = 3, nbits: int = 2048) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint as numpy array using MorganGenerator."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits, dtype=np.int8)

        generator = GetMorganGenerator(radius=radius, fpSize=nbits)
        fingerprint = generator.GetFingerprint(mol)

        return np.array(fingerprint, dtype=np.int8)

    except Exception:
        return np.zeros(nbits, dtype=np.int8)


def jaccard_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Jaccard distance between two binary vectors."""
    return jaccard(vec1, vec2)


def calculate_distance_chunk(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Calculate distance matrix for a chunk of fingerprints.
    
    Args:
        args: Tuple of (start_idx, end_idx, fingerprints_chunk, all_fingerprints)
    
    Returns:
        Tuple of (start_idx, distance_array)
    """
    start_idx, end_idx, fps_chunk, all_fps = args
    num_samples = len(all_fps)
    
    # Calculate distances for this chunk
    chunk_distances = np.zeros((end_idx - start_idx, num_samples))
    
    for local_i, global_i in enumerate(range(start_idx, end_idx)):
        for j in range(global_i + 1, num_samples):
            dist = jaccard_distance(all_fps[global_i], all_fps[j])
            chunk_distances[local_i, j] = dist
    
    return start_idx, chunk_distances



# ==================== UTILITY FUNCTIONS ====================

def bitstring_to_fp(bitstring: str) -> DataStructs.ExplicitBitVect:
    """Convert fingerprint bitstring to RDKit ExplicitBitVect."""
    return DataStructs.CreateFromBitString(bitstring.strip())


def is_active_by_similarity_bulk(fp_query, active_fp_list: List, threshold: float = 1.0) -> int:
    """
    Check if query FP is similar to any active FP using bulk similarity.
    
    Args:
        fp_query: Query fingerprint
        active_fp_list: List of active fingerprints
        threshold: Similarity threshold
    
    Returns:
        1 if active, 0 otherwise
    """
    if fp_query is None:
        return 0
    sims = DataStructs.BulkTanimotoSimilarity(fp_query, active_fp_list)
    return 1 if max(sims) >= threshold else 0


def safe_scaffold_conversion(smiles: str, scaffold_type: str) -> Optional[str]:
    """
    Safely convert SMILES to scaffold.
    
    Args:
        smiles: Input SMILES
        scaffold_type: 'csk' or 'murcko'
    
    Returns:
        Scaffold SMILES or None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if scaffold_type == 'csk':
            generic = MakeScaffoldGeneric(mol)
            return MurckoScaffoldSmiles(Chem.MolToSmiles(generic))
        elif scaffold_type == 'murcko':
            return MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def _scaffold_conversion_worker(args: Tuple[str, str]) -> Optional[str]:
    smiles, scaffold_type = args
    return safe_scaffold_conversion(smiles, scaffold_type)


def parallel_scaffold_conversion(
    smiles_list: List[str],
    scaffold_type: str,
    ncpus: Optional[int] = None,
    min_parallel_size: int = 2000,
    chunksize: int = 200,
) -> List[Optional[str]]:
    if ncpus is None:
        ncpus = max(1, cpu_count() - 2)

    if current_process().daemon:
        ncpus = 1

    if ncpus <= 1 or len(smiles_list) < min_parallel_size:
        return [safe_scaffold_conversion(smiles, scaffold_type) for smiles in smiles_list]

    args = [(smiles, scaffold_type) for smiles in smiles_list]
    with Pool(processes=ncpus) as pool:
        return pool.map(_scaffold_conversion_worker, args, chunksize=chunksize)


def _bitstring_to_fp_worker(bitstring: Optional[str]):
    if isinstance(bitstring, str):
        return bitstring_to_fp(bitstring)
    return None


def parallel_bitstring_to_fp(
    bitstrings: List[Optional[str]],
    ncpus: Optional[int] = None,
    min_parallel_size: int = 5000,
    chunksize: int = 500,
):
    if ncpus is None:
        ncpus = max(1, cpu_count() - 2)

    if current_process().daemon:
        ncpus = 1

    if ncpus <= 1 or len(bitstrings) < min_parallel_size:
        return [_bitstring_to_fp_worker(bitstring) for bitstring in bitstrings]

    with Pool(processes=ncpus) as pool:
        return pool.map(_bitstring_to_fp_worker, bitstrings, chunksize=chunksize)


def _read_first_existing_csv(candidates: List[str], **kwargs) -> pd.DataFrame:
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, **kwargs)
    raise FileNotFoundError(f"None of the expected files exist: {candidates}")


def _ensure_scaffold_dataframe(
    smiles_df: pd.DataFrame,
    scaffold_type: str,
    cache_path: Optional[str] = None,
    smiles_col: str = "smiles",
    scaffold_col: str = "scaf",
    ncpus: Optional[int] = None,
) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    df = smiles_df.copy()
    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column '{smiles_col}' in dataframe.")

    df[scaffold_col] = parallel_scaffold_conversion(
        df[smiles_col].tolist(),
        scaffold_type=scaffold_type,
        ncpus=ncpus,
    )
    df = df.dropna(subset=[scaffold_col]).drop_duplicates(subset=[smiles_col]).reset_index(drop=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def _load_output_fp_dataframe(
    base_path: str,
    receptor: str,
    generator_name: str,
    split: str,
    number: int,
) -> pd.DataFrame:
    candidates = [
        f"{base_path}/output_sets/ph4/{receptor}/{generator_name}/"
        f"phfp_of_output_set_cluster_{number}_{split}_{generator_name}_with_smiles.csv",
        f"{base_path}/data/output_sets/ph4/{receptor}/{generator_name}/"
        f"phfp_of_output_set_cluster_{number}_{split}_{generator_name}_with_smiles.csv",
    ]
    return _read_first_existing_csv(candidates)


def _load_recall_fp_dataframe(
    base_path: str,
    receptor: str,
    generator_name: str,
    split: str,
    number: int,
) -> pd.DataFrame:
    candidates = [
        f"{base_path}/input_recall_sets/ph4/{receptor}/phfp_of_recall_set_cluster_{number}_{split}_with_smiles.csv",
        f"{base_path}/output_sets/ph4/{receptor}/RS/phfp_of_recall_set_cluster_{number}_{split}_with_smiles.csv",
        f"{base_path}/data/output_sets/ph4/{receptor}/RS/phfp_of_recall_set_cluster_{number}_{split}_with_smiles.csv",
        f"{base_path}/data/output_sets/ph4/{receptor}/{generator_name}/"
        f"phfp_of_recall_set_cluster_{number}_{split}_{generator_name}_with_smiles.csv",
    ]
    return _read_first_existing_csv(candidates)


def _load_recall_smiles_dataframe(
    base_path: str,
    receptor: str,
    split: str,
    number: int,
) -> pd.DataFrame:
    candidates = [
        f"{base_path}/input_recall_sets/{receptor}/cRS_{receptor}_{split}_{number}.csv",
        f"{base_path}/data/input_recall_sets/{receptor}/cRS_{receptor}_{split}_{number}.csv",
    ]
    return _read_first_existing_csv(candidates, header=None, names=["smiles"])


def _load_or_create_output_scaffold_df(
    base_path: str,
    receptor: str,
    scaffold_type: str,
    generator_name: str,
    split: str,
    number: int,
    output_fp_df: pd.DataFrame,
    ncpus: Optional[int] = None,
) -> pd.DataFrame:
    existing_candidates = [
        f"{base_path}/output_sets/scaffold/{receptor}/{scaffold_type}/{generator_name}/"
        f"scaf_of_output_set_cluster_{number}_{split}_{generator_name}_with_smiles.csv",
        f"{base_path}/data/results_scaffold_based/{receptor}/{scaffold_type}_scaffolds/{split}/{generator_name}/"
        f"scaffolds_of_output_set_cluster_{number}_{split}_{generator_name}.csv",
    ]
    for path in existing_candidates:
        if os.path.exists(path):
            return pd.read_csv(path)

    cache_path = (
        f"{base_path}/output_sets/scaffold/{receptor}/{scaffold_type}/{generator_name}/"
        f"scaf_of_output_set_cluster_{number}_{split}_{generator_name}_with_smiles.csv"
    )
    scaffold_input = output_fp_df[["smiles"]].drop_duplicates().copy()
    return _ensure_scaffold_dataframe(
        scaffold_input,
        scaffold_type=scaffold_type,
        cache_path=cache_path,
        smiles_col="smiles",
        scaffold_col="scaf",
        ncpus=ncpus,
    )


def _load_or_create_recall_scaffold_df(
    base_path: str,
    receptor: str,
    scaffold_type: str,
    split: str,
    number: int,
    active_fp_df: pd.DataFrame,
    ncpus: Optional[int] = None,
) -> pd.DataFrame:
    existing_candidates = [
        f"{base_path}/input_recall_sets/scaffold/{receptor}/{scaffold_type}/"
        f"scaf_of_recall_set_cluster_{number}_{split}_with_smiles.csv",
        f"{base_path}/data/results_scaffold_based/{receptor}/{scaffold_type}_scaffolds/{split}/RS/"
        f"scaffolds_of_recall_set_cluster_{number}_{split}_RS.csv",
    ]
    for path in existing_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "scaffold" in df.columns and "scaf" not in df.columns:
                df = df.rename(columns={"scaffold": "scaf"})
            return df

    cache_path = os.path.join(
        _overlap_dir(base_path, receptor, "cache", "scaffold", scaffold_type, "RS"),
        f"scaf_of_recall_set_cluster_{number}_{split}_with_smiles.csv",
    )
    if "smiles" in active_fp_df.columns:
        scaffold_input = active_fp_df[["smiles"]].drop_duplicates().copy()
    else:
        scaffold_input = _load_recall_smiles_dataframe(base_path, receptor, split, number)

    return _ensure_scaffold_dataframe(
        scaffold_input,
        scaffold_type=scaffold_type,
        cache_path=cache_path,
        smiles_col="smiles",
        scaffold_col="scaf",
        ncpus=ncpus,
    )


def smiles_to_image(smiles: str, size: Tuple[int, int] = (300, 300)):
    """Convert SMILES to PIL image."""
    try:
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


# def prepare_phfp_factory():
#     """Prepare pharmacophore fingerprint factory."""
#     fdefName = 'BaseFeatures.fdef'
#     featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
#     sigFactory = SigFactory(featFactory, minPointCount=2, maxPointCount=3)
#     sigFactory.SetBins([(0, 2), (2, 4), (4, 6), (6, 8)])
#     sigFactory.Init()
#     return sigFactory


# def convert_to_phfp(smiles: str) -> Optional[DataStructs.ExplicitBitVect]:
#     """Convert SMILES to pharmacophore fingerprint."""
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None
#         sigFactory = prepare_phfp_factory()
#         fp_sparse = Generate.Gen2DFingerprint(mol, sigFactory)
#         return DataStructs.CreateFromBitString(fp_sparse.ToBitString())
#     except Exception:
#         return None

def prepradet_for_2Df_rdkit():
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)  
    sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3)
    sigFactory.SetBins([(0,2),(2,4),(4,6),(6,8)])
    sigFactory.Init()
    return sigFactory


def convert_to_phfp(mol):
    sigFactory = prepradet_for_2Df_rdkit()
    fp_sparse = Generate.Gen2DFingerprint(Chem.MolFromSmiles(mol), sigFactory)
    #return np.array(fp)
    bitstring = fp_sparse.ToBitString()
    fp = DataStructs.CreateFromBitString(bitstring)
    return fp


def _convert_to_phfp_worker(smiles: Optional[str]):
    if isinstance(smiles, str):
        try:
            return convert_to_phfp(smiles)
        except Exception:
            return None
    return None


def parallel_convert_to_phfp(
    smiles_list: List[Optional[str]],
    ncpus: Optional[int] = None,
    min_parallel_size: int = 500,
    chunksize: int = 100,
):
    if ncpus is None:
        ncpus = max(1, cpu_count() - 2)

    if current_process().daemon:
        ncpus = 1

    if ncpus <= 1 or len(smiles_list) < min_parallel_size:
        return [_convert_to_phfp_worker(smiles) for smiles in smiles_list]

    with Pool(processes=ncpus) as pool:
        return pool.map(_convert_to_phfp_worker, smiles_list, chunksize=chunksize)


def categorize_activity(row) -> str:
    """Categorize compound based on scaffold and FP activity."""
    if row['active_scaf'] == 1 and row['active_fp'] == 1:
        return 'both_active'
    elif row['active_scaf'] == 1:
        return 'only_scaf'
    elif row['active_fp'] == 1:
        return 'only_fp'
    else:
        return 'non_active'


def resolve_threshold(
    split: str,
    user_threshold: Optional[float] = None,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
) -> float:
    if user_threshold is not None:
        return user_threshold
    if split == "dis":
        return dis_threshold
    if split == "sim":
        return sim_threshold
    raise ValueError(f"Unsupported split: {split}")


def _format_threshold_value(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def build_threshold_suffix(
    user_threshold: Optional[float] = None,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
) -> str:
    if user_threshold is not None:
        return _format_threshold_value(user_threshold)
    return f"dis_{_format_threshold_value(dis_threshold)}_sim_{_format_threshold_value(sim_threshold)}"


def build_threshold_dirname(
    user_threshold: Optional[float] = None,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
) -> str:
    if user_threshold is not None:
        return f"threshold_{_format_threshold_value(user_threshold)}"
    return (
        f"threshold_dis_{_format_threshold_value(dis_threshold)}"
        f"_sim_{_format_threshold_value(sim_threshold)}"
    )


def resolve_threshold_dir_candidates(
    split: str,
    user_threshold: Optional[float] = None,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
) -> List[str]:
    resolved_threshold = resolve_threshold(split, user_threshold, dis_threshold, sim_threshold)
    candidates = [build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)]
    legacy_dir = f"threshold_{_format_threshold_value(resolved_threshold)}"
    if legacy_dir not in candidates:
        candidates.append(legacy_dir)
    return candidates


def load_merged_df(
    base_path: str,
    receptor: str,
    generator: str,
    type_scaffold: str,
    split: str,
    number: int,
    user_threshold: Optional[float] = None,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
) -> pd.DataFrame:
    threshold_dirs = resolve_threshold_dir_candidates(
        split,
        user_threshold,
        dis_threshold,
        sim_threshold,
    )
    for threshold_dir in threshold_dirs:
        candidate = os.path.join(
            _overlap_dir(base_path, receptor, "merged", threshold_dir, generator),
            f'merged_df_{generator}_{type_scaffold}_{split}_{number}.csv',
        )
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        "merged_df not found in any threshold directory: "
        + ", ".join(threshold_dirs)
    )


# ==================== PARALLELIZED SIMILARITY SEARCH ====================

def find_closest_molecule_worker(args: Tuple) -> Dict:
    """
    Worker function to find closest molecule for a single query.
    
    Args:
        args: Tuple of (index, query_smiles/fp, reference_list, comparison_type, scaffold_type)
    
    Returns:
        Dictionary with results
    """
    idx, query_item, reference_df, comparison_type, scaffold_type = args
    
    try:
        if comparison_type == 'scaffold':
            # For scaffold comparison, use Morgan fingerprints
            query_mol = Chem.MolFromSmiles(query_item)
            if query_mol is None:
                return {'idx': idx, 'similarity': 0, 'closest_smiles': None, 'closest_scaffold': None}
            
            #query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 3, nBits=4096)
            #reference_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, nBits=4096) 
            #               for s in reference_df['scaffold'].tolist() if Chem.MolFromSmiles(s)]

            fpgen = GetMorganGenerator(radius=3, fpSize=4096)

            query_fp = fpgen.GetFingerprint(query_mol)

            reference_fps = [
                fpgen.GetFingerprint(mol)
                for s in reference_df['scaffold'].tolist()
                if (mol := Chem.MolFromSmiles(s)) is not None
            ]

            
        elif comparison_type == 'fingerprint':
            # For fingerprint comparison, use pharmacophore FPs directly
            query_fp = query_item  # Already a fingerprint object
            reference_fps = reference_df['fp'].tolist()
        
        # Calculate similarities using bulk method
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps)
        
        # Find maximum
        max_idx = int(np.argmax(similarities))
        max_similarity = similarities[max_idx]
        
        # Get closest molecule info
        closest_row = reference_df.iloc[max_idx]
        
        return {
            'idx': idx,
            'similarity': max_similarity,
            'closest_smiles': closest_row['smiles'],
            'closest_scaffold': closest_row.get('scaffold', None)
        }
        
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return {'idx': idx, 'similarity': 0, 'closest_smiles': None, 'closest_scaffold': None}


def find_closest_molecules_parallel(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    comparison_type: str,
    scaffold_type: str,
    query_col: str,
    ncpus: int = None
    ) -> pd.DataFrame:
    """
    Find closest molecules in parallel.
    
    Args:
        query_df: DataFrame with query molecules
        reference_df: DataFrame with reference molecules
        comparison_type: 'scaffold' or 'fingerprint'
        scaffold_type: 'csk' or 'murcko'
        query_col: Column name containing query items
        ncpus: Number of CPUs to use
    
    Returns:
        DataFrame with closest molecule information
    """
    if ncpus is None:
        ncpus = max(1, cpu_count() - 2)
    
    print(f"Finding closest molecules using {ncpus} CPUs...")
    print(f"  Query set: {len(query_df)} molecules")
    print(f"  Reference set: {len(reference_df)} molecules")
    
    # Prepare arguments
    args_list = [
        (idx, query_item, reference_df, comparison_type, scaffold_type)
        for idx, query_item in enumerate(query_df[query_col].tolist())
    ]
    
    # Process in parallel
    with Pool(processes=ncpus) as pool:
        results = pool.map(find_closest_molecule_worker, args_list, chunksize=10)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.reindex(range(len(query_df)))
    
    return results_df


# ==================== MAIN ANALYSIS FUNCTIONS ====================

def analyze_cluster_worker(args: Tuple) -> Dict:
    """
    Worker function to analyze a single cluster in parallel.
    
    Args:
        args: Tuple of (generator_name, receptor, scaffold, split, number, 
                       user_threshold, base_path)
    
    Returns:
        Dictionary with cluster information and success status
    """
    generator_name, receptor, scaffold, split, number, user_threshold, base_path = args
    
    try:
        print(f"  Processing: {generator_name}, cluster {number}, split {split}")
        threshold_dir = build_threshold_dirname(
            None,
            DEFAULT_THRESHOLDS["dis"],
            DEFAULT_THRESHOLDS["sim"],
        )
        
        # Load original pharmacophore tables and create scaffold cache on demand
        fp_df = _load_output_fp_dataframe(
            base_path=base_path,
            receptor=receptor,
            generator_name=generator_name,
            split=split,
            number=number,
        )
        active_fp_df = _load_recall_fp_dataframe(
            base_path=base_path,
            receptor=receptor,
            generator_name=generator_name,
            split=split,
            number=number,
        )
        scaf_df = _load_or_create_output_scaffold_df(
            base_path=base_path,
            receptor=receptor,
            scaffold_type=scaffold,
            generator_name=generator_name,
            split=split,
            number=number,
            output_fp_df=fp_df,
            ncpus=None,
        )
        active_scaf_df = _load_or_create_recall_scaffold_df(
            base_path=base_path,
            receptor=receptor,
            scaffold_type=scaffold,
            split=split,
            number=number,
            active_fp_df=active_fp_df,
            ncpus=None,
        )
        
        # Remove duplicates
        scaf_df = scaf_df.drop_duplicates(subset='smiles', keep='first')
        fp_df = fp_df.drop_duplicates(subset='smiles', keep='first')
        
        # Merge
        merged_df = scaf_df.merge(fp_df, on='smiles')
        
        # Prepare active sets
        active_scaffolds = set(active_scaf_df['scaf'])
        active_fp_objects = parallel_bitstring_to_fp(
            active_fp_df['fp'].tolist(),
            ncpus=None,
            min_parallel_size=5000,
            chunksize=500,
        )
        active_fps = [fp for fp in active_fp_objects if fp is not None]
        
        # Convert FPs to RDKit objects
        merged_df['fp_rdk'] = parallel_bitstring_to_fp(
            merged_df['fp'].tolist(),
            ncpus=None,
            min_parallel_size=5000,
            chunksize=500,
        )
        
        # Mark active compounds
        merged_df['active_scaf'] = merged_df['scaf'].apply(lambda x: 1 if x in active_scaffolds else 0)
        merged_df['active_fp'] = merged_df['fp_rdk'].apply(
            lambda fp: is_active_by_similarity_bulk(fp, active_fps, threshold=user_threshold) if fp else 0
        )
        
        # Categorize
        merged_df['activity_category'] = merged_df.apply(categorize_activity, axis=1)
        
        # Save results
        folder = _overlap_dir(base_path, receptor, "merged", threshold_dir, generator_name)
        os.makedirs(folder, exist_ok=True)
        
        merged_df[['smiles', 'active_scaf', 'active_fp', 'activity_category']].to_csv(
            f'{folder}/merged_df_{generator_name}_{scaffold}_{split}_{number}.csv',
            index=False
        )
        
        # Get statistics
        stats = merged_df['activity_category'].value_counts()
        
        result = {
            'generator': generator_name,
            'receptor': receptor,
            'scaffold': scaffold,
            'split': split,
            'cluster': number,
            'both_active': stats.get('both_active', 0),
            'only_scaf': stats.get('only_scaf', 0),
            'only_fp': stats.get('only_fp', 0),
            'non_active': stats.get('non_active', 0),
            'total': len(merged_df),
            'status': 'success'
        }
        
        print(f"  ✓ {generator_name}, cluster {number}: both={result['both_active']}, "
              f"scaf={result['only_scaf']}, fp={result['only_fp']}")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error: {generator_name}, cluster {number}: {e}")
        return {
            'generator': generator_name,
            'receptor': receptor,
            'scaffold': scaffold,
            'split': split,
            'cluster': number,
            'both_active': 0,
            'only_scaf': 0,
            'only_fp': 0,
            'non_active': 0,
            'total': 0,
            'status': 'error',
            'error': str(e)
        }


def analyze_generator_single_cluster(
    generator_name: str,
    receptor: str,
    scaffold: str,
    split: str,
    number: int,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ):
    """
    Analyze a single cluster for a generator (wrapper for backward compatibility).
    
    Args:
        generator_name: Name of generator
        receptor: Receptor name
        scaffold: Scaffold type ('csk' or 'murcko')
        split: Split type ('dis' or 'sim')
        number: Cluster number
        user_threshold: Similarity threshold for FP
        base_path: Base data path
    """
    threshold = resolve_threshold(split, user_threshold, dis_threshold, sim_threshold)
    args = (generator_name, receptor, scaffold, split, number, threshold, base_path)
    result = analyze_cluster_worker(args)
    
    if result['status'] == 'success':
        print(f"\n{'='*60}")
        print(f"Analysis complete: {generator_name}, {scaffold}, {split}, cluster {number}")
        print(f"  Both active: {result['both_active']}")
        print(f"  Only scaffold: {result['only_scaf']}")
        print(f"  Only FP: {result['only_fp']}")
        print(f"{'='*60}\n")
    
    return result


def analyze_multiple_clusters_parallel(
    generator_list: List[str],
    receptor: str,
    scaffold: str,
    splits: List[str] = ['dis', 'sim'],
    clusters: List[int] = [0, 1, 2, 3, 4],
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    ncpus: int = None,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ) -> pd.DataFrame:
    """
    Analyze multiple generators and clusters in parallel.
    
    Args:
        generator_list: List of generator names
        receptor: Receptor name
        scaffold: Scaffold type
        splits: List of split types
        clusters: List of cluster numbers
        user_threshold: Similarity threshold
        base_path: Base data path
        ncpus: Number of CPUs (None = auto)
    
    Returns:
        DataFrame with all results
    """
    if ncpus is None:
        ncpus = max(1, cpu_count() - 2)
    
    print(f"\n{'='*70}")
    print(f"PARALLEL CLUSTER ANALYSIS")
    print(f"{'='*70}")
    print(f"Generators: {', '.join(generator_list)}")
    print(f"Receptor: {receptor}")
    print(f"Scaffold: {scaffold}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Clusters: {clusters}")
    print(f"Thresholds: dis={dis_threshold}, sim={sim_threshold}, override={user_threshold}")
    print(f"CPUs: {ncpus}")
    threshold_dir = build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)
    
    # Prepare all task arguments
    args_list = []
    for generator in generator_list:
        for split in splits:
            for cluster in clusters:
                threshold = resolve_threshold(split, user_threshold, dis_threshold, sim_threshold)
                args_list.append((generator, receptor, scaffold, split, cluster, threshold, base_path))
    
    print(f"Total tasks: {len(args_list)}")
    print(f"{'='*70}\n")
    
    # Process in parallel
    if ncpus > 1:
        print(f"Processing with {ncpus} parallel workers...")
        with Pool(processes=ncpus) as pool:
            results = pool.map(analyze_cluster_worker, args_list)
    else:
        print("Processing serially...")
        results = [analyze_cluster_worker(args) for args in args_list]
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Separate successful and failed results
    success_df = results_df[results_df['status'] == 'success']
    error_df = results_df[results_df['status'] == 'error']
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {len(success_df)}/{len(results_df)}")
    if len(error_df) > 0:
        print(f"Failed: {len(error_df)}")
        print("\nFailed tasks:")
        for _, row in error_df.iterrows():
            print(f"  - {row['generator']}, {row['split']}, cluster {row['cluster']}")
    print(f"{'='*70}\n")
    
    # Save summary
    output_dir = _overlap_dir(base_path, receptor, "summaries", threshold_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(
        f'{output_dir}/parallel_analysis_summary_{scaffold}.csv',
        index=False
    )
    
    print(f"Summary saved to: {output_dir}/parallel_analysis_summary_{scaffold}.csv\n")
    
    return results_df


# ==================== MAIN ANALYSIS FUNCTIONS ====================


def compare_fp_to_scaffold(
    receptor: str,
    generator: str,
    type_scaffold: str,
    type_split: str,
    number: int,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    ncpus: int = None,
    max_samples: int = 50,
    random_state: int = 42,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ):
    """
    Compare compounds with active FP but inactive scaffold to active scaffolds.
    
    Args:
        receptor: Receptor name
        generator: Generator name
        type_scaffold: Scaffold type
        type_split: Split type
        number: Cluster number
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        ncpus: Number of CPUs
        max_samples: Maximum number of diverse samples to analyze
        random_state: Random seed used for reproducible subset selection
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    print(f"\n{'='*60}")
    print(f"Comparing FP to Scaffold")
    print(f"Receptor: {receptor}, Generator: {generator}")
    print(f"{'='*60}")
    threshold = resolve_threshold(type_split, user_threshold, dis_threshold, sim_threshold)
    threshold_dir = build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)
    
    # Load merged data
    df = load_merged_df(
        base_path=base_path,
        receptor=receptor,
        generator=generator,
        type_scaffold=type_scaffold,
        split=type_split,
        number=number,
        user_threshold=user_threshold,
        dis_threshold=dis_threshold,
        sim_threshold=sim_threshold,
    )
    
    # Filter only_fp category
    only_fp = df[df['activity_category'] == 'only_fp'].copy()
    
    if only_fp.empty:
        print(f"No only_fp compounds found for {receptor}, {generator}, {type_scaffold}, {type_split}, {number}")
        return
    
    print(f"Found {len(only_fp)} only_fp compounds")
    
    # Add scaffolds
    only_fp['scaffold'] = parallel_scaffold_conversion(
        only_fp['smiles'].tolist(),
        scaffold_type=type_scaffold,
        ncpus=ncpus,
        min_parallel_size=500,
        chunksize=100,
    )
    only_fp = only_fp.dropna(subset=['scaffold'])
    
    print(f"After scaffold conversion: {len(only_fp)} compounds")
    
    # Diversify if too many
    if len(only_fp) > max_samples:
        only_fp['fp_rdk'] = parallel_convert_to_phfp(
            only_fp['smiles'].tolist(),
            ncpus=ncpus,
            min_parallel_size=100,
            chunksize=50,
        )
        only_fp = diversify_fps(
            only_fp,
            n_samples=max_samples,
            random_state=random_state,
        )
        print(f"After diversification: {len(only_fp)} compounds")
    
    # Load active scaffolds
    active_scaf_df = pd.read_csv(
        f'{base_path}/input_recall_sets/{receptor}/cRS_{receptor}_{type_split}_{number}.csv',
        header=None, names=['smiles']
    )
    active_scaf_df['scaffold'] = parallel_scaffold_conversion(
        active_scaf_df['smiles'].tolist(),
        scaffold_type=type_scaffold,
        ncpus=ncpus,
        min_parallel_size=500,
        chunksize=100,
    )
    active_scaf_df = active_scaf_df.dropna(subset=['scaffold'])
    
    print(f"Active scaffolds: {len(active_scaf_df)}")
    
    # Find closest molecules in parallel
    results_df = find_closest_molecules_parallel(
        query_df=only_fp,
        reference_df=active_scaf_df,
        comparison_type='scaffold',
        scaffold_type=type_scaffold,
        query_col='scaffold',
        ncpus=ncpus
    )
    
    # Merge results
    only_fp['tanimoto_similarity'] = results_df['similarity'].values
    only_fp['closest_scaffold_smiles'] = results_df['closest_smiles'].values
    
    # Generate images
    print("Generating images...")
    only_fp['scaffold_image'] = only_fp['scaffold'].apply(smiles_to_image)
    only_fp['closest_scaffold_image'] = only_fp['closest_scaffold_smiles'].apply(smiles_to_image)
    
    # Save results
    folder = _overlap_dir(base_path, receptor, "compare_results_fp_to_scaf", threshold_dir)
    os.makedirs(folder, exist_ok=True)
    
    output_cols = ['smiles', 'active_scaf', 'active_fp', 'activity_category', 
                   'scaffold', 'scaffold_image', 'tanimoto_similarity', 
                   'closest_scaffold_image', 'closest_scaffold_smiles']
    only_fp[output_cols].to_pickle(
        f'{folder}/compare_res_fp_to_active_scaf_{generator}_{type_scaffold}_{type_split}_{number}.pkl'
    )
    only_fp[output_cols].to_csv(
        f'{folder}/compare_res_fp_to_active_scaf_{generator}_{type_scaffold}_{type_split}_{number}.csv',
        index=False
    )
    
    print(f"Results saved to: {folder}")
    print(f"Threshold used: {threshold}")
    print(f"Average Tanimoto similarity: {only_fp['tanimoto_similarity'].mean():.3f}")


def compare_scaffold_to_fp(
    receptor: str,
    generator: str,
    type_scaffold: str,
    type_split: str,
    number: int,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    ncpus: int = None,
    max_samples: int = 50,
    random_state: int = 42,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ):
    """
    Compare compounds with active scaffold but inactive FP to active FPs.
    
    Args:
        receptor: Receptor name
        generator: Generator name
        type_scaffold: Scaffold type
        type_split: Split type
        number: Cluster number
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        ncpus: Number of CPUs
        max_samples: Maximum unique scaffolds to analyze
        random_state: Random seed reserved for reproducible sampling behavior
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    print(f"\n{'='*60}")
    print(f"Comparing Scaffold to FP")
    print(f"Receptor: {receptor}, Generator: {generator}")
    print(f"{'='*60}")
    threshold = resolve_threshold(type_split, user_threshold, dis_threshold, sim_threshold)
    threshold_dir = build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)
    
    # Load merged data
    df = load_merged_df(
        base_path=base_path,
        receptor=receptor,
        generator=generator,
        type_scaffold=type_scaffold,
        split=type_split,
        number=number,
        user_threshold=user_threshold,
        dis_threshold=dis_threshold,
        sim_threshold=sim_threshold,
    )
    
    # Filter only_scaf category
    only_scaf = df[df['activity_category'] == 'only_scaf'].copy()
    
    if only_scaf.empty:
        print(f"No only_scaf compounds found")
        return
    
    print(f"Found {len(only_scaf)} only_scaf compounds")
    
    # Add scaffolds
    only_scaf['scaffold'] = parallel_scaffold_conversion(
        only_scaf['smiles'].tolist(),
        scaffold_type=type_scaffold,
        ncpus=ncpus,
        min_parallel_size=500,
        chunksize=100,
    )
    only_scaf = only_scaf.dropna(subset=['scaffold'])
    
    # Convert to pharmacophore FPs
    print("Converting to pharmacophore fingerprints...")
    only_scaf['fp_neactive'] = parallel_convert_to_phfp(
        only_scaf['smiles'].tolist(),
        ncpus=ncpus,
        min_parallel_size=100,
        chunksize=50,
    )
    only_scaf = only_scaf.dropna(subset=['fp_neactive'])
    
    print(f"After conversion: {len(only_scaf)} compounds")
    
    # Keep only unique scaffolds (limit to max_samples)
    if len(only_scaf) > max_samples:
        only_scaf = (
            only_scaf
            .drop_duplicates(subset="scaffold")
            .sort_values(["scaffold", "smiles"], kind="stable")
            .head(max_samples)
            .reset_index(drop=True)
        )
        print(f"After unique scaffold filtering: {len(only_scaf)} compounds")
    
    # Load active fingerprints
    active_fp_df = pd.read_csv(
        f'{base_path}/input_recall_sets/ph4/{receptor}/'
        f'phfp_of_recall_set_cluster_{number}_{type_split}_with_smiles.csv'
    )
    active_fp_df = active_fp_df.drop_duplicates(subset='fp', keep='first')
    active_fp_df['fp'] = parallel_bitstring_to_fp(
        active_fp_df['fp'].tolist(),
        ncpus=ncpus,
        min_parallel_size=500,
        chunksize=100,
    )
    active_fp_df = active_fp_df.dropna(subset=['fp'])
    
    # Add scaffolds to active FPs
    active_fp_df['scaffold'] = parallel_scaffold_conversion(
        active_fp_df['smiles'].tolist(),
        scaffold_type=type_scaffold,
        ncpus=ncpus,
        min_parallel_size=500,
        chunksize=100,
    )
    
    print(f"Active fingerprints: {len(active_fp_df)}")
    
    # Find closest molecules in parallel
    results_df = find_closest_molecules_parallel(
        query_df=only_scaf,
        reference_df=active_fp_df,
        comparison_type='fingerprint',
        scaffold_type=type_scaffold,
        query_col='fp_neactive',
        ncpus=ncpus
    )
    
    # Merge results
    only_scaf['tanimoto_similarity'] = results_df['similarity'].values
    only_scaf['closest_smiles'] = results_df['closest_smiles'].values
    only_scaf['closest_scaffold'] = results_df['closest_scaffold'].values
    
    # Generate images
    print("Generating images...")
    only_scaf['scaffold_image'] = only_scaf['scaffold'].apply(smiles_to_image)
    only_scaf['smiles_image'] = only_scaf['smiles'].apply(smiles_to_image)
    only_scaf['closest_smiles_image'] = only_scaf['closest_smiles'].apply(smiles_to_image)
    only_scaf['closest_scaffold_image'] = only_scaf['closest_scaffold'].apply(smiles_to_image)
    
    # Save results
    folder = _overlap_dir(base_path, receptor, "compare_results_scaf_to_fp", threshold_dir)
    os.makedirs(folder, exist_ok=True)
    
    output_cols = ['smiles', 'active_scaf', 'active_fp', 'activity_category', 'scaffold',
                   'scaffold_image', 'smiles_image', 'tanimoto_similarity',
                   'closest_smiles_image', 'closest_scaffold_image', 'closest_smiles']
    only_scaf[output_cols].to_csv(
        f'{folder}/compare_res_scaf_to_active_fp_{generator}_{type_scaffold}_{type_split}_{number}.csv',
        index=False
    )
    only_scaf[output_cols].to_pickle(
        f'{folder}/compare_res_scaf_to_active_fp_{generator}_{type_scaffold}_{type_split}_{number}.pkl'
    )
    
    print(f"Results saved to: {folder}")
    print(f"Threshold used: {threshold}")
    print(f"Average Tanimoto similarity: {only_scaf['tanimoto_similarity'].mean():.3f}")


def diversify_fps(
    df: pd.DataFrame,
    n_samples: int = 50,
    fp_col: str = 'fp_rdk',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select n most diverse molecules based on fingerprints.
    
    Args:
        df: DataFrame with fingerprints
        n_samples: Number of samples to select
        fp_col: Column containing fingerprints
        random_state: Random seed used to choose the initial diverse seed
    
    Returns:
        Diversified DataFrame
    """
    
    df = df.dropna(subset=[fp_col])
    
    if len(df) <= n_samples:
        return df
    
    fps = list(df[fp_col])
    indices = list(range(len(fps)))
    rng = np.random.default_rng(random_state)
    
    # Start with a seeded random molecule so repeated runs stay reproducible.
    selected_idx = [indices[int(rng.integers(len(indices)))]]
    remaining_idx = set(indices) - set(selected_idx)

    
    # Greedy diversification
    while len(selected_idx) < n_samples and remaining_idx:
        max_sims = []
        ordered_remaining_idx = sorted(remaining_idx)
        for i in ordered_remaining_idx:
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in selected_idx])
            max_sims.append(max(sims))
        
        # Select molecule with minimum maximum similarity
        best_idx = ordered_remaining_idx[int(np.argmin(max_sims))]
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)
    
    return df.iloc[selected_idx]


# ==================== SUMMARIZATION FUNCTIONS ====================

def summarize_results(
    generators_list: List[str],
    receptor: str,
    scaffold: str,
    splits: List[str],
    numbers: List[int],
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
):
    """
    Summarize results across multiple clusters and splits.
    Creates averaged statistics and pivot tables.
    
    Args:
        generators_list: List of generator names
        receptor: Receptor name
        scaffold: Scaffold type
        splits: List of split types
        numbers: List of cluster numbers
        user_threshold: Optional shared threshold override for both splits
        base_path: Base data path
        dis_threshold: Threshold for dis split when user_threshold is None
        sim_threshold: Threshold for sim split when user_threshold is None
    """
    print(f"\n{'='*60}")
    print(f"Summarizing results for {receptor}")
    print(f"Generators: {', '.join(generators_list)}")
    print(f"{'='*60}")
    threshold_dir = build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)
    threshold_suffix = build_threshold_suffix(user_threshold, dis_threshold, sim_threshold)

    all_results = []
    summary_records = []
    
    for generator in generators_list:
        print(f"\nProcessing generator: {generator}")
        
        for split in splits:
            split_stats = defaultdict(list)
            
            for num in numbers:
                try:
                    df = load_merged_df(
                        base_path=base_path,
                        receptor=receptor,
                        generator=generator,
                        type_scaffold=scaffold,
                        split=split,
                        number=num,
                        user_threshold=user_threshold,
                        dis_threshold=dis_threshold,
                        sim_threshold=sim_threshold,
                    )
                    
                    df['activity_category'] = df.apply(categorize_activity, axis=1)
                    stats = df['activity_category'].value_counts()
                    
                    for category in ['both_active', 'only_scaf', 'only_fp']:
                        count = stats.get(category, 0)
                        split_stats[category].append(count)
                        
                        summary_records.append({
                            'generator': generator,
                            'split': split,
                            'cluster': num,
                            'category': category,
                            'count': count
                        })
                except FileNotFoundError:
                    print(f"  Warning: File not found for cluster {num}, split {split}")
                    continue
            
            # Calculate averages
            averaged_stats = {k: sum(v) / len(v) if v else 0 for k, v in split_stats.items()}
            averaged_stats['generator'] = generator
            averaged_stats['split'] = split
            all_results.append(averaged_stats)
    
    # Save results
    output_dir = _overlap_dir(base_path, receptor, "summaries", threshold_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    avg_df = pd.DataFrame(all_results)
    avg_df.to_csv(f'{output_dir}/all_avg_res_{scaffold}_{threshold_suffix}.csv', index=False)
    
    summary_df = pd.DataFrame(summary_records).pivot_table(
        index=['generator', 'split', 'cluster'],
        columns='category',
        values='count',
        fill_value=0
    ).reset_index()
    summary_df.to_csv(f'{output_dir}/summary_records_{scaffold}_{threshold_suffix}.csv', index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - all_avg_res_{scaffold}_{threshold_suffix}.csv")
    print(f"  - summary_records_{scaffold}_{threshold_suffix}.csv")
    
    return avg_df, summary_df


# ==================== VISUALIZATION FUNCTIONS ====================

def make_cmap_to_white(base_hex_color: str):
    """Create colormap from white to specified color."""
    base_rgb = to_rgb(base_hex_color)
    white_rgb = to_rgb('#f0f0f0')
    colors = [white_rgb, base_rgb]
    return LinearSegmentedColormap.from_list("custom_cmap", colors)


def _load_avg_results_table(
    receptor: str,
    scaffold: str,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
) -> pd.DataFrame:
    threshold_dir = build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)
    threshold_suffix = build_threshold_suffix(user_threshold, dis_threshold, sim_threshold)
    combined_path = os.path.join(
        _overlap_dir(base_path, receptor, "summaries", threshold_dir),
        f'all_avg_res_{scaffold}_{threshold_suffix}.csv',
    )

    if os.path.exists(combined_path):
        return pd.read_csv(combined_path)

    if user_threshold is not None:
        raise FileNotFoundError(combined_path)

    split_frames = []
    for split in ['dis', 'sim']:
        split_threshold = resolve_threshold(split, user_threshold, dis_threshold, sim_threshold)
        legacy_dir = build_threshold_dirname(split_threshold, dis_threshold, sim_threshold)
        legacy_suffix = build_threshold_suffix(split_threshold, dis_threshold, sim_threshold)
        legacy_path = os.path.join(
            _overlap_dir(base_path, receptor, "summaries", legacy_dir),
            f'all_avg_res_{scaffold}_{legacy_suffix}.csv',
        )
        if not os.path.exists(legacy_path):
            raise FileNotFoundError(
                f"Missing combined file {combined_path} and legacy fallback {legacy_path}"
            )
        split_df = pd.read_csv(legacy_path)
        split_frames.append(split_df[split_df['split'] == split].copy())

    if not split_frames:
        raise FileNotFoundError(combined_path)
    return pd.concat(split_frames, ignore_index=True)


def plot_activity_barplots(
    receptor: str,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    categories: List[str] = None,
    use_percentage: bool = True,
    total_count: int = 250000,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ):
    """
    Plot bar charts for both CSK and Murcko scaffolds.
    
    Args:
        receptor: Receptor name
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        categories: List of categories to plot (default: all)
                   Options: ['both_active', 'only_scaf', 'only_fp', 'non_active']
        use_percentage: If True, plot percentages; if False, plot absolute numbers
        total_count: Total number of compounds (for percentage calculation)
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    print(f"\nGenerating activity bar plots for {receptor}...")
    threshold_dir = build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)
    threshold_suffix = build_threshold_suffix(user_threshold, dis_threshold, sim_threshold)
    
    # Default to all categories if not specified
    if categories is None:
        categories = ['both_active', 'only_scaf', 'only_fp']
    
    sns.set(style="whitegrid")
    palette = {
        'both_active': '#1f77b4',
        'only_scaf': '#ff7f0e',
        'only_fp': '#2ca02c',
        'non_active': '#d62728'
    }
    
    # Load data
    df_csk = pd.read_csv(
        os.path.join(
            _overlap_dir(base_path, receptor, "summaries", threshold_dir),
            f'all_avg_res_csk_{threshold_suffix}.csv',
        )
    )
    df_murcko = pd.read_csv(
        os.path.join(
            _overlap_dir(base_path, receptor, "summaries", threshold_dir),
            f'all_avg_res_murcko_{threshold_suffix}.csv',
        )
    )
    
    # Convert to percentages if needed
    if use_percentage:
        for df in [df_csk, df_murcko]:
            for cat in categories:
                if cat in df.columns:
                    df[cat] = df[cat] / total_count * 100
        y_label = "Percentage of compounds (%)"
        value_format = "{:.2f}%"
    else:
        y_label = "Number of compounds"
        value_format = "{:.0f}"
    
    # Melt data (only selected categories)
    def melt_df(df):
        df_melt = df.melt(
            id_vars=['generator', 'split'],
            value_vars=[cat for cat in categories if cat in df.columns],
            var_name='activity_category',
            value_name='value'
        )
        df_melt['generator_split'] = df_melt['generator'] + "_" + df_melt['split']
        return df_melt
    
    df_csk_melt = melt_df(df_csk)
    df_murcko_melt = melt_df(df_murcko)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    def plot_subplot(ax, df_melt, title):
        x_labels = df_melt['generator_split'].unique()
        x_positions = np.arange(len(x_labels))
        bar_width = 0.8 / len(categories)  # Adjust width based on number of categories
        
        # Plot each category
        for i, category in enumerate(categories):
            cat_data = df_melt[df_melt['activity_category'] == category]
            if not cat_data.empty:
                values = cat_data['value'].values
                offset = (i - len(categories)/2 + 0.5) * bar_width
                ax.bar(x_positions + offset, values, width=bar_width,
                      color=palette.get(category, '#999999'), label=category)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_ylim(0, max(df_melt['value']) * 1.2 if not df_melt.empty else 10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format y-axis
        if use_percentage:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))
    
    plot_subplot(axes[0], df_csk_melt, f"CSK scaffold - {', '.join(categories)}")
    plot_subplot(axes[1], df_murcko_melt, f"Murcko scaffold - {', '.join(categories)}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    categories_str = '_'.join(categories)
    unit_str = 'percent' if use_percentage else 'absolute'
    output_file = os.path.join(
        _figure_dir(base_path),
        f"activity_barplots_{receptor}_{categories_str}_{unit_str}_{threshold_suffix}.png",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    output_file = os.path.join(
        _figure_dir(base_path),
        f"activity_barplots_{receptor}_{categories_str}_{unit_str}_{threshold_suffix}.svg",
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar plot saved to: {output_file}")
    plt.show()


def plot_combined_heatmap(
    receptors: List[str] = ['Glucocorticoid_receptor', 'Leukocyte_elastase'],
    generators: List[str] = None,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    title: str = None,
    save_name: str = None,
    metrics: List[str] = None,
    use_percentage: bool = True,
    total_count: int = 250000,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ):
    """
    Plot combined heatmaps for activity statistics across receptors and generators.
    
    Args:
        receptors: List of receptor names
        generators: List of generator names
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        title: Optional title for the plot
        save_name: Optional custom save name
        metrics: List of metrics to plot (default: all)
                Options: ['both_active', 'only_scaf', 'only_fp', 'non_active']
        use_percentage: If True, show percentages; if False, show absolute numbers
        total_count: Total number of compounds (for percentage calculation)
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    print(f"\nGenerating combined heatmap...")
    threshold_dir = build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)
    threshold_suffix = build_threshold_suffix(user_threshold, dis_threshold, sim_threshold)
    
    if generators is None:
        generators = ['Molpher_250k', 'REINVENT_250k', 'DrugEx_GT_epsilon_0.1_250k',
                     'DrugEx_GT_epsilon_0.6_250k', 'DrugEx_RNN_epsilon_0.1_250k',
                     'DrugEx_RNN_epsilon_0.6_250k', 'GB_GA_log_p_mut_r_0.01_250k',
                     'GB_GA_log_p_mut_r_0.5_250k', 'GB_GA_mut_r_0.01_250k',
                     'GB_GA_mut_r_0.5_250k', 'addcarbon_250k', 'enamine_250k']
    
    # Default to all activity metrics if not specified
    if metrics is None:
        metrics = ['both_active', 'only_scaf', 'only_fp']

    metric_display_names = {
        'both_active': 'both-active',
        'only_scaf': 'scaffold-only',
        'only_fp': 'pharm-only',
        'non_active': 'inactive',
    }
    
    metric_base_colors = {
        'both_active': "#97C2F0",
        'only_scaf': "#e97b32",
        'only_fp': "#71ad48",
        'non_active': "#d62728"
    }
    metric_base_colors = {
    'both_active': "#9E96D8",   # sytější levandulová / fialová
    'only_scaf':   "#D989A5",   # sytější růžová
    'only_fp':     "#E3C96A",   # pastelová okrová / tlumená žlutá
    'non_active':  "#B8B8B8"    # neutrální světle šedá
    }
    
    scaffolds = ['csk', 'murcko']
    splits = ['dis', 'sim']
    
    # Collect data (only for selected metrics)
    data = []
    for gen in generators:
        for receptor in receptors:
            for type_scaffold in scaffolds:
                for type_cluster in splits:
                    try:
                        dff = pd.read_csv(
                            os.path.join(
                                _overlap_dir(base_path, receptor, "summaries", threshold_dir),
                                f'all_avg_res_{type_scaffold}_{threshold_suffix}.csv',
                            )
                        )
                        for met in metrics:
                            if met in dff.columns:
                                value = dff[(dff.generator == gen) & (dff.split == type_cluster)][met].iloc[0]
                                if use_percentage:
                                    value = value / total_count * 100
                                data.append([gen, receptor, type_scaffold, type_cluster, met, value])
                    except (FileNotFoundError, IndexError, KeyError):
                        continue
    
    df = pd.DataFrame(data, columns=['Generator', 'Receptor', 'Scaffold', 'Split', 'Metric', 'Value'])
    
    # Create figure
    nrows = len(receptors)
    nmetrics = len(metrics)
    
    fig_width = 1.7 * 4 * nmetrics + 2
    fig_height = 6 * nrows * 1.3
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    outer_gs = fig.add_gridspec(nrows=nrows, ncols=nmetrics, wspace=0.2, hspace=0.1)
    
    for met_idx, metric in enumerate(metrics):
        metric_df = df[df['Metric'] == metric].copy()
        cmap_custom = make_cmap_to_white(metric_base_colors.get(metric, "#999999"))
        
        for rec_idx, receptor in enumerate(receptors):
            inner = outer_gs[rec_idx, met_idx].subgridspec(nrows=1, ncols=4, wspace=0.05, hspace=0.0)
            
            group_axes = []
            for sc_idx, scaffold_type in enumerate(["csk", "murcko"]):
                block_df = metric_df[
                    (metric_df['Receptor'] == receptor) &
                    (metric_df['Scaffold'] == scaffold_type)
                ]
                
                for split_idx, split in enumerate(["dis", "sim"]):
                    col = sc_idx * 2 + split_idx
                    ax = fig.add_subplot(inner[0, col])
                    group_axes.append(ax)
                    
                    sub_df = block_df[block_df['Split'] == split].copy()
                    sub_df = sub_df.set_index('Generator').reindex(generators)
                    heatmap_array = sub_df['Value'].to_numpy().reshape(-1, 1)

                    valid_values = heatmap_array[~np.isnan(heatmap_array)]
                    if valid_values.size == 0:
                        vmin, vmax = 0.0, 1.0
                        max_idx = None
                    else:
                        vmin = float(valid_values.min())
                        vmax = float(valid_values.max())
                        max_idx = int(np.nanargmax(heatmap_array))

                    heatmap_flat = heatmap_array.flatten()

                    # Format annotations based on percentage or absolute
                    annot_array = []
                    for i, val in enumerate(heatmap_flat):
                        if pd.isna(val):
                            text = "NA"
                            suffix = ""
                        elif use_percentage:
                            text = f"{val:.3f}"
                            suffix = " %"
                        else:
                            text = f"{int(val):,}"
                            suffix = ""
                        
                        if max_idx is not None and i == max_idx:
                            text = r"$\bf{" + text + "}$"
                        annot_array.append(f"{text}{suffix}")
                    
                    annot_array = np.array(annot_array).reshape(heatmap_array.shape)
                    
                    show_colorbar = (sc_idx == 1 and split_idx == 1)
                    if show_colorbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                    else:
                        cax = None
                    
                    sns.heatmap(heatmap_array, annot=annot_array, fmt="",
                               cmap=cmap_custom, ax=ax, cbar=show_colorbar,
                               cbar_ax=cax,
                               cbar_kws={'label': metric_display_names.get(metric, metric)} if show_colorbar else None,
                               annot_kws={"size": 14, "color": "black"},
                               vmin=vmin, vmax=vmax)
                    ax.set_aspect("auto")
                    
                    ax.set_xticks([0.5])
                    ax.set_xticklabels([split], rotation=0, ha="center", fontsize=12)
                    
                    if met_idx == 0 and sc_idx == 0 and split_idx == 0:
                        ax.set_ylabel(receptor.replace('_', ' '), fontsize=14)
                        ax.set_yticks(np.arange(len(generators)) + 0.5)
                        new_labels = [g.replace('_epsilon', '\n epsilon')
                                     .replace('_mut_r', '\n mut_r')
                                     .replace('addcarbon', 'AddCarbon')
                                     .replace('_250k', '')
                                     for g in generators]
                        ax.set_yticklabels(new_labels, rotation=0, fontsize=14)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticks([])
                        ax.set_yticklabels([])
            
            if rec_idx == 0:
                p0 = group_axes[0].get_position()
                p1 = group_axes[1].get_position()
                x_mid_csk = (p0.x0 + p1.x1) / 2
                y_top_csk = max(p0.y1, p1.y1) + 0.012
                display_metric = metric_display_names.get(metric, metric)
                fig.text(x_mid_csk, y_top_csk, f"{display_metric} - CSK", ha="center", va="bottom", fontsize=13)
                
                p2 = group_axes[2].get_position()
                p3 = group_axes[3].get_position()
                x_mid_mur = (p2.x0 + p3.x1) / 2
                y_top_mur = max(p2.y1, p3.y1) + 0.012
                fig.text(x_mid_mur, y_top_mur, f"{display_metric} - MURCKO", ha="center", va="bottom", fontsize=13)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    metrics_str = '_'.join(metrics)
    unit_str = 'percent' if use_percentage else 'absolute'
    output_file = save_name or os.path.join(
        _figure_dir(base_path),
        f"heatmap_{metrics_str}_{unit_str}_{threshold_suffix}.png",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    output_file = save_name or os.path.join(
        _figure_dir(base_path),
        f"heatmap_{metrics_str}_{unit_str}_{threshold_suffix}.svg",
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    plt.show()


def display_df_with_images(df: pd.DataFrame, max_rows: int = 20):
    """
    Display DataFrame with molecule images in HTML format.
    
    Args:
        df: DataFrame containing image columns
        max_rows: Maximum number of rows to display (default: 20)
    
    Returns:
        IPython.display.HTML object
    """
    from PIL import Image
    
    # Limit rows for performance
    display_df = df.head(max_rows) if len(df) > max_rows else df.copy()
    
    if len(df) > max_rows:
        print(f"Showing first {max_rows} of {len(df)} rows")
    
    html = """
    <style>
        .df-table { 
            border-collapse: collapse; 
            width: 100%; 
            font-family: Arial, sans-serif;
            table-layout: auto;
        }
        .df-table th { 
            background-color: #4CAF50; 
            color: white; 
            padding: 12px; 
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .df-table td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: center;
            vertical-align: middle;
        }
        .df-table tr:nth-child(even) { 
            background-color: #f9f9f9; 
        }
        .df-table tr:hover {
            background-color: #f5f5f5;
        }
        .smiles-cell {
            text-align: left;
            font-family: monospace;
            font-size: 11px;
            max-width: 400px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .image-cell {
            padding: 5px;
            min-width: 270px;
        }
        .activity-cell {
            font-weight: bold;
            font-size: 13px;
        }
    </style>
    <div style="overflow-x: auto; max-width: 100%;">
    <table class="df-table"><tr>
    """
    
    # Header
    for col in display_df.columns:
        html += f"<th>{col.replace('_', ' ').title()}</th>"
    html += "</tr>"
    
    # Image column names
    image_columns = {'scaffold_image', 'smiles_image', 'closest_scaffold_image', 
                    'closest_smiles_image', 'closest_molecule_image', 'molecule_image', 
                    'csk_image'}
    
    # Process each row
    for row_idx, (idx, row) in enumerate(display_df.iterrows()):
        html += "<tr>"
        
        for col in display_df.columns:
            cell_val = row[col]
            
            # Handle image columns
            if col in image_columns:
                if pd.isna(cell_val) or cell_val is None:
                    html += "<td class='image-cell'>-</td>"
                    continue
                
                try:
                    # Check if it's a PIL Image
                    if hasattr(cell_val, 'mode') and hasattr(cell_val, 'size'):
                        # Create a fresh copy of the image
                        img_copy = Image.new(cell_val.mode, cell_val.size)
                        img_copy.paste(cell_val)
                        
                        # Convert to RGB if necessary
                        if img_copy.mode in ('RGBA', 'LA', 'P'):
                            img_copy = img_copy.convert('RGB')
                        
                        buffer = io.BytesIO()
                        img_copy.save(buffer, format='PNG')
                        buffer.seek(0)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        html += f'<td class="image-cell"><img src="data:image/png;base64,{img_base64}" width="260" height="260" style="display: block; margin: auto;"></td>'
                        buffer.close()
                    else:
                        html += "<td class='image-cell'>Not an image</td>"
                except Exception as e:
                    # Try alternative method
                    try:
                        import numpy as np
                        arr = np.array(cell_val)
                        img_new = Image.fromarray(arr.astype('uint8'))
                        if img_new.mode in ('RGBA', 'LA', 'P'):
                            img_new = img_new.convert('RGB')
                        
                        buffer = io.BytesIO()
                        img_new.save(buffer, format='PNG')
                        buffer.seek(0)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        html += f'<td class="image-cell"><img src="data:image/png;base64,{img_base64}" width="260" height="260" style="display: block; margin: auto;"></td>'
                        buffer.close()
                    except:
                        html += "<td class='image-cell' style='color: red;'>⚠️</td>"
            
            # Handle SMILES columns
            elif col in ['smiles', 'closest_smiles', 'closest_scaffold_smiles', 'scaffold']:
                if pd.notna(cell_val):
                    smiles_str = str(cell_val)
                    html += f"<td class='smiles-cell'>{smiles_str}</td>"
                else:
                    html += "<td>-</td>"
            
            # Handle similarity scores
            elif col == 'tanimoto_similarity':
                if pd.notna(cell_val):
                    try:
                        score = float(cell_val)
                        # Color code by similarity
                        if score >= 0.8:
                            color = '#4CAF50'  # Green
                        elif score >= 0.6:
                            color = '#FF9800'  # Orange
                        else:
                            color = '#f44336'  # Red
                        html += f"<td style='color: {color}; font-weight: bold; font-size: 14px;'>{score:.3f}</td>"
                    except:
                        html += f"<td>{cell_val}</td>"
                else:
                    html += "<td>-</td>"
            
            # Handle activity columns
            elif col in ['active_scaf', 'active_fp']:
                if pd.notna(cell_val):
                    is_active = int(cell_val) == 1
                    color = '#4CAF50' if is_active else '#f44336'
                    text = '✓ Active' if is_active else '✗ Inactive'
                    html += f"<td class='activity-cell' style='color: {color};'>{text}</td>"
                else:
                    html += "<td>-</td>"
            
            # Handle activity category
            elif col == 'activity_category':
                if pd.notna(cell_val):
                    colors = {
                        'both_active': '#2196F3',
                        'only_scaf': '#FF9800', 
                        'only_fp': '#4CAF50',
                        'non_active': '#9E9E9E'
                    }
                    color = colors.get(str(cell_val), '#000')
                    text = str(cell_val).replace('_', ' ').title()
                    html += f"<td class='activity-cell' style='color: {color};'>{text}</td>"
                else:
                    html += "<td>-</td>"
            
            # All other columns
            else:
                if pd.notna(cell_val):
                    val_str = str(cell_val)
                    if len(val_str) > 100:
                        val_str = val_str[:97] + "..."
                    html += f"<td style='max-width: 200px; word-wrap: break-word;'>{val_str}</td>"
                else:
                    html += "<td>-</td>"
        
        html += "</tr>"
    
    html += "</table></div>"
    
    return HTML(html)


# ==================== DIVERSIFICATION ====================

def process_for_umap(
    receptor: str,
    type_scaffold: str,
    type_cluster: str,
    number: int,
    generator: str,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    ncpus: int = None,
    max_output_samples: int = 30000,
    per_active_category_samples: int = DEFAULT_UMAP_ACTIVE_CATEGORY_SAMPLE,
    include_non_active: bool = True,
    non_active_samples: int = DEFAULT_UMAP_NON_ACTIVE_SAMPLE,
    use_stratified_sampling: bool = True,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ):
    """
    Process SMILES datasets, compute fingerprints, and generate UMAP visualization.
    Parallelized distance matrix calculation.
    
    Args:
        receptor: Receptor name
        type_scaffold: Scaffold type
        type_cluster: Cluster type ('dis' or 'sim')
        number: Cluster number
        generator: Generator name
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        ncpus: Number of CPUs for parallel processing
        max_output_samples: Maximum output samples to include in the legacy non-stratified mode
        per_active_category_samples: Maximum number of output molecules per active category
        include_non_active: Whether to include sampled inactive output molecules
        non_active_samples: Maximum number of inactive output molecules
        use_stratified_sampling: If True, use the stratified UMAP workflow as the default canonical mode
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    if ncpus is None:
        ncpus = max(1, cpu_count() - 2)
    
    print(f"\n{'='*60}")
    print(f"Processing UMAP for {generator}")
    print(f"Receptor: {receptor}, Scaffold: {type_scaffold}")
    print(f"Cluster: {type_cluster}_{number}")
    threshold = resolve_threshold(type_cluster, user_threshold, dis_threshold, sim_threshold)
    threshold_dirs = resolve_threshold_dir_candidates(
        type_cluster,
        user_threshold,
        dis_threshold,
        sim_threshold,
    )
    threshold_dir = threshold_dirs[0]
    print(f"Using threshold: {threshold}")
    print(f"{'='*60}")
    
    # Load datasets
    print("Loading datasets...")
    IS = pd.read_csv(
        f'{base_path}/input_recall_sets/{receptor}/cIS_{receptor}_{type_cluster}_{number}.csv',
        names=['smiles']
    )
    RS = pd.read_csv(
        f'{base_path}/input_recall_sets/{receptor}/cRS_{receptor}_{type_cluster}_{number}.csv',
        names=['smiles']
    )
    
    os_file = None
    for candidate_dir in threshold_dirs:
        candidate_file = os.path.join(
            _overlap_dir(base_path, receptor, "merged", candidate_dir, generator),
            f'merged_df_{generator}_{type_scaffold}_{type_cluster}_{number}.csv',
        )
        if os.path.exists(candidate_file):
            os_file = candidate_file
            threshold_dir = candidate_dir
            break

    if os_file is None:
        raise FileNotFoundError(
            "Merged UMAP input not found in any threshold directory: "
            + ", ".join(threshold_dirs)
        )

    OS = pd.read_csv(os_file)


    
    df = _build_umap_input_dataframe(
        input_set_df=IS,
        recall_set_df=RS,
        output_set_df=OS,
        active_categories=("both_active", "only_scaf", "only_fp"),
        per_active_category_samples=per_active_category_samples,
        include_non_active=include_non_active,
        non_active_samples=non_active_samples,
        use_stratified_sampling=use_stratified_sampling,
        max_output_samples=max_output_samples,
        random_state=42,
    )
    
    combined_smiles = df.smiles.tolist()
    labels = df.activity_category.tolist()
    
    print(f"Total compounds: {len(combined_smiles)}")
    
    # Generate fingerprints in parallel
    print("Generating Morgan fingerprints...")
    with Pool(processes=ncpus) as pool:
        fingerprints = pool.map(smiles_to_morgan, combined_smiles, chunksize=100)
    
    fingerprints = np.array(fingerprints)
    print(f"Fingerprints shape: {fingerprints.shape}")
    
    # Calculate distance matrix with parallelization
    print(f"Calculating distance matrix using {ncpus} CPUs...")
    num_samples = len(fingerprints)
    distance_matrix = np.zeros((num_samples, num_samples))
    
    # Split work into chunks for parallel processing
    chunk_size = max(10, num_samples // (ncpus * 4))
    chunks = []
    
    for i in range(0, num_samples, chunk_size):
        end_i = min(i + chunk_size, num_samples)
        chunks.append((i, end_i, fingerprints[i:end_i], fingerprints))
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Process chunks in parallel
    with Pool(processes=ncpus) as pool:
        results = pool.map(calculate_distance_chunk, chunks)
    
    # Assemble distance matrix
    print("Assembling distance matrix...")
    for start_idx, chunk_distances in results:
        end_idx = start_idx + chunk_distances.shape[0]
        distance_matrix[start_idx:end_idx, :] = chunk_distances
    
    # Make symmetric
    distance_matrix = distance_matrix + distance_matrix.T
    
    print("Distance matrix complete")
    
    # Apply UMAP
    print("Running UMAP...")
    umap_model = umap.UMAP(metric='precomputed', n_components=2, random_state=42)
    umap_results = umap_model.fit_transform(distance_matrix)
    
    # Store results
    umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
    umap_df['set_label'] = labels
    
    # Save results
    folder = _umap_dir(base_path, receptor, threshold_dir, generator)
    os.makedirs(folder, exist_ok=True)
    
    output_file = f"{folder}/umap_results_{generator}_{type_scaffold}_{type_cluster}_{number}.csv"
    umap_df.to_csv(output_file, index=False)
    
    print(f"UMAP results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return umap_df


def process_umap_batch(
    generators: List[str],
    receptors: List[str],
    scaffolds: List[str] = ['csk'],
    splits: List[str] = ['dis', 'sim'],
    clusters: List[int] = [0, 1, 2, 3, 4],
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    ncpus: int = None,
    per_active_category_samples: int = DEFAULT_UMAP_ACTIVE_CATEGORY_SAMPLE,
    include_non_active: bool = True,
    non_active_samples: int = DEFAULT_UMAP_NON_ACTIVE_SAMPLE,
    use_stratified_sampling: bool = True,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
    ):
    """
    Process UMAP for multiple generators and receptors in batch.
    
    Args:
        generators: List of generator names
        receptors: List of receptor names
        scaffolds: List of scaffold types
        splits: List of split types
        clusters: List of cluster numbers
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        ncpus: Number of CPUs
        per_active_category_samples: Maximum number of output molecules per active category
        include_non_active: Whether to include sampled inactive output molecules
        non_active_samples: Maximum number of inactive output molecules
        use_stratified_sampling: If True, use the stratified UMAP workflow as the default canonical mode
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    print(f"\n{'='*70}")
    print(f"BATCH UMAP PROCESSING")
    print(f"{'='*70}")
    print(f"Generators: {len(generators)}")
    print(f"Receptors: {len(receptors)}")
    print(f"Total tasks: {len(generators) * len(receptors) * len(scaffolds) * len(splits) * len(clusters)}")
    print(f"Threshold directory: {build_threshold_dirname(user_threshold, dis_threshold, sim_threshold)}")
    print(f"{'='*70}\n")
    
    for receptor in receptors:
        for generator in generators:
            for scaffold in scaffolds:
                for split in splits:
                    for cluster in clusters:
                        try:
                            process_for_umap(
                                receptor=receptor,
                                type_scaffold=scaffold,
                                type_cluster=split,
                                number=cluster,
                                generator=generator,
                                user_threshold=user_threshold,
                                base_path=base_path,
                                ncpus=ncpus,
                                per_active_category_samples=per_active_category_samples,
                                include_non_active=include_non_active,
                                non_active_samples=non_active_samples,
                                use_stratified_sampling=use_stratified_sampling,
                                dis_threshold=dis_threshold,
                                sim_threshold=sim_threshold,
                            )
                        except Exception as e:
                            print(f"Error processing {generator}, {receptor}, {scaffold}, {split}, {cluster}: {e}")
                            continue
    
    print(f"\n{'='*70}")
    print(f"BATCH UMAP PROCESSING COMPLETE")
    print(f"{'='*70}\n")



def plot_umap_single(
    generator: str,
    type_scaffold: str,
    type_cluster: str,
    number: int,
    receptor: str,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    categories: List[str] = None,
    include_non_active: bool = False,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 300,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
):
    """
    Visualize UMAP results for a single cluster with customizable categories.
    
    Args:
        generator: Generator name
        type_scaffold: Scaffold type
        type_cluster: Cluster type ('dis' or 'sim')
        number: Cluster number
        receptor: Receptor name
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        categories: List of categories to plot (default: all)
                   Options: ['IS', 'RS', 'only_scaf', 'only_fp', 'both_active', 'non_active']
        include_non_active: If True and categories is None, include inactive compounds in the default view
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    threshold = resolve_threshold(type_cluster, user_threshold, dis_threshold, sim_threshold)
    threshold_dirs = resolve_threshold_dir_candidates(
        type_cluster,
        user_threshold,
        dis_threshold,
        sim_threshold,
    )
    threshold_dir = threshold_dirs[0]
    umap_file = None
    for candidate_dir in threshold_dirs:
        candidate_file = os.path.join(
            _umap_dir(base_path, receptor, candidate_dir, generator),
            f"umap_results_{generator}_{type_scaffold}_{type_cluster}_{number}.csv",
        )
        if os.path.exists(candidate_file):
            umap_file = candidate_file
            threshold_dir = candidate_dir
            break

    if umap_file is None:
        print(
            "UMAP file not found in any threshold directory: "
            + ", ".join(threshold_dirs)
        )
        return
    
    umap_df = pd.read_csv(umap_file)
    
    # Default categories
    if categories is None:
        categories = ['IS', 'RS', 'only_scaf', 'only_fp', 'both_active']
        if include_non_active:
            categories.append('non_active')
    elif include_non_active and 'non_active' not in categories:
        categories = list(categories) + ['non_active']
    
    # Filter data for selected categories
    umap_df = umap_df[umap_df['set_label'].isin(categories)]
    
    label_styles = UMAP_LABEL_STYLES
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot each category
    for label in categories:
        if label in umap_df['set_label'].unique():
            subset = umap_df[umap_df['set_label'] == label]
            style = label_styles.get(
                label,
                {'color': 'gray', 'marker': 'o', 'size': 50, 'alpha': 0.6, 'edgecolor': 'white', 'linewidth': 0.5},
            )
            
            ax.scatter(
                subset['UMAP1'], subset['UMAP2'],
                label=f"{label} (n={len(subset)})",
                color=style['color'],
                s=style['size'],
                marker=style['marker'],
                alpha=style['alpha'],
                edgecolors=style.get('edgecolor', 'white'),
                linewidths=style.get('linewidth', 0.5)
            )
    
    # Styling
    ax.set_xlabel('UMAP Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP Component 2', fontsize=14, fontweight='bold')
    ax.set_title(
        f'{receptor.replace("_", " ")} - {generator}\n'
        f'{type_scaffold.upper()} scaffold, {type_cluster}, cluster {number}, threshold {threshold}',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    
    # Save
    categories_str = '_'.join(categories)
    folder = _umap_dir(base_path, receptor, "img", "single_plot", threshold_dir)
    os.makedirs(folder, exist_ok=True)
    
    filename = f'umap_single_{generator}_{type_scaffold}_{type_cluster}_{number}_{categories_str}.png'
    plt.savefig(f'{folder}/{filename}', format='png', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {folder}/{filename}")
    plt.show()


def plot_umap_grid(
    generator: str,
    type_scaffold: str,
    type_cluster: str,
    receptor: str,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    clusters: List[int] = [0, 1, 2, 3, 4],
    categories: List[str] = None,
    include_non_active: bool = False,
    show_panels: List[str] = ['reference', 'activity', 'all'],
    figsize_per_row: int = 6,
    dpi: int = 300,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
):
    """
    Create grid visualization of UMAP results across multiple clusters.
    
    Args:
        generator: Generator name
        type_scaffold: Scaffold type
        type_cluster: Cluster type
        receptor: Receptor name
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        clusters: List of cluster numbers to plot
        categories: Categories to include (default: all)
        include_non_active: If True and categories is None, include inactive compounds in the combined view
        show_panels: Which panels to show ['reference', 'activity', 'all']
                    - 'reference': IS & RS only
                    - 'activity': Activity categories only
                    - 'all': All categories combined
        figsize_per_row: Height per row in inches
        dpi: Resolution for saved figure
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    threshold = resolve_threshold(type_cluster, user_threshold, dis_threshold, sim_threshold)
    threshold_dirs = resolve_threshold_dir_candidates(
        type_cluster,
        user_threshold,
        dis_threshold,
        sim_threshold,
    )
    threshold_dir = threshold_dirs[0]
    if categories is None:
        categories = ['IS', 'RS', 'only_scaf', 'only_fp', 'both_active']
        if include_non_active:
            categories.append('non_active')
    elif include_non_active and 'non_active' not in categories:
        categories = list(categories) + ['non_active']
    
    label_styles = UMAP_LABEL_STYLES
    
    # Load all data to get global axis limits
    all_data = []
    for number in clusters:
        loaded = False
        for candidate_dir in threshold_dirs:
            candidate_file = os.path.join(
                _umap_dir(base_path, receptor, candidate_dir, generator),
                f"umap_results_{generator}_{type_scaffold}_{type_cluster}_{number}.csv",
            )
            print(candidate_file)
            if os.path.exists(candidate_file):
                df = pd.read_csv(candidate_file)
                all_data.append(df)
                threshold_dir = candidate_dir
                loaded = True
                break
        if not loaded:
            print(f"Warning: UMAP file not found for cluster {number}")
    
    if not all_data:
        print("No UMAP data found!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    x_min, x_max = combined['UMAP1'].min(), combined['UMAP1'].max()
    y_min, y_max = combined['UMAP2'].min(), combined['UMAP2'].max()
    
    # Add padding
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    
    # Create grid
    n_panels = len(show_panels)
    n_rows = len(clusters)
    
    fig, axes = plt.subplots(
        n_rows, n_panels,
        figsize=(7 * n_panels, figsize_per_row * n_rows)
    )
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, number in enumerate(clusters):
        umap_df = None
        for candidate_dir in threshold_dirs:
            candidate_file = os.path.join(
                _umap_dir(base_path, receptor, candidate_dir, generator),
                f"umap_results_{generator}_{type_scaffold}_{type_cluster}_{number}.csv",
            )
            if os.path.exists(candidate_file):
                umap_df = pd.read_csv(candidate_file)
                threshold_dir = candidate_dir
                break
        if umap_df is None:
            continue
        
        panel_idx = 0
        
        # Panel 1: Reference sets (IS & RS)
        if 'reference' in show_panels:
            ax = axes[row_idx, panel_idx]
            for label in ['IS', 'RS']:
                if label in umap_df['set_label'].unique():
                    subset = umap_df[umap_df['set_label'] == label]
                    style = label_styles[label]
                    ax.scatter(subset['UMAP1'], subset['UMAP2'],
                              label=f"{label} (n={len(subset)})",
                              color=style['color'],
                              s=style['size'],
                              marker=style['marker'],
                              alpha=style['alpha'],
                              edgecolors=style.get('edgecolor', 'white'),
                              linewidths=style.get('linewidth', 0.5))
            
            ax.set_title(f'Cluster {number}: Reference Sets', fontsize=12, fontweight='bold')
            ax.set_xlabel('UMAP1', fontsize=10)
            ax.set_ylabel('UMAP2', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_facecolor('#f8f8f8')
            panel_idx += 1
        
        # Panel 2: Activity categories
        if 'activity' in show_panels:
            ax = axes[row_idx, panel_idx]
            for label in ['only_scaf', 'only_fp', 'both_active']:
                if label in umap_df['set_label'].unique() and label in categories:
                    subset = umap_df[umap_df['set_label'] == label]
                    style = label_styles[label]
                    ax.scatter(subset['UMAP1'], subset['UMAP2'],
                              label=f"{label} (n={len(subset)})",
                              color=style['color'],
                              s=style['size'],
                              marker=style['marker'],
                              alpha=style['alpha'],
                              edgecolors=style.get('edgecolor', 'white'),
                              linewidths=style.get('linewidth', 0.3))
            
            ax.set_title(f'Cluster {number}: Activity Categories', fontsize=12, fontweight='bold')
            ax.set_xlabel('UMAP1', fontsize=10)
            ax.set_ylabel('UMAP2', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_facecolor('#f8f8f8')
            panel_idx += 1
        
        # Panel 3: All categories
        if 'all' in show_panels:
            ax = axes[row_idx, panel_idx]
            for label in categories:
                if label in umap_df['set_label'].unique():
                    subset = umap_df[umap_df['set_label'] == label]
                    style = label_styles.get(
                        label,
                        {'color': 'gray', 'marker': 'o', 'size': 40, 'alpha': 0.5, 'edgecolor': 'white', 'linewidth': 0.3},
                    )
                    ax.scatter(subset['UMAP1'], subset['UMAP2'],
                              label=f"{label} (n={len(subset)})",
                              color=style['color'],
                              s=style['size'],
                              marker=style['marker'],
                              alpha=style['alpha'],
                              edgecolors=style.get('edgecolor', 'white'),
                              linewidths=style.get('linewidth', 0.3))
            
            ax.set_title(f'Cluster {number}: All Categories', fontsize=12, fontweight='bold')
            ax.set_xlabel('UMAP1', fontsize=10)
            ax.set_ylabel('UMAP2', fontsize=10)
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_facecolor('#f8f8f8')
    
    fig.suptitle(
        f'UMAP Grid: {receptor.replace("_", " ")} - {generator}\n'
        f'{type_scaffold.upper()} scaffold, {type_cluster} split, threshold {threshold}',
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    folder = _umap_dir(base_path, receptor, "img", "grid_plot", threshold_dir)
    os.makedirs(folder, exist_ok=True)
    
    panels_str = '_'.join(show_panels)
    filename = f'umap_grid_{generator}_{type_scaffold}_{type_cluster}_{panels_str}.png'
    plt.savefig(f'{folder}/{filename}', format='png', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {folder}/{filename}")
    plt.show()


# ==================== BATCH PROCESSING ====================

def run_full_analysis(
    generators: List[str],
    receptors: List[str],
    scaffolds: List[str] = ['csk', 'murcko'],
    splits: List[str] = ['dis', 'sim'],
    clusters: List[int] = [0, 1, 2, 3, 4],
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    ncpus: int = None,
    include_umap: bool = True,
    dis_threshold: float = DEFAULT_THRESHOLDS["dis"],
    sim_threshold: float = DEFAULT_THRESHOLDS["sim"],
):
    """
    Run complete analysis pipeline for multiple generators and receptors.
    
    Args:
        generators: List of generator names
        receptors: List of receptor names
        scaffolds: List of scaffold types
        splits: List of split types
        clusters: List of cluster numbers
        user_threshold: Optional shared similarity threshold override
        base_path: Base data path
        ncpus: Number of CPUs for parallel processing
        include_umap: Whether to generate UMAP visualizations
        dis_threshold: Threshold for dis split
        sim_threshold: Threshold for sim split
    """
    print(f"\n{'='*70}")
    print(f"STARTING FULL ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Generators: {', '.join(generators)}")
    print(f"Receptors: {', '.join(receptors)}")
    print(f"Thresholds: dis={dis_threshold}, sim={sim_threshold}, override={user_threshold}")
    print(f"CPUs: {ncpus or 'auto'}")
    print(f"Include UMAP: {include_umap}")
    print(f"{'='*70}\n")
    
    for receptor in receptors:
        for generator in generators:
            for scaffold in scaffolds:
                for split in splits:
                    for cluster in clusters:
                        print(f"\nProcessing: {generator}, {receptor}, {scaffold}, {split}, cluster {cluster}")
                        
                        # Step 1: Analyze single cluster
                        try:
                            analyze_generator_single_cluster(
                                generator, receptor, scaffold, split, cluster,
                                user_threshold, base_path, dis_threshold, sim_threshold
                            )
                        except Exception as e:
                            print(f"Error in analyze_generator_single_cluster: {e}")
                            continue
                        
                        # Step 2: Compare FP to Scaffold
                        try:
                            compare_fp_to_scaffold(
                                receptor, generator, scaffold, split, cluster,
                                user_threshold, base_path, ncpus, 50, dis_threshold, sim_threshold
                            )
                        except Exception as e:
                            print(f"Error in compare_fp_to_scaffold: {e}")
                        
                        # Step 3: Compare Scaffold to FP
                        try:
                            compare_scaffold_to_fp(
                                receptor, generator, scaffold, split, cluster,
                                user_threshold, base_path, ncpus, 50, dis_threshold, sim_threshold
                            )
                        except Exception as e:
                            print(f"Error in compare_scaffold_to_fp: {e}")
                        
                        # Step 4: UMAP (optional)
                        if include_umap:
                            try:
                                process_for_umap(
                                    receptor, scaffold, split, cluster, generator,
                                    user_threshold, base_path, ncpus, 50000, dis_threshold, sim_threshold
                                )
                            except Exception as e:
                                print(f"Error in process_for_umap: {e}")
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}\n")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    dis_threshold = 0.7
    sim_threshold = 0.8

    # Example 1: Analyze multiple generators and clusters in parallel
    print("="*70)
    print("EXAMPLE 1: Parallel analysis of multiple generators")
    print("="*70)
    
    results_df = analyze_multiple_clusters_parallel(
        generator_list=['DrugEx_GT_epsilon_0.1_250k', 'Molpher_250k', 'addcarbon_250k'],
        receptor='Glucocorticoid_receptor',
        scaffold='csk',
        splits=['dis', 'sim'],
        clusters=[0, 1, 2, 3, 4],
        user_threshold=None,
        dis_threshold=dis_threshold,
        sim_threshold=sim_threshold,
        ncpus=8  # Use 8 CPUs
    )
    
    # Example 2: Then summarize results
    print("\n" + "="*70)
    print("EXAMPLE 2: Summarizing results")
    print("="*70)
    
    avg_df, summary_df = summarize_results(
        generators_list=['DrugEx_GT_epsilon_0.1_250k', 'Molpher_250k', 'addcarbon_250k'],
        receptor='Glucocorticoid_receptor',
        scaffold='csk',
        splits=['dis', 'sim'],
        numbers=[0, 1, 2, 3, 4],
        user_threshold=None
    )
    
    # Example 3: Generate visualizations
    print("\n" + "="*70)
    print("EXAMPLE 3: Generating visualizations")
    print("="*70)
    
    plot_activity_barplots('Glucocorticoid_receptor', user_threshold=None)
    
    # Example 4: Complete workflow for single generator
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete workflow with comparisons")
    print("="*70)
    
    receptor = 'Glucocorticoid_receptor'
    generator = 'DrugEx_GT_epsilon_0.1_250k'
    scaffold = 'csk'
    
    # Step 1: Analyze all clusters (parallel)
    analyze_multiple_clusters_parallel(
        generator_list=[generator],
        receptor=receptor,
        scaffold=scaffold,
        user_threshold=None,
        dis_threshold=dis_threshold,
        sim_threshold=sim_threshold,
        ncpus=8
    )
    
    # Step 2: Summarize
    summarize_results(
        generators_list=[generator],
        receptor=receptor,
        scaffold=scaffold,
        splits=['dis', 'sim'],
        numbers=[0, 1, 2, 3, 4],
        user_threshold=None
    )
    
    # Step 3: Compare FP to Scaffold (for first cluster)
    compare_fp_to_scaffold(
        receptor, generator, scaffold, 'dis', 0,
        0.7, ncpus=8
    )
    
    # Step 4: Compare Scaffold to FP
    compare_scaffold_to_fp(
        receptor, generator, scaffold, 'dis', 0,
        0.7, ncpus=8
    )
    
    # Step 5: UMAP visualization
    process_for_umap(
        receptor, scaffold, 'dis', 0, generator,
        user_threshold=None,
        ncpus=8,
        dis_threshold=dis_threshold,
        sim_threshold=sim_threshold,
    )
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)

# ==================== METRIC-FAMILY COMPARISON ====================

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import path_utils


DEFAULT_PHARM_THRESHOLDS = {
    "dis": 0.7,
    "sim": 0.8,
}

METRICS = ("RS", "SED", "ASER")
SCAFFOLD_COMPARISON_MODES = ("csk", "murcko", "mean")


@dataclass(frozen=True)
class ComparisonResult:
    merged: pd.DataFrame
    summary: pd.DataFrame
    ranking: pd.DataFrame
    figure_path: Path | None


def _scaffold_results_dir(base_dir: Path, receptor: str, scaffold_type: str, split: str) -> Path:
    if hasattr(path_utils, "scaffold_results_dir"):
        return path_utils.scaffold_results_dir(base_dir, receptor, f"{scaffold_type}_scaffolds", split)
    return path_utils.data_subdir(
        base_dir,
        path_utils.SCAFFOLD_RESULTS_DIRNAME,
        receptor,
        f"{scaffold_type}_scaffolds",
        split,
    )


def _pharm_results_dir(base_dir: Path, receptor: str, phfp_type: str, split: str) -> Path:
    if hasattr(path_utils, "pharm_results_dir"):
        return path_utils.pharm_results_dir(base_dir, receptor, phfp_type, split)
    return path_utils.data_subdir(
        base_dir,
        path_utils.PHARM_RESULTS_DIRNAME,
        receptor,
        phfp_type,
        split,
    )


def _normalize_generator_name(name: str) -> str:
    return str(name).replace("_mean", "")


def _prettify_generator_label(name: str) -> str:
    return (
        name.replace("DrugEx_GT_", "DrugEx GT ")
        .replace("DrugEx_RNN_", "DrugEx RNN ")
        .replace("GB_GA_log_p_", "GB_GA logP ")
        .replace("GB_GA_", "GB_GA ")
        .replace("_epsilon_", "eps ")
        .replace("_mut_r_", "mut ")
        .replace("addcarbon", "AddCarbon")
        .replace("enamine", "Enamine")
        .replace("_", " ")
    )


def _normalize_generator_list(generators: Iterable[str] | None) -> list[str] | None:
    if generators is None:
        return None
    normalized = []
    for generator in generators:
        generator_name = _normalize_generator_name(str(generator).strip())
        if generator_name and generator_name not in normalized:
            normalized.append(generator_name)
    return normalized


def _filter_generators(df: pd.DataFrame, generators: Iterable[str] | None, context: str) -> pd.DataFrame:
    normalized_generators = _normalize_generator_list(generators)
    if not normalized_generators:
        return df

    filtered = df[df["generator"].isin(normalized_generators)].copy()
    if filtered.empty:
        available = sorted(df["generator"].dropna().unique().tolist())
        raise ValueError(
            f"No requested generators found in {context}. "
            f"Requested: {normalized_generators}. Available: {available}"
        )

    missing = sorted(set(normalized_generators) - set(filtered["generator"].unique()))
    if missing:
        available = sorted(df["generator"].dropna().unique().tolist())
        raise ValueError(
            f"Some requested generators are missing in {context}: {missing}. "
            f"Available: {available}"
        )
    return filtered


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"))


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    coef = _pearson_corr(x_rank, y_rank)
    try:
        from scipy.stats import spearmanr  # type: ignore

        _, pvalue = spearmanr(x, y)
        pvalue = float(pvalue)
    except Exception:
        pvalue = _estimate_spearman_pvalue(x_rank, y_rank, coef)
    return coef, pvalue


def _estimate_spearman_pvalue(
    x_rank: pd.Series,
    y_rank: pd.Series,
    observed_rho: float,
    n_permutations: int = 4000,
) -> float:
    valid = pd.concat([x_rank, y_rank], axis=1).dropna()
    if len(valid) < 3 or pd.isna(observed_rho):
        return float("nan")

    x_vals = valid.iloc[:, 0].to_numpy(dtype=float)
    y_vals = valid.iloc[:, 1].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    extreme = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(y_vals)
        permuted_rho = np.corrcoef(x_vals, permuted)[0, 1]
        if abs(permuted_rho) >= abs(observed_rho):
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)


def _load_scaffold_metrics(
    base_dir: Path,
    receptor: str,
    split: str,
    scaffold_type: str,
    generators: Iterable[str] | None = None,
) -> pd.DataFrame:
    root = _scaffold_results_dir(base_dir, receptor, scaffold_type, split)
    pattern = f"*_mean_{scaffold_type}_{split}.csv"
    files = sorted(root.glob(f"*/{pattern}"))
    if not files:
        raise FileNotFoundError(f"No scaffold mean files found in {root}")

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df["generator"] = df["name"].map(_normalize_generator_name)
        df["receptor"] = receptor
        df["split"] = split
        df["scaffold_type"] = scaffold_type
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    return _filter_generators(
        result,
        generators,
        f"scaffold metrics for receptor={receptor}, split={split}, scaffold_type={scaffold_type}",
    )


def _load_pharmacophore_metrics(
    base_dir: Path,
    receptor: str,
    split: str,
    threshold: float,
    phfp_type: str = "rdkit",
    generators: Iterable[str] | None = None,
) -> pd.DataFrame:
    threshold_str = f"{threshold:g}"
    root = _pharm_results_dir(base_dir, receptor, phfp_type, split)
    pattern = f"*_mean_{phfp_type}_{split}_threshold_{threshold_str}.csv"
    files = sorted(root.glob(f"*/threshold_{threshold_str}/{pattern}"))
    if not files:
        raise FileNotFoundError(
            f"No pharmacophore mean files found in {root} for threshold {threshold_str}"
        )

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df["generator"] = df["name"].map(_normalize_generator_name)
        df["receptor"] = receptor
        df["split"] = split
        df["threshold"] = threshold
        df["phfp_type"] = phfp_type
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    return _filter_generators(
        result,
        generators,
        f"pharmacophore metrics for receptor={receptor}, split={split}, threshold={threshold:g}",
    )


def _build_mean_scaffold_table(
    base_dir: Path,
    receptor: str,
    split: str,
    generators: Iterable[str] | None = None,
) -> pd.DataFrame:
    scaffold_frames = [
        _load_scaffold_metrics(
            base_dir,
            receptor,
            split,
            scaffold_type,
            generators=generators,
        )[["generator", *METRICS]]
        for scaffold_type in ("csk", "murcko")
    ]
    merged_scaffolds = scaffold_frames[0].merge(
        scaffold_frames[1],
        on="generator",
        how="inner",
        suffixes=("_csk", "_murcko"),
    )
    if merged_scaffolds.empty:
        raise ValueError(
            f"No overlapping generators between csk and murcko scaffold results for {receptor} {split}."
        )

    mean_df = merged_scaffolds[["generator"]].copy()
    for metric in METRICS:
        mean_df[metric] = merged_scaffolds[[f"{metric}_csk", f"{metric}_murcko"]].mean(axis=1)
    return mean_df


def _build_scaffold_reference_table(
    base_dir: Path,
    receptor: str,
    split: str,
    comparison_mode: str,
    generators: Iterable[str] | None = None,
) -> pd.DataFrame:
    if comparison_mode == "mean":
        scaffold_df = _build_mean_scaffold_table(base_dir, receptor, split, generators=generators)
    else:
        scaffold_df = _load_scaffold_metrics(
            base_dir,
            receptor,
            split,
            comparison_mode,
            generators=generators,
        )[
            ["generator", *METRICS]
        ]
    scaffold_df["comparison_mode"] = comparison_mode
    return scaffold_df


def load_comparison_table(
    receptor: str,
    split: str,
    comparison_mode: str,
    threshold: float | None = None,
    phfp_type: str = "rdkit",
    base_dir: str | Path = "../..",
    generators: Iterable[str] | None = None,
) -> pd.DataFrame:
    if comparison_mode not in SCAFFOLD_COMPARISON_MODES:
        raise ValueError(
            f"Unsupported comparison_mode '{comparison_mode}'. "
            f"Use one of {SCAFFOLD_COMPARISON_MODES}."
        )

    base_path = Path(base_dir).resolve()
    threshold = DEFAULT_PHARM_THRESHOLDS[split] if threshold is None else threshold

    normalized_generators = _normalize_generator_list(generators)
    scaffold_df = _build_scaffold_reference_table(
        base_path,
        receptor,
        split,
        comparison_mode,
        generators=normalized_generators,
    )
    pharmacophore_df = _load_pharmacophore_metrics(
        base_path,
        receptor,
        split,
        threshold,
        phfp_type,
        generators=normalized_generators,
    )

    merged = scaffold_df[["generator", *METRICS]].merge(
        pharmacophore_df[["generator", *METRICS]],
        on="generator",
        how="inner",
        suffixes=("_scaffold", "_pharmacophore"),
    )
    if merged.empty:
        raise ValueError(
            "No overlapping generators between scaffold and pharmacophore results "
            f"for receptor={receptor}, split={split}, comparison_mode={comparison_mode}."
        )

    merged = merged.sort_values("generator").reset_index(drop=True)
    merged["receptor"] = receptor
    merged["split"] = split
    merged["comparison_mode"] = comparison_mode
    merged["phfp_type"] = phfp_type
    merged["threshold"] = threshold
    return merged


def compute_correlation_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        pearson_r = _pearson_corr(
            merged[f"{metric}_scaffold"],
            merged[f"{metric}_pharmacophore"],
        )
        rho_value, p_value = _safe_spearman(
            merged[f"{metric}_scaffold"],
            merged[f"{metric}_pharmacophore"],
        )
        rank_scaffold = merged[f"{metric}_scaffold"].rank(ascending=False, method="average")
        rank_pharm = merged[f"{metric}_pharmacophore"].rank(ascending=False, method="average")
        rho_rank, p_rank = _safe_spearman(rank_scaffold, rank_pharm)

        rows.append(
            {
                "metric": metric,
                "pearson_r": pearson_r,
                "spearman_rho": rho_value,
                "spearman_pvalue": p_value,
                "rank_spearman_rho": rho_rank,
                "rank_spearman_pvalue": p_rank,
                "n_generators": len(merged),
            }
        )
    return pd.DataFrame(rows)


def compute_ranking_table(merged: pd.DataFrame) -> pd.DataFrame:
    ranking = merged[["generator"]].copy()
    for metric in METRICS:
        ranking[f"{metric}_scaffold_rank"] = merged[f"{metric}_scaffold"].rank(
            ascending=False,
            method="min",
        )
        ranking[f"{metric}_pharmacophore_rank"] = merged[f"{metric}_pharmacophore"].rank(
            ascending=False,
            method="min",
        )
        ranking[f"delta_{metric}"] = (
            merged[f"{metric}_pharmacophore"] - merged[f"{metric}_scaffold"]
        )
    return ranking.sort_values("generator").reset_index(drop=True)


def _build_generator_color_map(generators: Iterable[str]) -> dict[str, tuple[float, float, float, float]]:
    generators = sorted(set(generators))
    fallback_palette = [
        (0.141, 0.353, 0.545, 1.0),
        (0.756, 0.294, 0.208, 1.0),
        (0.231, 0.557, 0.302, 1.0),
        (0.635, 0.318, 0.580, 1.0),
        (0.858, 0.533, 0.172, 1.0),
        (0.302, 0.451, 0.698, 1.0),
        (0.780, 0.467, 0.698, 1.0),
        (0.400, 0.400, 0.400, 1.0),
        (0.550, 0.620, 0.173, 1.0),
        (0.200, 0.600, 0.600, 1.0),
    ]
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("Set2", max(len(generators), 1))
        return {generator: cmap(index) for index, generator in enumerate(generators)}
    except ModuleNotFoundError:
        return {
            generator: fallback_palette[index % len(fallback_palette)]
            for index, generator in enumerate(generators)
        }


def plot_metric_correlations(
    merged: pd.DataFrame,
    output_path: str | Path | None = None,
    color_map: dict[str, tuple[float, float, float, float]] | None = None,
    title_prefix: str | None = None,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required only for plotting. Install it in the notebook environment."
        ) from exc

    if color_map is None:
        color_map = _build_generator_color_map(merged["generator"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    summary = compute_correlation_summary(merged).set_index("metric")

    for ax, metric in zip(axes, METRICS):
        x = merged[f"{metric}_scaffold"]
        y = merged[f"{metric}_pharmacophore"]

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        pad = (max_val - min_val) * 0.08 if max_val > min_val else 0.05
        ax.plot(
            [min_val - pad, max_val + pad],
            [min_val - pad, max_val + pad],
            linestyle="--",
            linewidth=1.2,
            color="#7a7a7a",
            zorder=1,
        )

        for _, row in merged.iterrows():
            ax.scatter(
                row[f"{metric}_scaffold"],
                row[f"{metric}_pharmacophore"],
                s=80,
                color=color_map[row["generator"]],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.95,
                zorder=3,
            )

        rho = summary.loc[metric, "spearman_rho"]
        pvalue = summary.loc[metric, "spearman_pvalue"]
        p_text = f"{pvalue:.3g}" if pd.notna(pvalue) else "NA"
        ax.set_title(f"{metric}\nSpearman rho = {rho:.3f}, p = {p_text}", fontsize=11)
        mode_label = merged["comparison_mode"].iat[0]
        scaffold_label = "mean(CSK, Murcko)" if mode_label == "mean" else mode_label.upper()
        ax.set_xlabel(f"{metric} scaffold metric ({scaffold_label})", fontsize=10)
        ax.set_ylabel(f"{metric} RDKit pharmacophore", fontsize=10)
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_ylim(min_val - pad, max_val + pad)
        ax.grid(alpha=0.22, linewidth=0.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[generator],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7,
            label=_prettify_generator_label(generator),
        )
        for generator in sorted(color_map)
        if generator in set(merged["generator"])
    ]

    receptor = merged["receptor"].iat[0]
    split = merged["split"].iat[0]
    comparison_mode = merged["comparison_mode"].iat[0]
    threshold = merged["threshold"].iat[0]
    mode_label = "mean(CSK, Murcko)" if comparison_mode == "mean" else comparison_mode
    title = (
        f"{title_prefix}\n" if title_prefix else ""
    ) + f"{receptor} | {split} | {mode_label} vs RDKit pharmacophore (threshold {threshold:g})"
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(5, max(1, len(legend_handles))),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    if output_path is None:
        plt.show()
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overview_heatmap(
    summary_df: pd.DataFrame,
    output_path: str | Path | None = None,
    value_col: str = "spearman_rho",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import Normalize
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.cm import ScalarMappable
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and seaborn are required only for plotting. "
            "Install them in the notebook environment."
        ) from exc

    df = summary_df.copy()
    receptor_order = ["Glucocorticoid_receptor", "Leukocyte_elastase"]
    split_order = ["dis", "sim"]
    receptor_labels = {
        "Glucocorticoid_receptor": "Glucocorticoid receptor",
        "Leukocyte_elastase": "Leukocyte elastase",
    }
    mode_order = ["csk", "murcko", "mean"]
    mode_labels = {
        "csk": "CSK",
        "murcko": "MURCKO",
        "mean": "Mean",
    }
    metric_base_colors = {
        "RS": "#e97b32",
        "SED": "#97C2F0",
        "ASER": "#71ad48",
    }
    metric_cmaps = {
        metric: LinearSegmentedColormap.from_list(
            f"{metric}_custom",
            ["#fcfcfb", base_color],
        )
        for metric, base_color in metric_base_colors.items()
    }

    fig = plt.figure(figsize=(15.0, 7.2))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.90,
        top=1,
        bottom=0.14,
        wspace=0.10,
        hspace=0.17,
    )

    for row_idx, receptor in enumerate(receptor_order):
        for col_idx, split in enumerate(split_order):
            panel_spec = outer[row_idx, col_idx].subgridspec(1, 3, wspace=0.02)
            panel_df = df[(df["receptor"] == receptor) & (df["split"] == split)].copy()

            for metric_idx, metric in enumerate(METRICS):
                ax = fig.add_subplot(panel_spec[0, metric_idx])
                metric_df = panel_df[panel_df["metric"] == metric].copy()
                heatmap_rows = []
                for row_metric, row_label in (("spearman_rho", "Spearman rho"), ("pearson_r", "Pearson r")):
                    row_df = metric_df.set_index("comparison_mode")[row_metric].reindex(mode_order)
                    heatmap_rows.append(row_df.rename(row_label))
                pivot = pd.DataFrame(heatmap_rows)
                pivot.columns = [mode_labels[col] for col in pivot.columns]

                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".2f",
                    cmap=metric_cmaps[metric],
                    vmin=0,
                    vmax=1,
                    linewidths=1.0,
                    linecolor="white",
                    cbar=False,
                    ax=ax,
                    annot_kws={"fontsize": 11, "fontweight": "semibold", "color": "black"},
                )

                if row_idx == 0:
                    ax.set_title(metric, fontsize=12, pad=3, fontweight="bold", y=1.00)
                else:
                    ax.set_title("")

                if metric_idx == 0:
                    if col_idx == 0:
                        ax.set_ylabel("Pharmacophore-based metrics", fontsize=11)
                        ax.set_yticklabels(["Spearman rho", "Pearson r"], rotation=0, fontsize=10)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", length=0)
                else:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", length=0)

                ax.set_xlabel("")
                if row_idx == len(receptor_order) - 1:
                    ax.tick_params(axis="x", rotation=0, labelsize=10, pad=2)
                else:
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", length=0)

                if metric_idx == 1:
                    if receptor == 'Leukocyte_elastase':
                        ax.text(
                            0.5,
                            1.06,
                            f"{receptor_labels[receptor]} | {split} split",
                            ha="center",
                            va="center",
                            fontsize=12,
                            transform=ax.transAxes,
                        )
                    else:
                        ax.text(
                            0.5,
                            1.15,
                            f"{receptor_labels[receptor]} | {split} split",
                            ha="center",
                            va="center",
                            fontsize=12,
                            transform=ax.transAxes,
                        )

    fig.text(0.27, 0.07, "Scaffold-based metrics", ha="center", va="center", fontsize=11)
    fig.text(0.71, 0.07, "Scaffold-based metrics", ha="center", va="center", fontsize=11)

    cbar_positions = {
        "RS": [0.915, 0.68, 0.012, 0.14],
        "SED": [0.915, 0.45, 0.012, 0.14],
        "ASER": [0.915, 0.22, 0.012, 0.14],
    }
    for metric in METRICS:
        cax = fig.add_axes(cbar_positions[metric])
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=metric_cmaps[metric])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(f"{metric} correlation", fontsize=10)

    if output_path is None:
        plt.show()
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_comparison(
    receptor: str,
    split: str,
    comparison_mode: str,
    threshold: float | None = None,
    phfp_type: str = "rdkit",
    base_dir: str | Path = "../..",
    output_dir: str | Path | None = None,
    color_map: dict[str, tuple[float, float, float, float]] | None = None,
    generators: Iterable[str] | None = None,
) -> ComparisonResult:
    merged = load_comparison_table(
        receptor=receptor,
        split=split,
        comparison_mode=comparison_mode,
        threshold=threshold,
        phfp_type=phfp_type,
        base_dir=base_dir,
        generators=generators,
    )
    summary = compute_correlation_summary(merged)
    ranking = compute_ranking_table(merged)

    figure_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path / "merged_metrics.csv", index=False)
        summary.to_csv(output_path / "correlation_summary.csv", index=False)
        ranking.to_csv(output_path / "ranking_comparison.csv", index=False)
        try:
            figure_path = plot_metric_correlations(
                merged,
                output_path=output_path / "metric_correlations.png",
                color_map=color_map,
            )
        except ModuleNotFoundError:
            figure_path = None

    return ComparisonResult(
        merged=merged,
        summary=summary,
        ranking=ranking,
        figure_path=figure_path,
    )


def run_all_comparisons(
    receptors: Iterable[str] = ("Leukocyte_elastase", "Glucocorticoid_receptor"),
    splits: Iterable[str] = ("dis", "sim"),
    comparison_modes: Iterable[str] = SCAFFOLD_COMPARISON_MODES,
    base_dir: str | Path = "../..",
    phfp_type: str = "rdkit",
    output_root: str | Path | None = None,
    generators: Iterable[str] | None = None,
) -> pd.DataFrame:
    base_path = Path(base_dir).resolve()
    if output_root is None:
        output_root_path = Path(_data_root(base_path)) / "comparison_outputs" / "metric_correlations"
    else:
        output_root_path = Path(output_root).expanduser()

    all_generators: set[str] = set()
    all_merged = []
    for receptor in receptors:
        for split in splits:
            for comparison_mode in comparison_modes:
                merged = load_comparison_table(
                    receptor=receptor,
                    split=split,
                    comparison_mode=comparison_mode,
                    phfp_type=phfp_type,
                    base_dir=base_path,
                    generators=generators,
                )
                all_generators.update(merged["generator"].tolist())
                all_merged.append(merged)

    color_map = _build_generator_color_map(all_generators)

    all_summaries = []
    for merged in all_merged:
        receptor = merged["receptor"].iat[0]
        split = merged["split"].iat[0]
        comparison_mode = merged["comparison_mode"].iat[0]
        threshold = merged["threshold"].iat[0]
        output_dir = (
            output_root_path
            / receptor
            / split
            / comparison_mode
            / f"threshold_{threshold:g}"
        )
        result = run_comparison(
            receptor=receptor,
            split=split,
            comparison_mode=comparison_mode,
            threshold=threshold,
            phfp_type=phfp_type,
            base_dir=base_path,
            output_dir=output_dir,
            color_map=color_map,
            generators=generators,
        )
        summary = result.summary.copy()
        summary["receptor"] = receptor
        summary["split"] = split
        summary["comparison_mode"] = comparison_mode
        summary["phfp_type"] = phfp_type
        summary["threshold"] = threshold
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    output_root_path.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_root_path / "all_correlation_summaries.csv", index=False)
    try:
        plot_overview_heatmap(
            combined,
            output_path=output_root_path / "correlation_overview_heatmap.png",
        )
        plot_overview_heatmap(
            combined,
            output_path=output_root_path / "correlation_overview_heatmap.svg",
        )
    except ModuleNotFoundError:
        pass
    return combined


def comparison_cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scaffold-based and RDKit pharmacophore-based metrics."
    )
    parser.add_argument("--receptor", choices=["Leukocyte_elastase", "Glucocorticoid_receptor"])
    parser.add_argument("--split", choices=["dis", "sim"])
    parser.add_argument("--comparison_mode", choices=list(SCAFFOLD_COMPARISON_MODES))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--phfp_type", default="rdkit")
    parser.add_argument("--base_dir", default="../..")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=None,
        help="Explicit generator names to include. Only these generators will be used.",
    )
    args = parser.parse_args()

    if args.run_all:
        combined = run_all_comparisons(
            base_dir=args.base_dir,
            phfp_type=args.phfp_type,
            output_root=args.output_root,
            generators=args.generators,
        )
        print(combined)
        return

    if not all([args.receptor, args.split, args.comparison_mode]):
        parser.error("--receptor, --split and --comparison_mode are required unless --run_all is used.")

    result = run_comparison(
        receptor=args.receptor,
        split=args.split,
        comparison_mode=args.comparison_mode,
        threshold=args.threshold,
        phfp_type=args.phfp_type,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        generators=args.generators,
    )
    print(result.summary)

# ==================== CROSS-MODALITY MISS ANALYSIS ====================

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


MODE_SPECS = {
    "fp_to_scaf": {
        "subdir": "compare_results_fp_to_scaf",
        "pattern": re.compile(
            r"compare_res_fp_to_active_scaf_(?P<generator>.+)_(?P<scaffold_type>csk|murcko)_(?P<split>dis|sim)_(?P<cluster>\d+)\.csv$"
        ),
        "query_category": "only_fp",
        "target_modality": "scaffold",
    },
    "scaf_to_fp": {
        "subdir": "compare_results_scaf_to_fp",
        "pattern": re.compile(
            r"compare_res_scaf_to_active_fp_(?P<generator>.+)_(?P<scaffold_type>csk|murcko)_(?P<split>dis|sim)_(?P<cluster>\d+)\.csv$"
        ),
        "query_category": "only_scaf",
        "target_modality": "pharmacophore",
    },
}

MODE_DISPLAY_NAMES = {
    "fp_to_scaf": "pharm-only -> scaffold",
    "scaf_to_fp": "scaffold-only -> pharm",
}


def _prettify_generator_name(name: str) -> str:
    return (
        name.replace("DrugEx_GT_", "DrugEx GT ")
        .replace("DrugEx_RNN_", "DrugEx RNN ")
        .replace("GB_GA_log_p_", "GB_GA logP ")
        .replace("GB_GA_", "GB_GA ")
        .replace("_epsilon_", " eps ")
        .replace("_mut_r_", " mut ")
        .replace("addcarbon", "AddCarbon")
    )


def _extract_threshold(path: Path) -> str:
    for part in path.parts:
        if part.startswith("threshold_"):
            return part.replace("threshold_", "")
    return "unknown"


def _drop_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    image_cols = [col for col in df.columns if col.endswith("_image")]
    return df.drop(columns=image_cols, errors="ignore")


def load_cross_modality_results(
    base_dir: str | Path,
    receptors: Iterable[str] = ("Glucocorticoid_receptor", "Leukocyte_elastase"),
) -> pd.DataFrame:
    base_path = Path(base_dir).resolve()
    all_rows = []

    for receptor in receptors:
        receptor_root = base_path / receptor
        if not receptor_root.exists():
            continue

        for mode, spec in MODE_SPECS.items():
            mode_root = receptor_root / spec["subdir"]
            if not mode_root.exists():
                continue

            for csv_path in sorted(mode_root.rglob("*.csv")):
                match = spec["pattern"].match(csv_path.name)
                if not match:
                    continue

                df = pd.read_csv(csv_path)
                df = _drop_image_columns(df)
                df["receptor"] = receptor
                df["mode"] = mode
                df["query_category"] = spec["query_category"]
                df["target_modality"] = spec["target_modality"]
                df["threshold_label"] = _extract_threshold(csv_path)
                df["generator"] = match.group("generator")
                df["generator_label"] = _prettify_generator_name(match.group("generator"))
                df["scaffold_type"] = match.group("scaffold_type")
                df["split"] = match.group("split")
                df["cluster"] = int(match.group("cluster"))
                df["source_file"] = str(csv_path)

                if "tanimoto_similarity" not in df.columns:
                    continue

                all_rows.append(df)

    if not all_rows:
        raise FileNotFoundError(f"No cross-modality result CSVs found under {base_path}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined["threshold_value"] = pd.to_numeric(combined["threshold_label"], errors="coerce")
    return combined


def summarize_cross_modality_results(results_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results_df.groupby(
            [
                "receptor",
                "threshold_label",
                "threshold_value",
                "mode",
                "query_category",
                "target_modality",
                "generator",
                "generator_label",
                "scaffold_type",
                "split",
                "cluster",
            ],
            dropna=False,
        )
        .agg(
            n_compounds=("smiles", "size"),
            mean_similarity=("tanimoto_similarity", "mean"),
            median_similarity=("tanimoto_similarity", "median"),
            std_similarity=("tanimoto_similarity", "std"),
            q25_similarity=("tanimoto_similarity", lambda x: x.quantile(0.25)),
            q75_similarity=("tanimoto_similarity", lambda x: x.quantile(0.75)),
            max_similarity=("tanimoto_similarity", "max"),
            min_similarity=("tanimoto_similarity", "min"),
        )
        .reset_index()
    )
    grouped["std_similarity"] = grouped["std_similarity"].fillna(0.0)
    return grouped


def summarize_across_clusters(cluster_summary_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cluster_summary_df.groupby(
            [
                "receptor",
                "threshold_label",
                "threshold_value",
                "mode",
                "query_category",
                "target_modality",
                "generator",
                "generator_label",
                "scaffold_type",
                "split",
            ],
            dropna=False,
        )
        .agg(
            n_clusters=("cluster", "nunique"),
            total_n_compounds=("n_compounds", "sum"),
            mean_of_mean_similarity=("mean_similarity", "mean"),
            std_of_mean_similarity=("mean_similarity", "std"),
            mean_of_median_similarity=("median_similarity", "mean"),
            mean_q25_similarity=("q25_similarity", "mean"),
            mean_q75_similarity=("q75_similarity", "mean"),
        )
        .reset_index()
    )
    summary["std_of_mean_similarity"] = summary["std_of_mean_similarity"].fillna(0.0)
    return summary


def select_example_rows(
    results_df: pd.DataFrame,
    n_best: int = 8,
    n_worst: int = 8,
) -> pd.DataFrame:
    example_rows = []
    group_cols = ["receptor", "threshold_label", "mode", "generator", "scaffold_type", "split"]
    for _, group in results_df.groupby(group_cols):
        ordered = group.sort_values("tanimoto_similarity", ascending=False)
        example_rows.append(ordered.head(n_best).assign(example_rank="best"))
        example_rows.append(ordered.tail(n_worst).assign(example_rank="worst"))
    return pd.concat(example_rows, ignore_index=True)



def plot_similarity_distributions(
    results_df: pd.DataFrame,
    output_path: str | Path,
    receptor: str,
    threshold_label: str,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Patch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib and seaborn are required for plotting.") from exc

    plot_df = results_df[
        (results_df["receptor"] == receptor) & (results_df["threshold_label"] == threshold_label)
    ].copy()
    if plot_df.empty:
        raise ValueError(f"No rows for receptor={receptor}, threshold={threshold_label}")
    plot_df["mode_display"] = plot_df["mode"].map(MODE_DISPLAY_NAMES).fillna(plot_df["mode"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True, sharey=True)
    palette = {"fp_to_scaf": "#4f86c6", "scaf_to_fp": "#e97b32"}
    display_palette = {
        MODE_DISPLAY_NAMES["fp_to_scaf"]: palette["fp_to_scaf"],
        MODE_DISPLAY_NAMES["scaf_to_fp"]: palette["scaf_to_fp"],
    }

    for ax, split in zip(axes, ["dis", "sim"]):
        split_df = plot_df[plot_df["split"] == split].copy()
        split_df = split_df.dropna(subset=["tanimoto_similarity", "mode"])

        if split_df.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{split.upper()} split")
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.2)
            continue

        mode_count = split_df["mode_display"].nunique()
        if mode_count >= 2:
            sns.kdeplot(
                data=split_df,
                x="tanimoto_similarity",
                hue="mode_display",
                fill=True,
                common_norm=False,
                alpha=0.35,
                palette=display_palette,
                ax=ax,
                warn_singular=False,
            )
        else:
            mode_value = split_df["mode_display"].iloc[0]
            sns.kdeplot(
                data=split_df,
                x="tanimoto_similarity",
                fill=True,
                alpha=0.35,
                color=display_palette.get(mode_value, "#7a7a7a"),
                ax=ax,
                warn_singular=False,
            )
        ax.set_title(f"{split.upper()} split")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Tanimoto similarity")
        ax.grid(alpha=0.2)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    legend_handles = [
        Patch(
            facecolor=display_palette[label],
            edgecolor=display_palette[label],
            alpha=0.35,
            label=label,
        )
        for label in [
            MODE_DISPLAY_NAMES["fp_to_scaf"],
            MODE_DISPLAY_NAMES["scaf_to_fp"],
        ]
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.945, 0.98),
        frameon=False,
        fontsize=10,
    )
    if receptor == 'Glucocorticoid_receptor':
        receptor_str = 'Glucocorticoid receptor'
    elif receptor == 'Leukocyte_elastase':
        receptor_str = 'Leukocyte elastase'
    else:
        receptor_str = receptor

    fig.suptitle(receptor_str, fontsize=14, y=0.94)
    fig.tight_layout(rect=[0, 0, 0.96, 0.92])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path



def run_cross_modality_miss_analysis(
    input_root: str | Path,
    output_root: str | Path,
    receptors: Iterable[str] = ("Glucocorticoid_receptor", "Leukocyte_elastase"),
) -> dict[str, pd.DataFrame]:
    input_path = Path(input_root).resolve()
    output_path = Path(output_root).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = load_cross_modality_results(input_path, receptors=receptors)
    cluster_summary_df = summarize_cross_modality_results(raw_df)
    overall_summary_df = summarize_across_clusters(cluster_summary_df)
    examples_df = select_example_rows(raw_df)

    raw_df.to_csv(output_path / "cross_modality_miss_raw.csv", index=False)
    cluster_summary_df.to_csv(output_path / "cross_modality_miss_cluster_summary.csv", index=False)
    overall_summary_df.to_csv(output_path / "cross_modality_miss_overall_summary.csv", index=False)
    examples_df.to_csv(output_path / "cross_modality_miss_examples.csv", index=False)

    threshold_labels = [label for label in overall_summary_df["threshold_label"].dropna().unique()]
    for receptor in receptors:
        for threshold_label in threshold_labels:
            receptor_slug = receptor.replace(" ", "_")
            safe_threshold = str(threshold_label).replace(".", "p")


            plot_similarity_distributions(
                    raw_df,
                    output_path=output_path / f"{receptor_slug}_threshold_{safe_threshold}_distribution.png",
                    receptor=receptor,
                    threshold_label=threshold_label,
                )
            plot_similarity_distributions(
                    raw_df,
                    output_path=output_path / f"{receptor_slug}_threshold_{safe_threshold}_distribution.svg",
                    receptor=receptor,
                    threshold_label=threshold_label,
                )

    return {
        "raw_df": raw_df,
        "cluster_summary_df": cluster_summary_df,
        "overall_summary_df": overall_summary_df,
        "examples_df": examples_df,
    }


def cross_modality_cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cross-modality misses using existing compare_results_fp_to_scaf and "
            "compare_results_scaf_to_fp outputs."
        )
    )
    parser.add_argument(
        "--input_root",
        default="../../data/comparison_outputs/overlap",
        help="Root directory with comparison overlap outputs.",
    )
    parser.add_argument(
        "--output_root",
        default="../../data/comparison_outputs/cross_modality",
        help="Directory where new summaries and plots will be saved.",
    )
    parser.add_argument(
        "--receptors",
        nargs="+",
        default=["Glucocorticoid_receptor", "Leukocyte_elastase"],
    )
    args = parser.parse_args()

    outputs = run_cross_modality_miss_analysis(
        input_root=args.input_root,
        output_root=args.output_root,
        receptors=args.receptors,
    )
    print(outputs["overall_summary_df"].head())

# ==================== PRINT-ORIENTED UMAP HELPERS ====================

UMAP_LABEL_STYLES = {
    'IS': {'color': '#808080', 'marker': 'o', 'size': 120, 'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 0.4},
    'RS': {'color': '#000000', 'marker': 'o', 'size': 120, 'alpha': 0.8, 'edgecolor': 'white', 'linewidth': 0.4},
    'only_scaf': {'color': '#ff7f0e', 'marker': '^', 'size': 100, 'alpha': 0.7, 'edgecolor': 'white', 'linewidth': 0.3},
    'only_fp': {'color': '#2ca02c', 'marker': 'D', 'size': 100, 'alpha': 0.7, 'edgecolor': 'white', 'linewidth': 0.3},
    'both_active': {'color': '#1f77b4', 'marker': 's', 'size': 100, 'alpha': 0.7, 'edgecolor': 'white', 'linewidth': 0.4},
    'non_active': {'color': '#d62728', 'marker': 'x', 'size': 80, 'alpha': 0.6, 'edgecolor': 'white', 'linewidth': 0.4},
}

UMAP_LABEL_DISPLAY_NAMES = {
    'IS': 'IS',
    'RS': 'RS',
    'only_scaf': 'scaffold-only',
    'only_fp': 'pharm-only',
    'both_active': 'both-active',
    'non_active': 'inactive',
}

UMAP_VIEW_CATEGORIES = {
    'reference': ['IS', 'RS'],
    'activity': ['only_scaf', 'only_fp', 'both_active'],
    'context': ['IS', 'RS', 'only_scaf', 'only_fp', 'both_active', 'non_active'],
}


def _load_umap_results_for_print(
    generator: str,
    type_scaffold: str,
    type_cluster: str,
    number: int,
    receptor: str,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    dis_threshold: float = DEFAULT_THRESHOLDS['dis'],
    sim_threshold: float = DEFAULT_THRESHOLDS['sim'],
) -> tuple[pd.DataFrame, str, float]:
    threshold = resolve_threshold(type_cluster, user_threshold, dis_threshold, sim_threshold)
    threshold_dirs = resolve_threshold_dir_candidates(
        type_cluster,
        user_threshold,
        dis_threshold,
        sim_threshold,
    )

    for candidate_dir in threshold_dirs:
        candidate_file = os.path.join(
            _umap_dir(base_path, receptor, candidate_dir, generator),
            f'umap_results_{generator}_{type_scaffold}_{type_cluster}_{number}.csv',
        )
        if os.path.exists(candidate_file):
            return pd.read_csv(candidate_file), candidate_dir, threshold

    raise FileNotFoundError(
        'UMAP file not found in any threshold directory: ' + ', '.join(threshold_dirs)
    )


def _plot_umap_points(
    ax,
    umap_df: pd.DataFrame,
    categories: Iterable[str],
    legend_fontsize: int = 9,
    title: str | None = None,
    title_fontsize: int = 12,
    axis_label_fontsize: int = 10,
    tick_label_fontsize: int = 10,
):
    selected = [label for label in categories if label in set(umap_df['set_label'])]
    for label in selected:
        subset = umap_df[umap_df['set_label'] == label]
        style = UMAP_LABEL_STYLES.get(
            label,
            {'color': 'gray', 'marker': 'o', 'size': 40, 'alpha': 0.5, 'edgecolor': 'white', 'linewidth': 0.3},
        )
        display_name = UMAP_LABEL_DISPLAY_NAMES.get(label, label)
        ax.scatter(
            subset['UMAP1'],
            subset['UMAP2'],
            label=f"{display_name} (n={len(subset)})",
            color=style['color'],
            s=style['size'],
            marker=style['marker'],
            alpha=style['alpha'],
            edgecolors=style.get('edgecolor', 'white'),
            linewidths=style.get('linewidth', 0.3),
        )

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel('UMAP1', fontsize=axis_label_fontsize)
    ax.set_ylabel('UMAP2', fontsize=axis_label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f8f8')
    if selected:
        ax.legend(fontsize=legend_fontsize, loc='best')


def plot_umap_grid_print(
    generator: str,
    type_scaffold: str,
    type_cluster: str,
    receptor: str,
    user_threshold: Optional[float] = None,
    base_path: str = '../data',
    clusters: List[int] = [0, 1, 2, 3, 4],
    categories: List[str] = None,
    include_non_active: bool = True,
    title: str | None = None,
    figsize_per_row: float = 9,
    dpi: int = 300,
    dis_threshold: float = DEFAULT_THRESHOLDS['dis'],
    sim_threshold: float = DEFAULT_THRESHOLDS['sim'],
):
    threshold = resolve_threshold(type_cluster, user_threshold, dis_threshold, sim_threshold)
    threshold_dirs = resolve_threshold_dir_candidates(
        type_cluster,
        user_threshold,
        dis_threshold,
        sim_threshold,
    )
    threshold_dir = threshold_dirs[0]
    if categories is None:
        categories = ['IS', 'RS', 'only_scaf', 'only_fp', 'both_active']
        if include_non_active:
            categories.append('non_active')
    elif include_non_active and 'non_active' not in categories:
        categories = list(categories) + ['non_active']

    all_data = []
    for number in clusters:
        loaded = False
        for candidate_dir in threshold_dirs:
            candidate_file = os.path.join(
                _umap_dir(base_path, receptor, candidate_dir, generator),
                f'umap_results_{generator}_{type_scaffold}_{type_cluster}_{number}.csv',
            )
            if os.path.exists(candidate_file):
                df = pd.read_csv(candidate_file)
                all_data.append(df)
                threshold_dir = candidate_dir
                loaded = True
                break
        if not loaded:
            print(f'Warning: UMAP file not found for cluster {number}')

    if not all_data:
        print('No UMAP data found!')
        return

    combined = pd.concat(all_data, ignore_index=True)
    x_min, x_max = combined['UMAP1'].min(), combined['UMAP1'].max()
    y_min, y_max = combined['UMAP2'].min(), combined['UMAP2'].max()

    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding

    fig, axes = plt.subplots(len(clusters), 2, figsize=(30, figsize_per_row * len(clusters)))

    if len(clusters) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, number in enumerate(clusters):
        umap_df = None
        for candidate_dir in threshold_dirs:
            candidate_file = os.path.join(
                _umap_dir(base_path, receptor, candidate_dir, generator),
                f'umap_results_{generator}_{type_scaffold}_{type_cluster}_{number}.csv',
            )
            if os.path.exists(candidate_file):
                umap_df = pd.read_csv(candidate_file)
                threshold_dir = candidate_dir
                break
        if umap_df is None:
            continue

        ax = axes[row_idx, 0]
        _plot_umap_points(
            ax,
            umap_df,
            UMAP_VIEW_CATEGORIES['reference'],
            legend_fontsize=18,
            title=f'Cluster {number}: Reference Sets',
            title_fontsize=20,
            axis_label_fontsize=20,
            tick_label_fontsize=15,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax = axes[row_idx, 1]
        context_categories = [label for label in UMAP_VIEW_CATEGORIES['context'] if label in categories]
        _plot_umap_points(
            ax,
            umap_df,
            context_categories,
            legend_fontsize=18,
            title=f'Cluster {number}: Full Context',
            title_fontsize=20,
            axis_label_fontsize=20,
            tick_label_fontsize=15,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if title is not None:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
    else:
        plt.tight_layout()

    folder = _umap_dir(base_path, receptor, 'img', 'grid_plot_print', threshold_dir)
    os.makedirs(folder, exist_ok=True)

    filename = f'umap_grid_print_{generator}_{type_scaffold}_{type_cluster}.png'
    plt.savefig(os.path.join(folder, filename), format='png', dpi=dpi, bbox_inches='tight')
    filename = f'umap_grid_print_{generator}_{type_scaffold}_{type_cluster}.svg'
    plt.savefig(os.path.join(folder, filename), format='svg', dpi=dpi, bbox_inches='tight')
    print(f'Saved: {os.path.join(folder, filename)}')
    plt.show()
