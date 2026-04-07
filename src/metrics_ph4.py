"""
Pharmacophore Fingerprint Metrics Calculator

Calculates similarity metrics between output and recall sets using Tanimoto similarity.
Can be run from command line or imported into Jupyter notebooks.

Usage:
    Command line:
        python metrics_phfp.py --type_cluster dis --type_phfp phfp2d --generator addcarbon_250k --receptor Glucocorticoid_receptor --threshold 0.7
    
    Jupyter:
        from metrics_phfp import Metrics_phfp
        mt = Metrics_phfp('dis', 'phfp2d', 'addcarbon_250k', 'Glucocorticoid_receptor', threshold=0.7)
        mt.calculate()
"""

import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, shared_memory
import argparse
from typing import List, Tuple, Optional

from src.path_utils import data_subdir, pharm_results_dir


# Global variables for shared memory (set by worker processes)
_shm = None
_output_fps_shape = None
_output_fps_dtype = None

VALID_CLUSTER_TYPES = {"dis", "sim"}


def tanimoto_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Tanimoto similarity between two binary vectors.
    
    Args:
        vec1: First binary vector
        vec2: Second binary vector
    
    Returns:
        Tanimoto similarity score (0-1)
    """
    intersection = np.count_nonzero(np.logical_and(vec1, vec2))
    union = np.count_nonzero(np.logical_or(vec1, vec2))
    
    return 1.0 if union == 0 else intersection / union


def convert_fp_to_array(fp_string: str) -> np.ndarray:
    """Convert fingerprint string to numpy array."""
    return np.array([int(bit) for bit in fp_string], dtype=np.int8)


def init_worker(shm_name: Optional[str], shape: Optional[Tuple], dtype: Optional[np.dtype]):
    """
    Initialize worker process with shared memory.
    
    Args:
        shm_name: Name of shared memory block
        shape: Shape of the array
        dtype: Data type of the array
    """
    global _shm, _output_fps_shape, _output_fps_dtype
    if shm_name is not None:
        _shm = shared_memory.SharedMemory(name=shm_name)
        _output_fps_shape = shape
        _output_fps_dtype = dtype


def process_single_recall_fp_shared(args: Tuple[int, np.ndarray, float]) -> dict:
    """
    Process a single recall fingerprint using shared memory for output FPs.
    
    Args:
        args: Tuple of (index, recall_fp, threshold)
    
    Returns:
        Dictionary with matching statistics
    """
    idx, recall_fp, threshold = args
    
    # Access shared memory
    output_fps_arr = np.ndarray(_output_fps_shape, dtype=_output_fps_dtype, buffer=_shm.buf)
    
    match_count = sum(
        tanimoto_similarity(recall_fp, output_fp) >= threshold 
        for output_fp in output_fps_arr
    )
    
    return {
        'label': f'FP-{idx}',
        'UAFo': 1 if match_count > 0 else 0,
        'CwAFo': match_count
    }


def process_single_recall_fp(args: Tuple[int, np.ndarray, List[np.ndarray], float]) -> dict:
    """
    Process a single recall fingerprint (non-shared memory version).
    
    Args:
        args: Tuple of (index, recall_fp, output_fps_list, threshold)
    
    Returns:
        Dictionary with matching statistics
    """
    idx, recall_fp, output_fps_arr, threshold = args
    
    match_count = sum(
        tanimoto_similarity(recall_fp, output_fp) >= threshold 
        for output_fp in output_fps_arr
    )
    
    return {
        'label': f'FP-{idx}',
        'UAFo': 1 if match_count > 0 else 0,
        'CwAFo': match_count
    }


def parallel_convert_fps(fp_list: List[str], ncpus: int = 1, batch_size: int = 5000) -> List[np.ndarray]:
    """
    Convert fingerprints to arrays - single threaded to avoid memory issues.
    
    Args:
        fp_list: List of fingerprint strings
        ncpus: Number of CPUs (not used, kept for API compatibility)
        batch_size: Batch size (not used)
    
    Returns:
        List of numpy arrays
    """
    total = len(fp_list)
    print(f"  Converting {total:,} fingerprints...")
    
    result = []
    for i, fp in enumerate(fp_list):
        result.append(convert_fp_to_array(fp))
        if (i + 1) % 50000 == 0:
            print(f"    Progress: {i+1:,}/{total:,} ({100*(i+1)/total:.1f}%)")
    
    return result


def create_matching_dataframe(recall_fps: pd.DataFrame, output_fps: pd.DataFrame, threshold: float, ncpus: int = 1) -> List[dict]:
    """
    Create matching dataframe by comparing recall and output fingerprints.
    Uses shared memory for efficient multiprocessing.
    
    Args:
        recall_fps: DataFrame with recall fingerprints
        output_fps: DataFrame with output fingerprints
        threshold: Tanimoto similarity threshold (e.g., 0.7)
        ncpus: Number of CPUs to use for parallel processing
    
    Returns:
        List of dictionaries with matching statistics
    """
    print('Converting recall fingerprints...')
    recall_fps_arr = parallel_convert_fps(recall_fps['fp'].tolist(), 1)
    
    print('Converting output fingerprints...')
    output_fps_clean = [fp for fp in output_fps['fp'].tolist() if pd.notna(fp)]
    output_fps_list = parallel_convert_fps(output_fps_clean, 1)
    
    # Convert to 2D numpy array for shared memory
    print('Creating shared memory array...')
    output_fps_arr = np.array(output_fps_list, dtype=np.int8)
    
    print(f"\nProcessing {len(recall_fps_arr)} recall FPs against {len(output_fps_arr)} output FPs...")
    print(f"Total comparisons: {len(recall_fps_arr) * len(output_fps_arr):,}")
    print(f"Memory usage: ~{output_fps_arr.nbytes / (1024**2):.1f} MB for output FPs")
    print(f"Using {ncpus} CPU(s) for comparisons")
    
    if ncpus > 1 and len(recall_fps_arr) > 10:
        # Use shared memory for parallel processing
        print("Running in parallel mode with shared memory...")
        
        try:
            # Create shared memory
            shm = shared_memory.SharedMemory(create=True, size=output_fps_arr.nbytes)
            shared_arr = np.ndarray(output_fps_arr.shape, dtype=output_fps_arr.dtype, buffer=shm.buf)
            shared_arr[:] = output_fps_arr[:]
            
            # Prepare arguments (without output_fps_arr to save memory)
            args_list = [
                (idx, recall_fp, threshold)
                for idx, recall_fp in enumerate(recall_fps_arr)
            ]
            
            # Process in parallel with shared memory
            with Pool(
                processes=ncpus, 
                initializer=init_worker,
                initargs=(shm.name, output_fps_arr.shape, output_fps_arr.dtype)
            ) as pool:
                data = list(pool.imap_unordered(
                    process_single_recall_fp_shared, 
                    args_list,
                    chunksize=max(1, len(args_list) // (ncpus * 4))
                ))
            
            # Clean up shared memory
            shm.close()
            shm.unlink()
            
            # Sort by label to maintain order
            data.sort(key=lambda x: int(x['label'].split('-')[1]))
            
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            print("Falling back to single-threaded mode...")
            ncpus = 1
    
    if ncpus == 1:
        # Single-threaded processing
        print("Running in single-threaded mode...")
        data = []
        for idx, recall_fp in enumerate(recall_fps_arr):
            if idx % 100 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(recall_fps_arr)} ({100*idx/len(recall_fps_arr):.1f}%)")
            
            match_count = sum(
                tanimoto_similarity(recall_fp, output_fp) >= threshold 
                for output_fp in output_fps_arr
            )
            
            data.append({
                'label': f'FP-{idx}',
                'UAFo': 1 if match_count > 0 else 0,
                'CwAFo': match_count
            })
    
    return data


def normalize_ncpus(ncpus: Optional[int]) -> int:
    """Return a safe CPU count bounded by available hardware."""
    available = os.cpu_count() or 1
    if ncpus is None:
        return 1
    return max(1, min(int(ncpus), available))


class Metrics_phfp:
    """
    Pharmacophore fingerprint metrics calculator.
    
    Attributes:
        type_cluster: Type of clustering ('dis' or 'sim')
        type_phfp: Type of pharmacophore fingerprint
        generator_name: Name of the molecule generator
        receptor: Receptor name
        threshold: Tanimoto similarity threshold (default: 1.0)
        data_folder: Base path to data folder (default: '')
        ncpus: Number of CPUs for multiprocessing (default: 1)
    """
    
    def __init__(
        self, 
        type_cluster: str,
        type_phfp: str,
        generator_name: str,
        receptor: str,
        threshold: float = 1.0,
        data_folder: str = '',
        ncpus: int = 1
    ):
        if type_cluster not in VALID_CLUSTER_TYPES:
            raise ValueError(
                f"Unsupported cluster type '{type_cluster}'. Use one of {sorted(VALID_CLUSTER_TYPES)}."
            )
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1.")

        self.type_cluster = type_cluster
        self.type_phfp = type_phfp
        self.generator_name = generator_name
        self.receptor = receptor
        self.threshold = threshold
        self.data_folder = data_folder
        self.ncpus = normalize_ncpus(ncpus)
        
        # Will be populated during calculation
        self.number_of_calculation: Optional[int] = None
        self.output_set_phfp: Optional[pd.DataFrame] = None
        self.recall_set_phfp: Optional[pd.DataFrame] = None
        self.count_metrics: Optional[pd.DataFrame] = None

    def _get_output_dir(self) -> str:
        """Get output directory path."""
        return str(
            pharm_results_dir(
                self.data_folder,
                self.receptor,
                self.type_phfp,
                self.type_cluster,
                self.generator_name,
                f"threshold_{self.threshold}",
            )
        ) + "/"

    def load(self, filepath_output_set: str, filepath_recall_set: str):
        """
        Load output and recall sets from CSV files.
        
        Args:
            filepath_output_set: Path to output set CSV
            filepath_recall_set: Path to recall set CSV
        """
        print(f"\nLoading data from:")
        print(f"  Output: {filepath_output_set}")
        print(f"  Recall: {filepath_recall_set}")
        
        self.output_set_phfp = pd.read_csv(filepath_output_set, usecols=["fp"])
        self.recall_set_phfp = pd.read_csv(filepath_recall_set, usecols=["fp"])
        
        print(f'Original output length: {len(self.output_set_phfp):,}')
        print(f'Original recall length: {len(self.recall_set_phfp):,}')
        
        # Keep only unique recall fingerprints
        self.recall_set_phfp = self.recall_set_phfp.drop_duplicates(keep='first').reset_index(drop=True)
        print(f'Unique recall length: {len(self.recall_set_phfp):,}')

    def calculate_metrics(self):
        """Calculate similarity metrics between output and recall sets."""
        print("\nCalculating matching statistics...")
        df = create_matching_dataframe(self.recall_set_phfp, self.output_set_phfp, self.threshold, self.ncpus)
        self.count_metrics = pd.DataFrame(df)
        
        # Calculate metrics
        UFo = len(self.output_set_phfp['fp'].drop_duplicates())  # Unique fingerprints in output
        SSo = len(self.output_set_phfp)  # Total output set size
        CwAFo = self.count_metrics['CwAFo'].sum()  # Total compound matches
        UAFo = self.count_metrics['UAFo'].sum()  # Unique active fingerprints in output
        UAFr = len(self.count_metrics)  # Unique active fingerprints in recall
        
        # Calculate derived metrics
        rs = UAFo / UAFr if UAFr > 0 else 0
        rs_text = f"{UAFo}/{UAFr}"
        SED = UFo / SSo if SSo > 0 else 0
        ASER = CwAFo / SSo if SSo > 0 else 0
        
        # Create results dataframe
        results = pd.DataFrame({
            'name': [f"{self.generator_name}_{self.number_of_calculation}"],
            'type_cluster': [self.type_cluster],
            'phfp': [self.type_phfp],
            'UFo': [UFo],
            'SSo': [SSo],
            'RS_': [rs_text],
            'RS': [rs],
            'SED': [SED],
            'ASER': [ASER],
            'CwAFo': [CwAFo]
        })
        
        # Save results
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        count_file = f"{output_dir}count_of_occurrence_cluster_{self.number_of_calculation}_{self.type_cluster}_{self.generator_name}_threshold_{self.threshold}.csv"
        metrics_file = f"{output_dir}metrics_cluster_{self.number_of_calculation}_{self.type_cluster}_{self.generator_name}_threshold_{self.threshold}.csv"
        
        self.count_metrics.to_csv(count_file, index=False)
        results.to_csv(metrics_file, index=False)
        
        print(f"\nResults saved:")
        print(f"  Counts: {count_file}")
        print(f"  Metrics: {metrics_file}")
        print(f"\nMetrics: RS={rs:.4f}, SED={SED:.4f}, ASER={ASER:.4f}")

        return results.copy()

    def average_value(self, numbers: List[int]) -> pd.DataFrame:
        """
        Calculate average metrics across multiple clusters.
        
        Args:
            numbers: List of cluster numbers to average
        
        Returns:
            Combined DataFrame with mean values
        """
        output_dir = self._get_output_dir()
        
        # Load all cluster results
        file_paths = [
            f"{output_dir}metrics_cluster_{num}_{self.type_cluster}_{self.generator_name}_threshold_{self.threshold}.csv"
            for num in numbers
        ]
        
        print(f"\nAveraging results across {len(numbers)} clusters...")
        combined_df = pd.concat([pd.read_csv(path) for path in file_paths], ignore_index=True)
        
        # Calculate mean for numeric columns
        mean_values = combined_df.mean(numeric_only=True)
        
        # Create mean row
        mean_row = {
            'name': f"{self.generator_name}_mean",
            'type_cluster': self.type_cluster,
            'phfp': self.type_phfp,
            'UFo': mean_values.get('UFo', np.nan),
            'SSo': mean_values.get('SSo', np.nan),
            'RS_': '-',
            'RS': mean_values.get('RS', np.nan),
            'SED': mean_values.get('SED', np.nan),
            'ASER': mean_values.get('ASER', np.nan),
            'CwAFo': mean_values.get('CwAFo', np.nan)
        }
        
        # Append mean row
        combined_df = pd.concat([combined_df, pd.DataFrame([mean_row])], ignore_index=True)
        combined_df = combined_df.round(7)
        
        # Create formatted version for display
        formatted_df = combined_df.copy()
        for col in ['SSo', 'UFo', 'CwAFo']:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: "{:,}".format(int(x)) if pd.notnull(x) else x
            )
        
        # Save results
        formatted_df.to_csv(f"{output_dir}df_all_clusters_with_mean_with_coma_threshold_{self.threshold}.csv", index=False)
        combined_df.to_csv(f"{output_dir}df_all_clusters_with_mean_threshold_{self.threshold}.csv", index=False)
        combined_df.tail(1).to_csv(f"{output_dir}{self.generator_name}_mean_{self.type_phfp}_{self.type_cluster}_threshold_{self.threshold}.csv", index=False)
        
        print(f"\nAverage results saved to: {output_dir}")
        
        return combined_df

    def calculate(self, cluster_range: range = range(5)):
        """
        Calculate metrics for all available clusters.
        
        Args:
            cluster_range: Range of cluster numbers to process (default: 0-4)
        """
        print(f"\n{'='*60}")
        print(f"Starting calculation for {self.generator_name}")
        print(f"Receptor: {self.receptor}")
        print(f"Cluster type: {self.type_cluster}")
        print(f"Threshold: {self.threshold}")
        print(f"{'='*60}")
        
        numbers = []
        
        for number in cluster_range:
            self.number_of_calculation = number
            
            output_file_path = (
                data_subdir(
                    self.data_folder,
                    "output_sets",
                    "ph4",
                    self.receptor,
                    self.generator_name,
                )
                / f"phfp_of_output_set_cluster_{number}_{self.type_cluster}_{self.generator_name}_with_smiles.csv"
            )
            recall_file_path = (
                data_subdir(
                    self.data_folder,
                    "output_sets",
                    "ph4",
                    self.receptor,
                    "RS",
                )
                / f"phfp_of_recall_set_cluster_{number}_{self.type_cluster}_with_smiles.csv"
            )
            
            if os.path.exists(output_file_path):
                print(f"\n--- Processing cluster {number} ---")
                self.load(str(output_file_path), str(recall_file_path))
                numbers.append(number)
                self.calculate_metrics()
            else:
                print(f"\nSkipping cluster {number} (file not found)")
        
        if numbers:
            result = self.average_value(numbers)
            print(f"\n{'='*60}")
            print(f"Calculation complete! Processed {len(numbers)} clusters")
            print(f"{'='*60}")
            return result[['name', 'type_cluster', 'phfp', 'RS', 'SED', 'ASER']].copy()
        else:
            print('\nNo data found for calculation')
            return pd.DataFrame()


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Calculate pharmacophore-based recall metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics_ph4.py --type_cluster dis --type_phfp rdkit --generator addcarbon_250k --receptor Glucocorticoid_receptor --threshold 0.7
  python metrics_ph4.py --type_cluster sim --type_phfp rdkit --generator Molpher_250k --receptor Glucocorticoid_receptor --threshold 0.8 --data_folder /path/to/data
        """
    )
    
    parser.add_argument('--type_cluster', type=str, required=True, 
                        help='Type of clustering (dis/sim)')
    parser.add_argument('--type_phfp', type=str, required=True,
                        help='Type of pharmacophore fingerprint')
    parser.add_argument('--generator', type=str, required=True,
                        help='Generator name (e.g., addcarbon_250k)')
    parser.add_argument('--receptor', type=str, required=True,
                        help='Receptor name (e.g., Glucocorticoid_receptor)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Tanimoto similarity threshold (default: 1.0)')
    parser.add_argument('--data_folder', type=str, default='',
                        help='Base path to data folder (default: current directory)')
    parser.add_argument('--ncpus', type=int, default=1,
                        help='Number of CPUs for parallel processing (default: 1)')
    
    args = parser.parse_args()
    
    # Create metrics calculator and run
    mt = Metrics_phfp(
        type_cluster=args.type_cluster,
        type_phfp=args.type_phfp,
        generator_name=args.generator,
        receptor=args.receptor,
        threshold=args.threshold,
        data_folder=args.data_folder,
        ncpus=args.ncpus
    )
    mt.calculate()


if __name__ == "__main__":
    main()


MetricsPh4 = Metrics_phfp
