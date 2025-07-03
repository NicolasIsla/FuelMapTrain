import random
from tqdm import tqdm
import numpy as np
from scipy.special import rel_entr
from shapeft.datasets.base import GeoFMDataset
from shapeft.datasets.base import GeoFMSubset

# Calculate image-wise class distributions for segmentation
def calculate_class_distributions(dataset: GeoFMDataset|GeoFMSubset):
    num_classes = dataset.num_classes
    ignore_index = dataset.ignore_index
    class_distributions = []

    for idx in tqdm(range(len(dataset)), desc="Calculating class distributions per sample"):
        target = dataset[idx]['target']

        if ignore_index is not None:
            target=target[(target != ignore_index)]

        total_pixels = target.numel()
        if total_pixels == 0:
            class_distributions.append([0] * num_classes)
            continue
        else:
            class_counts = [(target == i).sum().item() for i in range(num_classes)]
            class_ratios = [count / total_pixels for count in class_counts]
            class_distributions.append(class_ratios)

    return np.array(class_distributions)



# Function to bin class distributions using ceil
def bin_class_distributions(class_distributions, num_bins=3, logger=None):
    logger.info(f"Class distributions are being binned into {num_bins} categories using ceil")
    
    bin_edges = np.linspace(0, 1, num_bins + 1)[1]
    binned_distributions = np.ceil(class_distributions / bin_edges).astype(int) - 1
    return binned_distributions


def balance_seg_indices(
        dataset:GeoFMDataset|GeoFMSubset, 
        strategy, 
        label_fraction=1.0, 
        num_bins=3, 
        logger=None):
    """
    Balances and selects indices from a segmentation dataset based on the specified strategy.

    Args:
    dataset : GeoFMDataset | GeoFMSubset
        The dataset from which to select indices, typically containing geospatial segmentation data.
    
    strategy : str
        The strategy to use for selecting indices. Options include:
        - "stratified": Proportionally selects indices from each class bin based on the class distribution.
        - "oversampled": Prioritizes and selects indices from bins with lower class representation.
    
    label_fraction : float, optional, default=1.0
        The fraction of labels (indices) to select from each class or bin. Values should be between 0 and 1.
    
    num_bins : int, optional, default=3
        The number of bins to divide the class distributions into, used for stratification or oversampling.
    
    logger : object, optional
        A logger object for tracking progress or logging messages (e.g., `logging.Logger`)

    ------
    
    Returns:
    selected_idx : list of int
        The indices of the selected samples based on the strategy and label fraction.

    other_idx : list of int
        The remaining indices that were not selected.

    """
    # Calculate class distributions with progress tracking
    class_distributions = calculate_class_distributions(dataset)

    # Bin the class distributions
    binned_distributions = bin_class_distributions(class_distributions, num_bins=num_bins, logger=logger)
    combined_bins = np.apply_along_axis(lambda row: ''.join(map(str, row)), axis=1, arr=binned_distributions)

    indices_per_bin = {}
    for idx, bin_id in enumerate(combined_bins):
        if bin_id not in indices_per_bin:
            indices_per_bin[bin_id] = []
        indices_per_bin[bin_id].append(idx)

    if strategy == "stratified":
        # Select a proportion of indices from each bin   
        # selected_idx = []
        # for bin_id, indices in indices_per_bin.items():
        #     num_to_select = int(max(1, len(indices) * label_fraction))  # Ensure at least one index is selected
        #     selected_idx.extend(np.random.choice(indices, num_to_select, replace=False))
        
        # Compute the "mean bin" vector and select samples whose bins are closest
        mean_bin = np.round(binned_distributions.mean(axis=0)).astype(int)
        # Compute distances to mean bin
        distances = np.array([js_divergence(b, mean_bin) for b in binned_distributions])
        total_to_select = int(len(dataset) * label_fraction)
        sorted_idx = np.argsort(distances)
        selected_idx = sorted_idx[:total_to_select].tolist()
        # Ensure every class appears at least once
        num_classes = binned_distributions.shape[1]
        for c in range(num_classes):
            if not any(binned_distributions[i, c] > 0 for i in selected_idx):
                # find nearest unselected sample with class c present
                candidates = [i for i, b in enumerate(binned_distributions)
                              if b[c] > 0 and i not in selected_idx]
                if candidates:
                    nearest = min(candidates, key=lambda i: distances[i])
                    selected_idx.append(int(nearest))

    elif strategy == "oversampled":
        # Prioritize the bins with the lowest values
        sorted_indices = np.argsort(combined_bins)
        selected_idx = sorted_indices[:int(len(dataset) * label_fraction)]

    # Determine the remaining indices not selected
    other_idx = list(set(range(len(dataset))) - set(selected_idx))

    return selected_idx, other_idx


# Function to get subset indices based on the strategy
def get_subset_indices(dataset: GeoFMDataset, 
                       task="segmentation",
                       strategy="random", 
                       label_fraction=0.5, 
                       num_bins=3, 
                       logger=None):
    logger.info(
        f"Creating a subset of the {dataset.split} dataset using {strategy} strategy, with {label_fraction * 100}% of labels utilized."
    )
    assert strategy in ["random", "stratified", "oversampled"], "Unsupported dataset subsampling strategy"
    
    if strategy == "random":
        n_samples = len(dataset)
        indices = random.sample(
            range(n_samples), int(n_samples * label_fraction)
        )
        return indices
    
    if task == "segmentation":
        indices, _ = balance_seg_indices(
            dataset, strategy=strategy, label_fraction=label_fraction, num_bins=num_bins, logger=logger
        )
    
    return indices



# Define Jensen-Shannon divergence based on KL
def js_divergence(p, q, eps=1e-12):
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # symmetrized KL
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))