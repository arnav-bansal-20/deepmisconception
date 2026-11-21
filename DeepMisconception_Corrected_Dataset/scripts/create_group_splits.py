"""
DeepMisconception Split-First Partitioning Engine (Safeguard 4)
Partitions the 65 Scenario Families into Train (46), Validation (10), and Test (9)
PRIOR to dependent synthetic sample generation.
Ensures zero scenario family or template leakage between partitions.
"""

import numpy as np

def assign_scenario_families_to_splits(scenario_families: list, seed: int = 42):
    """
    Assigns scenario families to partitions BEFORE any dependent synthetic generation.
    Returns: {
        'train_sf': set of sf_ids,
        'val_sf': set of sf_ids,
        'test_sf': set of sf_ids,
        'sf_to_split': dict mapping sf_id -> 'train' | 'validation' | 'test'
    }
    """
    np.random.seed(seed)
    
    sf_ids = [sf["id"] for sf in scenario_families]
    np.random.shuffle(sf_ids)
    
    n_total_sf = len(sf_ids)
    n_train_sf = int(round(0.70 * n_total_sf))
    n_val_sf = int(round(0.15 * n_total_sf))
    
    train_sf = set(sf_ids[:n_train_sf])
    val_sf = set(sf_ids[n_train_sf:n_train_sf + n_val_sf])
    test_sf = set(sf_ids[n_train_sf + n_val_sf:])
    
    # Audit partition disjointness
    assert len(train_sf.intersection(val_sf)) == 0, "Leakage: Train and Validation overlap!"
    assert len(train_sf.intersection(test_sf)) == 0, "Leakage: Train and Test overlap!"
    assert len(val_sf.intersection(test_sf)) == 0, "Leakage: Validation and Test overlap!"
    
    sf_to_split = {}
    for sf_id in train_sf:
        sf_to_split[sf_id] = "train"
    for sf_id in val_sf:
        sf_to_split[sf_id] = "validation"
    for sf_id in test_sf:
        sf_to_split[sf_id] = "test"
        
    return {
        "train_sf": train_sf,
        "val_sf": val_sf,
        "test_sf": test_sf,
        "sf_to_split": sf_to_split
    }
