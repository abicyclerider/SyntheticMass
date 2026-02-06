"""
Field-by-field comparison for candidate record pairs.

Calculates similarity scores using appropriate methods for each field type.
"""

import pandas as pd
import recordlinkage as rl
from recordlinkage.compare import String, Exact, Date
import logging

logger = logging.getLogger(__name__)


def build_comparison_features(pairs: pd.MultiIndex, df: pd.DataFrame,
                              config: dict) -> pd.DataFrame:
    """
    Calculate similarity scores for all candidate pairs.

    Args:
        pairs: MultiIndex of candidate pairs
        df: Patient DataFrame with record_id as index
        config: Configuration dictionary with comparison thresholds

    Returns:
        DataFrame with similarity scores for each field
    """
    # Ensure record_id is the index
    if 'record_id' in df.columns and df.index.name != 'record_id':
        df = df.set_index('record_id')

    comp_config = config.get('comparison', {})

    # Create custom comparator
    compare = create_custom_comparator(comp_config)

    # Compute comparison features
    logger.info(f"Computing comparison features for {len(pairs)} candidate pairs...")
    features = compare.compute(pairs, df)

    logger.info(f"Generated {len(features.columns)} comparison features")

    return features


def create_custom_comparator(config: dict) -> rl.Compare:
    """
    Create a recordlinkage Compare object with custom comparison methods.

    Args:
        config: Configuration dictionary with comparison thresholds

    Returns:
        Configured Compare object
    """
    compare = rl.Compare()

    # First name: Jaro-Winkler similarity
    compare.string(
        'first_name', 'first_name',
        method='jarowinkler',
        threshold=config.get('first_name_threshold', 0.85),
        label='first_name_sim'
    )

    # Last name: Jaro-Winkler similarity
    compare.string(
        'last_name', 'last_name',
        method='jarowinkler',
        threshold=config.get('last_name_threshold', 0.85),
        label='last_name_sim'
    )

    # Maiden name: Jaro-Winkler similarity (if available)
    compare.string(
        'maiden_name', 'maiden_name',
        method='jarowinkler',
        threshold=config.get('maiden_name_threshold', 0.85),
        label='maiden_name_sim',
        missing_value=0.5  # Neutral score if missing
    )

    # Address: Jaro-Winkler with lower threshold (handles abbreviations)
    compare.string(
        'address', 'address',
        method='jarowinkler',
        threshold=config.get('address_threshold', 0.80),
        label='address_sim'
    )

    # City: Jaro-Winkler
    compare.string(
        'city', 'city',
        method='jarowinkler',
        threshold=config.get('city_threshold', 0.90),
        label='city_sim'
    )

    # State: Exact match
    if config.get('state_exact_match', True):
        compare.exact('state', 'state', label='state_match')
    else:
        compare.string('state', 'state', method='jarowinkler', label='state_sim')

    # ZIP: Exact match or Levenshtein
    if config.get('zip_exact_match', True):
        compare.exact('zip', 'zip', label='zip_match')
    else:
        compare.string('zip', 'zip', method='levenshtein', label='zip_sim')

    # SSN: Exact or Levenshtein distance
    if config.get('ssn_exact_match', True):
        compare.exact('ssn', 'ssn', label='ssn_match')
    else:
        compare.string(
            'ssn', 'ssn',
            method='levenshtein',
            threshold=config.get('ssn_levenshtein_threshold', 2),
            label='ssn_sim'
        )

    # Birthdate: Date comparison with tolerance
    birthdate_tolerance = config.get('birthdate_tolerance_days', 1)
    if birthdate_tolerance > 0:
        # Use custom date comparison with tolerance
        compare.date(
            'birthdate', 'birthdate',
            label='birthdate_match'
        )
    else:
        # Exact date match
        compare.exact('birthdate', 'birthdate', label='birthdate_match')

    # Gender: Exact match
    compare.exact('gender', 'gender', label='gender_match')

    return compare


def calculate_custom_similarity(df1: pd.DataFrame, df2: pd.DataFrame,
                                field: str, method: str = 'jarowinkler') -> pd.Series:
    """
    Calculate custom similarity scores for a specific field.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        field: Field name to compare
        method: Similarity method (jarowinkler, levenshtein, etc.)

    Returns:
        Series of similarity scores
    """
    if method == 'jarowinkler':
        from jellyfish import jaro_winkler_similarity
        return df1[field].combine(df2[field], lambda a, b: jaro_winkler_similarity(str(a), str(b)))

    elif method == 'levenshtein':
        from Levenshtein import distance as levenshtein_distance
        return df1[field].combine(df2[field], lambda a, b: 1 - levenshtein_distance(str(a), str(b)) / max(len(str(a)), len(str(b))))

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def analyze_similarity_distribution(features: pd.DataFrame, field: str,
                                   true_matches: pd.Series = None) -> dict:
    """
    Analyze the distribution of similarity scores for a field.

    Args:
        features: DataFrame with similarity scores
        field: Field name to analyze
        true_matches: Boolean series indicating true matches (optional)

    Returns:
        Dictionary with distribution statistics
    """
    if field not in features.columns:
        raise ValueError(f"Field {field} not found in features")

    stats = {
        'field': field,
        'mean': features[field].mean(),
        'median': features[field].median(),
        'std': features[field].std(),
        'min': features[field].min(),
        'max': features[field].max(),
        'q25': features[field].quantile(0.25),
        'q75': features[field].quantile(0.75)
    }

    if true_matches is not None:
        stats['mean_matches'] = features.loc[true_matches, field].mean()
        stats['mean_non_matches'] = features.loc[~true_matches, field].mean()

    return stats


def add_composite_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite features combining multiple similarity scores.

    Args:
        features: DataFrame with individual similarity scores

    Returns:
        DataFrame with added composite features
    """
    features = features.copy()

    # Total similarity score (sum of all features)
    features['total_score'] = features.sum(axis=1)

    # Name similarity (average of first, last, maiden)
    name_cols = [c for c in ['first_name_sim', 'last_name_sim', 'maiden_name_sim'] if c in features.columns]
    if name_cols:
        features['name_score'] = features[name_cols].mean(axis=1)

    # Address similarity (average of address, city, state, zip)
    addr_cols = [c for c in ['address_sim', 'city_sim', 'state_match', 'zip_match'] if c in features.columns]
    if addr_cols:
        features['address_score'] = features[addr_cols].mean(axis=1)

    # High confidence indicators
    if 'ssn_match' in features.columns and 'birthdate_match' in features.columns:
        features['high_confidence'] = (features['ssn_match'] == 1) & (features['birthdate_match'] == 1)

    return features
