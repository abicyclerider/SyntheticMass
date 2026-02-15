"""Unit tests for entity_resolution/core/golden_record.py — cluster building and conflict resolution."""

import pandas as pd
import pytest

from entity_resolution.core.golden_record import (
    apply_field_specific_rules,
    build_match_clusters,
    merge_cluster_records,
    resolve_field_conflict,
)


@pytest.mark.unit
class TestBuildMatchClusters:
    def _make_matches(self, pairs):
        """Helper: create a boolean Series with MultiIndex from a list of (id1, id2) pairs."""
        if not pairs:
            index = pd.MultiIndex.from_tuples([], names=["record_id_1", "record_id_2"])
            return pd.Series([], dtype=bool, index=index)
        index = pd.MultiIndex.from_tuples(pairs)
        return pd.Series([True] * len(pairs), index=index)

    def test_simple_cluster(self):
        """2 pairs sharing a node -> 1 cluster of 3."""
        matches = self._make_matches([("a", "b"), ("b", "c")])
        clusters = build_match_clusters(matches)
        assert len(clusters) == 1
        assert clusters[0] == {"a", "b", "c"}

    def test_disjoint_clusters(self):
        """2 unrelated pairs -> 2 clusters of 2."""
        matches = self._make_matches([("a", "b"), ("c", "d")])
        clusters = build_match_clusters(matches)
        assert len(clusters) == 2
        cluster_sets = [frozenset(c) for c in clusters]
        assert frozenset({"a", "b"}) in cluster_sets
        assert frozenset({"c", "d"}) in cluster_sets

    def test_chain(self):
        """A-B, B-C, C-D -> 1 cluster of 4."""
        matches = self._make_matches([("a", "b"), ("b", "c"), ("c", "d")])
        clusters = build_match_clusters(matches)
        assert len(clusters) == 1
        assert clusters[0] == {"a", "b", "c", "d"}

    def test_empty(self):
        """No matches -> empty list."""
        matches = self._make_matches([])
        clusters = build_match_clusters(matches)
        assert clusters == []

    def test_false_matches_ignored(self):
        """Pairs with False value are not included in clusters."""
        index = pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")])
        matches = pd.Series([True, False], index=index)
        clusters = build_match_clusters(matches)
        assert len(clusters) == 1
        assert clusters[0] == {"a", "b"}


@pytest.mark.unit
class TestResolveFieldConflict:
    def test_most_frequent(self):
        """Most common value wins."""
        result = resolve_field_conflict(
            ["John", "Jon", "John"], strategy="most_frequent"
        )
        assert result == "John"

    def test_all_nan(self):
        """All None/NaN/empty -> None."""
        result = resolve_field_conflict(
            [None, float("nan"), ""], strategy="most_frequent"
        )
        assert result is None

    def test_single_value(self):
        """Single valid value -> that value."""
        result = resolve_field_conflict(["John"], strategy="most_frequent")
        assert result == "John"

    def test_single_value_with_nans(self):
        """One valid value among NaNs -> that value."""
        result = resolve_field_conflict([None, "John", ""], strategy="most_frequent")
        assert result == "John"


@pytest.mark.unit
class TestFieldSpecificRules:
    def test_address_prefers_longest(self):
        """Address: prefer longest (non-abbreviated) form."""
        result = apply_field_specific_rules(
            ["123 Main St", "123 Main Street"], field_name="address"
        )
        assert result == "123 Main Street"

    def test_ssn_prefers_dashed(self):
        """SSN: prefer format with dashes."""
        result = apply_field_specific_rules(
            ["123456789", "123-45-6789"], field_name="ssn"
        )
        assert result == "123-45-6789"

    def test_name_prefers_title_case(self):
        """Name: prefer title case."""
        result = apply_field_specific_rules(["JOHN", "John"], field_name="first_name")
        assert result == "John"


@pytest.mark.unit
class TestMergeClusterRecords:
    def test_merge_produces_golden_record(self, sample_patients_standardized):
        """3 records -> golden record with correct fields and provenance."""
        cluster_records = sample_patients_standardized.iloc[:3]  # 3 records
        config = {"conflict_resolution": "most_frequent", "include_provenance": True}

        golden = merge_cluster_records(cluster_records, cluster_id=0, config=config)

        assert golden["golden_id"] == "golden_000000"
        assert golden["num_records"] == 3
        assert "facilities" in golden
        assert "source_record_ids" in golden
        assert "first_name" in golden
        assert "last_name" in golden

    def test_merge_without_provenance(self, sample_patients_standardized):
        """Provenance can be disabled via config."""
        cluster_records = sample_patients_standardized.iloc[:2]
        config = {"conflict_resolution": "most_frequent", "include_provenance": False}

        golden = merge_cluster_records(cluster_records, cluster_id=1, config=config)

        assert golden["golden_id"] == "golden_000001"
        assert "facilities" not in golden
        assert "source_record_ids" not in golden
