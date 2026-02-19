"""Unit tests for confusable patient group generation."""

import pandas as pd
import pytest

from augmentation.core.confusable_groups import ConfusableGroupGenerator
from augmentation.tests.fixtures.sample_data import create_sample_patients


def _make_patient_facilities(
    patients_df: pd.DataFrame, num_facilities: int = 3
) -> dict[str, list[int]]:
    """Assign every patient to facility 0, spreading some across others."""
    pf: dict[str, list[int]] = {}
    for i, pid in enumerate(patients_df["Id"].values):
        fids = [0]
        if i % 3 == 0 and num_facilities > 1:
            fids.append(1)
        pf[pid] = fids
    return pf


@pytest.mark.unit
class TestConfusableGroupGenerator:
    """Test pair selection and metadata output."""

    def test_requested_count(self):
        patients_df = create_sample_patients(50)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(random_seed=7)

        meta = gen.generate(patients_df, pf, total_pairs=10)
        assert len(meta) == 10

    def test_zero_is_noop(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(random_seed=7)
        original = patients_df.copy()

        meta = gen.generate(patients_df, pf, total_pairs=0)

        assert len(meta) == 0
        pd.testing.assert_frame_equal(patients_df, original)

    def test_no_duplicate_patients(self):
        patients_df = create_sample_patients(50)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(random_seed=7)

        meta = gen.generate(patients_df, pf, total_pairs=15)

        all_ids = list(meta["source_patient_id"]) + list(meta["target_patient_id"])
        assert len(all_ids) == len(set(all_ids)), "A patient appears in multiple pairs"

    def test_type_distribution(self):
        patients_df = create_sample_patients(200)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(random_seed=42)

        meta = gen.generate(patients_df, pf, total_pairs=80)
        counts = meta["group_type"].value_counts()

        # With 80 pairs, each type should appear at least once
        for t in ["twin", "parent_child", "sibling"]:
            assert t in counts.index, f"Missing group type {t}"
            assert counts[t] >= 1

    def test_insufficient_patients_partial(self):
        patients_df = create_sample_patients(6)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(random_seed=7)

        meta = gen.generate(patients_df, pf, total_pairs=100)

        # Can't make 100 pairs from 6 patients (max 3 pairs)
        assert len(meta) <= 3

    def test_reproducibility(self):
        patients_df = create_sample_patients(50)
        pf = _make_patient_facilities(patients_df)

        df1 = patients_df.copy()
        gen1 = ConfusableGroupGenerator(random_seed=99)
        meta1 = gen1.generate(df1, pf, total_pairs=10)

        df2 = patients_df.copy()
        gen2 = ConfusableGroupGenerator(random_seed=99)
        meta2 = gen2.generate(df2, pf, total_pairs=10)

        pd.testing.assert_frame_equal(meta1, meta2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_metadata_schema(self):
        patients_df = create_sample_patients(50)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(random_seed=7)

        meta = gen.generate(patients_df, pf, total_pairs=5)

        assert list(meta.columns) == [
            "source_patient_id",
            "target_patient_id",
            "group_type",
            "fields_copied",
        ]
        for _, row in meta.iterrows():
            assert row["source_patient_id"] != row["target_patient_id"]
            assert row["group_type"] in ("twin", "parent_child", "sibling")
            assert len(row["fields_copied"]) > 0


@pytest.mark.unit
class TestTwinType:
    """Test twin demographic overwrite."""

    def test_correct_fields_copied(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(
            type_weights={"twin": 1.0, "parent_child": 0.0, "sibling": 0.0},
            random_seed=7,
        )
        original = patients_df.copy()

        meta = gen.generate(patients_df, pf, total_pairs=5)
        assert (meta["group_type"] == "twin").all()

        for _, row in meta.iterrows():
            src = original.loc[original["Id"] == row["source_patient_id"]].iloc[0]
            tgt = patients_df.loc[patients_df["Id"] == row["target_patient_id"]].iloc[0]

            assert tgt["LAST"] == src["LAST"]
            assert tgt["BIRTHDATE"] == src["BIRTHDATE"]
            assert tgt["ADDRESS"] == src["ADDRESS"]
            assert tgt["CITY"] == src["CITY"]
            assert tgt["ZIP"] == src["ZIP"]
            assert tgt["GENDER"] == src["GENDER"]
            assert tgt["STATE"] == src["STATE"]

    def test_first_preserved_different(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(
            type_weights={"twin": 1.0, "parent_child": 0.0, "sibling": 0.0},
            random_seed=7,
        )
        original = patients_df.copy()

        meta = gen.generate(patients_df, pf, total_pairs=5)

        for _, row in meta.iterrows():
            orig_tgt = original.loc[original["Id"] == row["target_patient_id"]].iloc[0]
            new_tgt = patients_df.loc[
                patients_df["Id"] == row["target_patient_id"]
            ].iloc[0]
            # FIRST should NOT have been overwritten
            assert new_tgt["FIRST"] == orig_tgt["FIRST"]

    def test_ssn_preserved_different(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(
            type_weights={"twin": 1.0, "parent_child": 0.0, "sibling": 0.0},
            random_seed=7,
        )
        original = patients_df.copy()

        meta = gen.generate(patients_df, pf, total_pairs=5)

        for _, row in meta.iterrows():
            orig_tgt = original.loc[original["Id"] == row["target_patient_id"]].iloc[0]
            new_tgt = patients_df.loc[
                patients_df["Id"] == row["target_patient_id"]
            ].iloc[0]
            assert new_tgt["SSN"] == orig_tgt["SSN"]


@pytest.mark.unit
class TestParentChildType:
    """Test parent-child namesake demographic overwrite."""

    def test_correct_fields_and_birthdate_offset(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(
            type_weights={"twin": 0.0, "parent_child": 1.0, "sibling": 0.0},
            random_seed=7,
        )
        original = patients_df.copy()

        meta = gen.generate(patients_df, pf, total_pairs=5)
        assert (meta["group_type"] == "parent_child").all()

        for _, row in meta.iterrows():
            src = original.loc[original["Id"] == row["source_patient_id"]].iloc[0]
            tgt = patients_df.loc[patients_df["Id"] == row["target_patient_id"]].iloc[0]

            # FIRST and LAST copied
            assert tgt["FIRST"] == src["FIRST"]
            assert tgt["LAST"] == src["LAST"]
            assert tgt["ADDRESS"] == src["ADDRESS"]

            # Birthdate offset 25-30 years
            src_bd = pd.Timestamp(src["BIRTHDATE"])
            tgt_bd = pd.Timestamp(tgt["BIRTHDATE"])
            year_diff = abs(tgt_bd.year - src_bd.year)
            assert 25 <= year_diff <= 30, f"Year diff {year_diff} not in [25, 30]"

    def test_ssn_preserved(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(
            type_weights={"twin": 0.0, "parent_child": 1.0, "sibling": 0.0},
            random_seed=7,
        )
        original = patients_df.copy()

        gen.generate(patients_df, pf, total_pairs=5)

        for pid in patients_df["Id"].values:
            orig_ssn = original.loc[original["Id"] == pid, "SSN"].values[0]
            new_ssn = patients_df.loc[patients_df["Id"] == pid, "SSN"].values[0]
            assert new_ssn == orig_ssn


@pytest.mark.unit
class TestSiblingType:
    """Test sibling demographic overwrite."""

    def test_correct_fields_and_birthdate_offset(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(
            type_weights={"twin": 0.0, "parent_child": 0.0, "sibling": 1.0},
            random_seed=7,
        )
        original = patients_df.copy()

        meta = gen.generate(patients_df, pf, total_pairs=5)
        assert (meta["group_type"] == "sibling").all()

        for _, row in meta.iterrows():
            src = original.loc[original["Id"] == row["source_patient_id"]].iloc[0]
            tgt = patients_df.loc[patients_df["Id"] == row["target_patient_id"]].iloc[0]

            # LAST and address copied
            assert tgt["LAST"] == src["LAST"]
            assert tgt["ADDRESS"] == src["ADDRESS"]
            assert tgt["CITY"] == src["CITY"]
            assert tgt["ZIP"] == src["ZIP"]

            # Birthdate offset 1-5 years
            src_bd = pd.Timestamp(src["BIRTHDATE"])
            tgt_bd = pd.Timestamp(tgt["BIRTHDATE"])
            year_diff = abs(tgt_bd.year - src_bd.year)
            assert 1 <= year_diff <= 5, f"Year diff {year_diff} not in [1, 5]"

    def test_first_different(self):
        patients_df = create_sample_patients(20)
        pf = _make_patient_facilities(patients_df)
        gen = ConfusableGroupGenerator(
            type_weights={"twin": 0.0, "parent_child": 0.0, "sibling": 1.0},
            random_seed=7,
        )
        original = patients_df.copy()

        meta = gen.generate(patients_df, pf, total_pairs=5)

        for _, row in meta.iterrows():
            orig_tgt = original.loc[original["Id"] == row["target_patient_id"]].iloc[0]
            new_tgt = patients_df.loc[
                patients_df["Id"] == row["target_patient_id"]
            ].iloc[0]
            # FIRST should NOT have been overwritten
            assert new_tgt["FIRST"] == orig_tgt["FIRST"]
