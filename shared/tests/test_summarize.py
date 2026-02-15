"""Unit tests for shared/summarize.py — patient summary generation."""

import pandas as pd
import pytest

from shared.summarize import summarize_diff_friendly_from_records


@pytest.mark.unit
class TestSummarizeDiffFriendly:
    def test_conditions_grouped_by_year(self):
        """CONDITIONS section groups conditions by onset year."""
        records = {
            "conditions": pd.DataFrame(
                {
                    "START": ["2020-01-15", "2020-06-01", "2021-03-10"],
                    "STOP": ["2020-02-15", "2020-07-01", pd.NaT],
                    "DESCRIPTION": ["Flu", "Sprain", "Diabetes"],
                }
            )
        }
        result = summarize_diff_friendly_from_records(records)
        assert "CONDITIONS:" in result
        assert "2020:" in result
        assert "2021:" in result
        # Ongoing condition gets asterisk
        assert "Diabetes *" in result

    def test_medications_ongoing(self):
        """Drug with no STOP shows as 'ongoing'."""
        records = {
            "medications": pd.DataFrame(
                {
                    "START": ["2020-01-01"],
                    "STOP": [pd.NaT],
                    "DESCRIPTION": ["Metformin"],
                }
            )
        }
        result = summarize_diff_friendly_from_records(records)
        assert "MEDICATIONS:" in result
        assert "ongoing" in result
        assert "Metformin" in result

    def test_allergies_sorted(self):
        """Allergies are listed in alphabetical order."""
        records = {
            "allergies": pd.DataFrame(
                {
                    "DESCRIPTION": ["Peanuts", "Dust", "Latex", "Dust"],
                }
            )
        }
        result = summarize_diff_friendly_from_records(records)
        assert "ALLERGIES:" in result
        # Alphabetical: Dust, Latex, Peanuts
        allergy_line = [line for line in result.split("\n") if "ALLERGIES:" in line][0]
        items = allergy_line.replace("ALLERGIES: ", "").split("; ")
        assert items == sorted(items)

    def test_observations_latest_two(self):
        """Only the last 2 values per metric are shown."""
        records = {
            "observations": pd.DataFrame(
                {
                    "DATE": ["2019-01-01", "2020-01-01", "2021-01-01"],
                    "DESCRIPTION": ["Body Height", "Body Height", "Body Height"],
                    "VALUE": ["170", "171", "172"],
                    "UNITS": ["cm", "cm", "cm"],
                }
            )
        }
        result = summarize_diff_friendly_from_records(records)
        assert "OBSERVATIONS:" in result
        # Should have 2021 and 2020 values, not 2019
        assert "172" in result
        assert "171" in result
        assert "170" not in result

    def test_empty_records(self):
        """No data at all -> 'No clinical records available.'"""
        result = summarize_diff_friendly_from_records({})
        assert result == "No clinical records available."

    def test_only_some_sections(self):
        """Records with only conditions produce only a CONDITIONS section."""
        records = {
            "conditions": pd.DataFrame(
                {
                    "START": ["2020-01-01"],
                    "STOP": ["2020-02-01"],
                    "DESCRIPTION": ["Flu"],
                }
            )
        }
        result = summarize_diff_friendly_from_records(records)
        assert "CONDITIONS:" in result
        assert "MEDICATIONS:" not in result
        assert "ALLERGIES:" not in result
        assert "OBSERVATIONS:" not in result
