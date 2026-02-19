"""Generate confusable patient groups (twins, siblings, namesakes).

Selects pairs of different Synthea patients sharing at least one facility,
then overwrites one patient's demographics to mimic a family relationship.
Clinical records remain untouched — only demographic fields are copied.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Fields copied per group type (target gets source's values for these)
_TWIN_FIELDS = [
    "LAST",
    "BIRTHDATE",
    "ADDRESS",
    "CITY",
    "ZIP",
    "GENDER",
    "STATE",
    "LAT",
    "LON",
]
_PARENT_CHILD_FIELDS = [
    "FIRST",
    "LAST",
    "ADDRESS",
    "CITY",
    "ZIP",
    "STATE",
    "LAT",
    "LON",
]
_SIBLING_FIELDS = [
    "LAST",
    "ADDRESS",
    "CITY",
    "ZIP",
    "STATE",
    "LAT",
    "LON",
]

GROUP_TYPES = ("twin", "parent_child", "sibling")


class ConfusableGroupGenerator:
    """Creates confusable patient pairs by copying demographics between patients."""

    def __init__(
        self,
        type_weights: Dict[str, float] | None = None,
        random_seed: int = 42,
    ):
        self.type_weights = type_weights or {
            "twin": 0.40,
            "parent_child": 0.30,
            "sibling": 0.30,
        }
        self.rng = np.random.default_rng(random_seed)

    def generate(
        self,
        patients_df: pd.DataFrame,
        patient_facilities: Dict[str, List[int]],
        total_pairs: int,
    ) -> pd.DataFrame:
        """Mutate *patients_df* in-place and return confusable-pair metadata.

        Returns an empty DataFrame with the correct schema when total_pairs=0
        or when there are insufficient eligible patients.
        """
        meta_cols = [
            "source_patient_id",
            "target_patient_id",
            "group_type",
            "fields_copied",
        ]
        if total_pairs <= 0:
            return pd.DataFrame(columns=meta_cols)

        pairs = self._select_eligible_pairs(
            patients_df, patient_facilities, total_pairs
        )
        if not pairs:
            return pd.DataFrame(columns=meta_cols)

        # Assign group types to pairs
        types = list(self.type_weights.keys())
        weights = np.array([self.type_weights[t] for t in types])
        weights = weights / weights.sum()
        assigned_types = self.rng.choice(types, size=len(pairs), p=weights)

        records: list[dict] = []
        for (src_id, tgt_id), group_type in zip(pairs, assigned_types):
            if group_type == "twin":
                fields = self._apply_twin(patients_df, src_id, tgt_id)
            elif group_type == "parent_child":
                fields = self._apply_parent_child(patients_df, src_id, tgt_id)
            elif group_type == "sibling":
                fields = self._apply_sibling(patients_df, src_id, tgt_id)
            else:
                raise ValueError(f"Unknown group type: {group_type}")

            records.append(
                {
                    "source_patient_id": src_id,
                    "target_patient_id": tgt_id,
                    "group_type": group_type,
                    "fields_copied": ",".join(fields),
                }
            )

        return pd.DataFrame(records, columns=meta_cols)

    # ── Pair selection ────────────────────────────────────────────────

    def _select_eligible_pairs(
        self,
        patients_df: pd.DataFrame,
        patient_facilities: Dict[str, List[int]],
        n_pairs: int,
    ) -> List[Tuple[str, str]]:
        """Greedy selection of patient pairs sharing at least one facility.

        Each patient appears in at most one pair.
        """
        # Build facility→patients index
        fac_to_patients: Dict[int, List[str]] = {}
        patient_ids_in_df = set(patients_df["Id"].values)
        for pid, fids in patient_facilities.items():
            if pid not in patient_ids_in_df:
                continue
            for fid in fids:
                fac_to_patients.setdefault(fid, []).append(pid)

        # Collect candidate pairs from each facility
        candidate_set: set[Tuple[str, str]] = set()
        for pids in fac_to_patients.values():
            if len(pids) < 2:
                continue
            # Only sample a manageable number per facility
            arr = np.array(pids)
            self.rng.shuffle(arr)
            limit = min(len(arr), max(20, n_pairs * 2))
            for i in range(0, limit - 1, 2):
                a, b = arr[i], arr[i + 1]
                pair = (a, b) if a < b else (b, a)
                candidate_set.add(pair)

        # Shuffle and greedily pick, ensuring no patient reuse
        candidates = list(candidate_set)
        self.rng.shuffle(candidates)

        used: set[str] = set()
        pairs: List[Tuple[str, str]] = []
        for a, b in candidates:
            if a in used or b in used:
                continue
            pairs.append((a, b))
            used.add(a)
            used.add(b)
            if len(pairs) >= n_pairs:
                break

        return pairs

    # ── Type-specific demographic overwrites ──────────────────────────

    def _apply_twin(
        self, patients_df: pd.DataFrame, src_id: str, tgt_id: str
    ) -> List[str]:
        """Twin: copy LAST, BIRTHDATE, ADDRESS, CITY, ZIP, GENDER, STATE, LAT, LON."""
        src_mask = patients_df["Id"] == src_id
        tgt_mask = patients_df["Id"] == tgt_id

        for field in _TWIN_FIELDS:
            if field in patients_df.columns:
                patients_df.loc[tgt_mask, field] = patients_df.loc[
                    src_mask, field
                ].values[0]

        return _TWIN_FIELDS

    def _apply_parent_child(
        self, patients_df: pd.DataFrame, src_id: str, tgt_id: str
    ) -> List[str]:
        """Parent-child namesake: copy FIRST, LAST, ADDRESS etc; offset DOB 25-30y."""
        src_mask = patients_df["Id"] == src_id
        tgt_mask = patients_df["Id"] == tgt_id

        for field in _PARENT_CHILD_FIELDS:
            if field in patients_df.columns:
                patients_df.loc[tgt_mask, field] = patients_df.loc[
                    src_mask, field
                ].values[0]

        # Offset birthdate by 25-30 years
        if "BIRTHDATE" in patients_df.columns:
            src_bd = patients_df.loc[src_mask, "BIRTHDATE"].values[0]
            offset_years = int(self.rng.integers(25, 31))
            src_bd = pd.Timestamp(src_bd)
            try:
                new_bd = src_bd.replace(year=src_bd.year + offset_years)
            except ValueError:
                # Feb 29 edge case
                new_bd = src_bd.replace(
                    year=src_bd.year + offset_years, month=2, day=28
                )
            patients_df.loc[tgt_mask, "BIRTHDATE"] = new_bd

        return _PARENT_CHILD_FIELDS + ["BIRTHDATE"]

    def _apply_sibling(
        self, patients_df: pd.DataFrame, src_id: str, tgt_id: str
    ) -> List[str]:
        """Sibling: copy LAST, ADDRESS, CITY, ZIP, STATE, LAT, LON; offset DOB 1-5y."""
        src_mask = patients_df["Id"] == src_id
        tgt_mask = patients_df["Id"] == tgt_id

        for field in _SIBLING_FIELDS:
            if field in patients_df.columns:
                patients_df.loc[tgt_mask, field] = patients_df.loc[
                    src_mask, field
                ].values[0]

        # Offset birthdate by 1-5 years
        if "BIRTHDATE" in patients_df.columns:
            src_bd = patients_df.loc[src_mask, "BIRTHDATE"].values[0]
            offset_years = int(self.rng.integers(1, 6))
            sign = self.rng.choice([-1, 1])
            src_bd = pd.Timestamp(src_bd)
            try:
                new_bd = src_bd.replace(year=src_bd.year + sign * offset_years)
            except ValueError:
                new_bd = src_bd.replace(
                    year=src_bd.year + sign * offset_years, month=2, day=28
                )
            patients_df.loc[tgt_mask, "BIRTHDATE"] = new_bd

        return _SIBLING_FIELDS + ["BIRTHDATE"]
