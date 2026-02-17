"""Facility assignment logic for patients and encounters."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import FacilityDistributionConfig


class FacilityAssigner:
    """Assigns patients to facilities and distributes encounters chronologically."""

    def __init__(self, config: FacilityDistributionConfig, random_seed: int = 42):
        """
        Initialize facility assigner.

        Args:
            config: Facility distribution configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.default_rng(random_seed)

        # Prepare facility count distribution
        self._prepare_facility_count_distribution()

    def _prepare_facility_count_distribution(self) -> None:
        """Prepare arrays for weighted random sampling of facility counts."""
        weights = self.config.facility_count_weights
        self.facility_counts = list(weights.keys())
        self.facility_count_probabilities = list(weights.values())

    def assign_patients_to_facilities(
        self,
        patients_df: pd.DataFrame,
        encounters_df: pd.DataFrame,
    ) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
        """
        Assign each patient to 1-5+ facilities and distribute encounters chronologically.

        Args:
            patients_df: DataFrame with patient records (must have 'Id' column)
            encounters_df: DataFrame with encounter records (must have 'Id', 'PATIENT', 'START' columns)

        Returns:
            Tuple of:
                - patient_facilities: Dict[patient_uuid -> List[facility_ids]]
                - encounter_facilities: Dict[encounter_uuid -> facility_id]
        """
        patient_facilities = {}
        encounter_facilities = {}

        # Pre-group encounters by patient (O(n_encounters) once, vs O(n_patients × n_encounters))
        sorted_enc = encounters_df.sort_values("START")
        encounters_by_patient: Dict[str, List[str]] = {}
        for patient_id, enc_id in zip(sorted_enc["PATIENT"], sorted_enc["Id"]):
            encounters_by_patient.setdefault(patient_id, []).append(enc_id)

        patient_ids = patients_df["Id"].values
        max_available = self.config.num_facilities
        facility_range = np.arange(1, self.config.num_facilities + 1)

        # Vectorized facility-count selection for all patients at once
        num_facilities_arr = self.rng.choice(
            self.facility_counts,
            size=len(patient_ids),
            p=self.facility_count_probabilities,
        )
        num_facilities_arr = np.minimum(num_facilities_arr, max_available)

        for i, patient_uuid in enumerate(patient_ids):
            num_facilities = int(num_facilities_arr[i])

            # Select specific facility IDs
            assigned_facilities = self.rng.choice(
                facility_range,
                size=num_facilities,
                replace=False,
            ).tolist()

            patient_facilities[patient_uuid] = assigned_facilities

            # Look up pre-sorted encounters for this patient (O(1) dict lookup)
            enc_ids = encounters_by_patient.get(patient_uuid)
            if not enc_ids:
                continue

            # Distribute encounters across facilities
            encounter_distribution = self._distribute_encounters_chronologically(
                enc_ids, assigned_facilities
            )
            encounter_facilities.update(encounter_distribution)

        return patient_facilities, encounter_facilities

    def _distribute_encounters_chronologically(
        self, encounter_ids: List[str], assigned_facilities: List[int]
    ) -> Dict[str, int]:
        """
        Distribute encounters across facilities with chronological switching pattern.

        Args:
            encounter_ids: List of encounter UUIDs (sorted chronologically)
            assigned_facilities: List of facility IDs for this patient

        Returns:
            Dict mapping encounter_uuid -> facility_id
        """
        num_facilities = len(assigned_facilities)
        num_encounters = len(encounter_ids)

        if num_facilities == 1:
            # All encounters go to single facility
            return {enc_id: assigned_facilities[0] for enc_id in encounter_ids}

        # Multi-facility case: split with primary facility bias
        primary_facility = assigned_facilities[0]
        secondary_facilities = assigned_facilities[1:]

        # Calculate split point based on primary_facility_weight
        primary_count = int(num_encounters * self.config.primary_facility_weight)

        distribution = {}

        # First 60% (configurable) go to primary facility
        for i in range(primary_count):
            distribution[encounter_ids[i]] = primary_facility

        # Remaining encounters distributed chronologically across secondary facilities
        remaining_encounters = encounter_ids[primary_count:]

        for i, enc_id in enumerate(remaining_encounters):
            # Round-robin through secondary facilities
            facility_idx = i % len(secondary_facilities)
            distribution[enc_id] = secondary_facilities[facility_idx]

        return distribution

    def get_facility_statistics(
        self,
        patient_facilities: Dict[str, List[int]],
        encounter_facilities: Dict[str, int],
    ) -> Dict:
        """
        Generate statistics about the facility distribution.

        Args:
            patient_facilities: Patient to facilities mapping
            encounter_facilities: Encounter to facility mapping

        Returns:
            Dictionary with distribution statistics
        """
        # Count patients by number of facilities
        facility_count_distribution = {}
        for facilities in patient_facilities.values():
            count = len(facilities)
            facility_count_distribution[count] = (
                facility_count_distribution.get(count, 0) + 1
            )

        # Count encounters per facility
        encounter_count_per_facility = {}
        for facility_id in encounter_facilities.values():
            encounter_count_per_facility[facility_id] = (
                encounter_count_per_facility.get(facility_id, 0) + 1
            )

        # Count unique patients per facility
        patient_count_per_facility = {}
        for patient_uuid, facilities in patient_facilities.items():
            for facility_id in facilities:
                if facility_id not in patient_count_per_facility:
                    patient_count_per_facility[facility_id] = 0
                patient_count_per_facility[facility_id] += 1

        return {
            "total_patients": len(patient_facilities),
            "total_encounters": len(encounter_facilities),
            "facility_count_distribution": facility_count_distribution,
            "encounter_count_per_facility": encounter_count_per_facility,
            "patient_count_per_facility": patient_count_per_facility,
        }
