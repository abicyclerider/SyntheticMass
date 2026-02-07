"""
Medical record summarizer for LLM-based entity resolution.

Condenses raw clinical records into a structured text summary per patient.
Designed for MedGemma 4B's limited context window — observations (avg 418 rows)
and encounters (avg 35 rows) are aggregated, while conditions and medications
are listed in full since they're compact and highly discriminating.
"""

import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def summarize_patient_records(patient_id: str, facility_id: str,
                               medical_records: dict[str, pd.DataFrame]) -> str:
    """
    Create a structured text summary of a patient's medical history.

    Includes all record types. DSPy optimization determines which
    sections are most useful via instruction tuning.

    Args:
        patient_id: Patient UUID (PATIENT column in Synthea CSVs)
        facility_id: Facility identifier (e.g., 'facility_001')
        medical_records: Dict from shared.medical_records.load_medical_records()

    Returns:
        Structured text summary of the patient's medical history
    """
    sections = []
    sections.append("=== MEDICAL HISTORY ===")

    # Filter records for this patient at this facility
    def get_patient_df(record_type: str) -> Optional[pd.DataFrame]:
        df = medical_records.get(record_type)
        if df is None:
            return None
        mask = (df['PATIENT'] == patient_id) & (df['facility_id'] == facility_id)
        result = df[mask]
        return result if not result.empty else None

    # CONDITIONS
    conditions_df = get_patient_df('conditions')
    sections.append(_summarize_conditions(conditions_df))

    # MEDICATIONS
    medications_df = get_patient_df('medications')
    sections.append(_summarize_medications(medications_df))

    # ENCOUNTERS
    encounters_df = get_patient_df('encounters')
    sections.append(_summarize_encounters(encounters_df))

    # OBSERVATIONS (key vitals + labs only)
    observations_df = get_patient_df('observations')
    sections.append(_summarize_observations(observations_df))

    # PROCEDURES
    procedures_df = get_patient_df('procedures')
    sections.append(_summarize_procedures(procedures_df))

    # IMMUNIZATIONS
    immunizations_df = get_patient_df('immunizations')
    sections.append(_summarize_immunizations(immunizations_df))

    # ALLERGIES
    allergies_df = get_patient_df('allergies')
    sections.append(_summarize_allergies(allergies_df))

    # IMAGING
    imaging_df = get_patient_df('imaging_studies')
    sections.append(_summarize_imaging(imaging_df))

    # DEVICES
    devices_df = get_patient_df('devices')
    sections.append(_summarize_devices(devices_df))

    # CARE PLANS
    careplans_df = get_patient_df('careplans')
    sections.append(_summarize_careplans(careplans_df))

    return "\n\n".join(s for s in sections if s)


def _summarize_conditions(df: Optional[pd.DataFrame]) -> str:
    """Summarize conditions with onset dates and active/resolved status."""
    if df is None:
        return "CONDITIONS: none"

    lines = ["CONDITIONS (active/historical):"]
    df = df.sort_values('START')

    for _, row in df.iterrows():
        desc = row.get('DESCRIPTION', 'Unknown')
        start = str(row.get('START', ''))[:10]
        stop = row.get('STOP', '')
        if pd.isna(stop) or stop == '':
            status = "ongoing"
        else:
            status = f"resolved {str(stop)[:10]}"
        lines.append(f"- {desc} (onset: {start}, {status})")

    return "\n".join(lines)


def _summarize_medications(df: Optional[pd.DataFrame]) -> str:
    """Summarize medications with dates and reasons."""
    if df is None:
        return "MEDICATIONS: none"

    lines = ["MEDICATIONS (current/past):"]
    df = df.sort_values('START')

    for _, row in df.iterrows():
        desc = row.get('DESCRIPTION', 'Unknown')
        start = str(row.get('START', ''))[:10]
        stop = row.get('STOP', '')
        reason = row.get('REASONDESCRIPTION', '')

        if pd.isna(stop) or stop == '':
            period = f"{start} to present"
        else:
            period = f"{start} to {str(stop)[:10]}"

        reason_str = f" for {reason}" if pd.notna(reason) and reason else ""
        lines.append(f"- {desc} ({period}){reason_str}")

    return "\n".join(lines)


def _summarize_encounters(df: Optional[pd.DataFrame]) -> str:
    """Aggregate encounters by type and time period."""
    if df is None:
        return "ENCOUNTERS: none"

    lines = ["ENCOUNTERS (summarized):"]

    # Parse dates
    df = df.copy()
    df['start_dt'] = pd.to_datetime(df['START'], errors='coerce')

    # Count by encounter class
    class_counts = df['ENCOUNTERCLASS'].value_counts()
    for enc_class, count in class_counts.items():
        class_df = df[df['ENCOUNTERCLASS'] == enc_class]
        min_year = class_df['start_dt'].min()
        max_year = class_df['start_dt'].max()
        if pd.notna(min_year) and pd.notna(max_year):
            year_range = f"{min_year.year}-{max_year.year}"
        else:
            year_range = "unknown dates"
        lines.append(f"- {count} {enc_class} visits ({year_range})")

    # Note recent encounters with reasons
    recent = df.nlargest(3, 'start_dt')
    if not recent.empty:
        lines.append("Recent:")
        for _, row in recent.iterrows():
            date = str(row.get('START', ''))[:10]
            enc_class = row.get('ENCOUNTERCLASS', '')
            reason = row.get('REASONDESCRIPTION', '')
            reason_str = f" — {reason}" if pd.notna(reason) and reason else ""
            lines.append(f"  - {date} {enc_class}{reason_str}")

    return "\n".join(lines)


def _summarize_observations(df: Optional[pd.DataFrame]) -> str:
    """Summarize to latest vitals + key lab values (not hundreds of raw rows)."""
    if df is None:
        return "KEY OBSERVATIONS: none"

    lines = ["KEY OBSERVATIONS:"]

    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['DATE'], errors='coerce')

    # Key vital signs and labs to extract (by DESCRIPTION or CODE)
    key_obs = [
        'Body Height', 'Body Weight', 'Body Mass Index',
        'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Hemoglobin A1c/Hemoglobin.total in Blood',
        'Glucose', 'Total Cholesterol',
        'Heart rate', 'Respiratory rate',
    ]

    for obs_name in key_obs:
        obs_df = df[df['DESCRIPTION'].str.contains(obs_name, case=False, na=False)]
        if obs_df.empty:
            continue
        # Get most recent
        latest = obs_df.sort_values('date_dt').iloc[-1]
        value = latest.get('VALUE', '')
        units = latest.get('UNITS', '')
        date = str(latest.get('DATE', ''))[:10]
        if pd.notna(value) and value != '':
            units_str = f" {units}" if pd.notna(units) and units else ""
            lines.append(f"- {obs_name}: {value}{units_str} ({date})")

    # Count total observations for context
    lines.append(f"- Total observations on file: {len(df)}")

    return "\n".join(lines)


def _summarize_procedures(df: Optional[pd.DataFrame]) -> str:
    """List procedures with dates."""
    if df is None:
        return "PROCEDURES: none"

    lines = ["PROCEDURES:"]
    df = df.sort_values('START')

    # If many procedures, just list unique descriptions with counts
    if len(df) > 15:
        desc_counts = df['DESCRIPTION'].value_counts()
        for desc, count in desc_counts.head(15).items():
            lines.append(f"- {desc} (x{count})")
        if len(desc_counts) > 15:
            lines.append(f"  ... and {len(desc_counts) - 15} more procedure types")
    else:
        for _, row in df.iterrows():
            desc = row.get('DESCRIPTION', 'Unknown')
            date = str(row.get('START', ''))[:10]
            lines.append(f"- {desc} ({date})")

    return "\n".join(lines)


def _summarize_immunizations(df: Optional[pd.DataFrame]) -> str:
    """Compact immunization list."""
    if df is None:
        return "IMMUNIZATIONS: none"

    desc_counts = df['DESCRIPTION'].value_counts()
    items = [f"{desc} (x{count})" for desc, count in desc_counts.items()]
    return "IMMUNIZATIONS: " + ", ".join(items)


def _summarize_allergies(df: Optional[pd.DataFrame]) -> str:
    """List allergies with reactions."""
    if df is None:
        return "ALLERGIES: none"

    lines = ["ALLERGIES:"]
    for _, row in df.iterrows():
        desc = row.get('DESCRIPTION', 'Unknown')
        reaction = row.get('DESCRIPTION1', '')
        severity = row.get('SEVERITY1', '')

        extras = []
        if pd.notna(reaction) and reaction:
            extras.append(str(reaction))
        if pd.notna(severity) and severity:
            extras.append(str(severity))
        extra_str = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {desc}{extra_str}")

    return "\n".join(lines)


def _summarize_imaging(df: Optional[pd.DataFrame]) -> str:
    """Aggregate imaging studies by body site and modality."""
    if df is None:
        return "IMAGING: none"

    # Group by body site + modality
    if 'BODYSITE_DESCRIPTION' in df.columns and 'MODALITY_DESCRIPTION' in df.columns:
        grouped = df.groupby(['BODYSITE_DESCRIPTION', 'MODALITY_DESCRIPTION']).size()
        items = [f"{count} {bodysite} {modality}" for (bodysite, modality), count in grouped.items()]
        return "IMAGING: " + ", ".join(items)

    return f"IMAGING: {len(df)} studies"


def _summarize_devices(df: Optional[pd.DataFrame]) -> str:
    """List medical devices."""
    if df is None:
        return "DEVICES: none"

    descs = df['DESCRIPTION'].unique()
    return "DEVICES: " + ", ".join(str(d) for d in descs)


def _summarize_careplans(df: Optional[pd.DataFrame]) -> str:
    """List active care plans."""
    if df is None:
        return "CARE PLANS: none"

    descs = df['DESCRIPTION'].unique()
    return "CARE PLANS: " + ", ".join(str(d) for d in descs)
