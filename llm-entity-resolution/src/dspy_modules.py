"""
DSPy modules for medical record entity resolution.

Defines the signature and module for comparing two patients' medical histories
to determine if they are the same person — using only clinical data, no demographics.
"""

import dspy


class MedicalRecordMatchSignature(dspy.Signature):
    """Given medical histories from two patients at different facilities,
    determine if they are the same person based solely on clinical patterns.

    Focus on chronic condition overlap, medication continuity, encounter timing,
    and distinctive clinical features. Ignore any demographic information."""

    medical_history_a: str = dspy.InputField(
        desc="Structured medical history summary for Patient A")
    medical_history_b: str = dspy.InputField(
        desc="Structured medical history summary for Patient B")

    reasoning: str = dspy.OutputField(
        desc="Clinical reasoning for the match decision")
    is_match: bool = dspy.OutputField(
        desc="True if the two histories belong to the same patient, False otherwise")
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 (uncertain) to 1.0 (certain)")


class MedicalRecordMatcher(dspy.Module):
    """DSPy module that uses ChainOfThought to compare medical histories."""

    def __init__(self):
        super().__init__()
        self.matcher = dspy.ChainOfThought(MedicalRecordMatchSignature)

    def forward(self, medical_history_a, medical_history_b):
        return self.matcher(
            medical_history_a=medical_history_a,
            medical_history_b=medical_history_b,
        )
