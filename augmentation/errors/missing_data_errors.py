"""Missing data error transformations."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base_error import BaseError


class MissingFieldValue(BaseError):
    """Set a field to empty/null, removing its evidence from linkage scoring.

    Applicable to non-name fields only — blanking names would break
    Splink blocking rules.  Returns np.nan for most fields (safe for
    both numeric and string dtypes) and pd.NaT for BIRTHDATE.
    """

    def get_applicable_fields(self) -> List[str]:
        return ["SSN", "BIRTHDATE", "ADDRESS", "CITY", "ZIP"]

    def get_error_type_name(self) -> str:
        return "missing_field_value"

    def should_apply(self, value: Any) -> bool:
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
        try:
            if np.isnan(value):
                return False
        except (TypeError, ValueError):
            pass
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        return True

    def apply(self, value: Any, context: Dict) -> Any:
        if not self.should_apply(value):
            return value

        field_name = context.get("field_name", "")

        if field_name == "BIRTHDATE":
            return pd.NaT

        return np.nan
