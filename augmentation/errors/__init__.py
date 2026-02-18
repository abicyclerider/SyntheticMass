"""Error transformation modules."""

from .base_error import BaseError
from .demographic_errors import (
    AddressAbbreviation,
    ApartmentFormatVariation,
    DateDigitTransposition,
    DateOffByOne,
    FullAddressChange,
    MaidenNameUsage,
    MultiCharacterNameTypo,
    NameTypo,
    NicknameSubstitution,
)
from .formatting_errors import (
    CapitalizationError,
    ExtraWhitespace,
    LeadingTrailingWhitespace,
    MissingWhitespace,
    SpecialCharacterVariation,
)
from .identifier_errors import (
    DriversLicenseError,
    PassportError,
    SSNDigitError,
    SSNFormatVariation,
    SSNTransposition,
)
from .missing_data_errors import MissingFieldValue

__all__ = [
    "BaseError",
    # Demographic
    "NicknameSubstitution",
    "NameTypo",
    "MultiCharacterNameTypo",
    "AddressAbbreviation",
    "ApartmentFormatVariation",
    "FullAddressChange",
    "DateOffByOne",
    "DateDigitTransposition",
    "MaidenNameUsage",
    # Identifier
    "SSNTransposition",
    "SSNDigitError",
    "SSNFormatVariation",
    "DriversLicenseError",
    "PassportError",
    # Formatting
    "CapitalizationError",
    "ExtraWhitespace",
    "MissingWhitespace",
    "LeadingTrailingWhitespace",
    "SpecialCharacterVariation",
    # Missing data
    "MissingFieldValue",
]
