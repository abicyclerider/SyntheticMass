"""Error transformation modules."""

from .base_error import BaseError
from .demographic_errors import (
    NicknameSubstitution,
    NameTypo,
    AddressAbbreviation,
    ApartmentFormatVariation,
    DateOffByOne,
    MaidenNameUsage,
)
from .identifier_errors import (
    SSNTransposition,
    SSNDigitError,
    SSNFormatVariation,
    DriversLicenseError,
    PassportError,
)
from .formatting_errors import (
    CapitalizationError,
    ExtraWhitespace,
    MissingWhitespace,
    LeadingTrailingWhitespace,
    SpecialCharacterVariation,
)

__all__ = [
    "BaseError",
    # Demographic
    "NicknameSubstitution",
    "NameTypo",
    "AddressAbbreviation",
    "ApartmentFormatVariation",
    "DateOffByOne",
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
]
