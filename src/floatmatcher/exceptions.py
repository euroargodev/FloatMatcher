# floatmatcher/exceptions.py
# All library-specific errors live here, under a single base class so callers
# can catch everything from the library with `except FloatMatcherError`.
 
 
class FloatMatcherError(Exception):
    """Base class for all FloatMatcher errors."""
 
 
class ProfileFormatError(FloatMatcherError):
    """Raised when point data cannot be extracted from the given source
    (missing coordinate, unrecognized structure)."""
 
