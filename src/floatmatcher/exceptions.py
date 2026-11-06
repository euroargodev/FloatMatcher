# floatmatcher/exceptions.py
# All library-specific errors live here, under a single base class so callers
# can catch everything from the library with `except FloatMatcherError`.
 
 
class FloatMatcherError(Exception):
    """Base class for all FloatMatcher errors

    Catch it to handle every error raised by the library.
    """
 
 
class ProfileFormatError(FloatMatcherError):
    """Raise when the points cannot be extracted from the given data

    A coordinate or variable is missing, or the structure is not recognized.
    """
 
class CDSRequestTooLarge(FloatMatcherError):
    """Raise when CDS refuses a request because it exceeds the cost limits

    Attributes: n_days, n_hours, n_variables, so the message can suggest a split or fewer hours.
    """