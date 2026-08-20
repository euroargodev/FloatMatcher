# tests/test_method.py
import pytest

from floatmatcher.method import MatchupMethod, Constraints


def test_cannot_instantiate_abstract_method():
    """The abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MatchupMethod()


def test_incomplete_subclass_cannot_instantiate():
    """A subclass that does not implement match() is still abstract."""
    class Incomplete(MatchupMethod):
        pass                      # forgot to implement match

    with pytest.raises(TypeError):
        Incomplete()


def test_complete_subclass_instantiates():
    """A subclass implementing match() can be instantiated."""
    class Complete(MatchupMethod):
        def match(self, grid, points, constraints):
            return None           # dummy, just to satisfy the contract

    Complete()                    # should not raise


def test_constraints_defaults():
    """Constraints carry sensible defaults."""
    c = Constraints()
    assert c.max_dist_km == 100.0
    assert c.max_time_days == 1.0