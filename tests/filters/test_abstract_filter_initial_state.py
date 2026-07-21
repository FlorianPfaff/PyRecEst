import unittest

from pyrecest.filters.abstract_filter import AbstractFilter


class _MutableState:
    def __init__(self):
        self.values = [1.0]

    def mean(self):
        return self.values

    @property
    def dim(self):
        return len(self.values)


class _ConcreteFilter(AbstractFilter):
    def __init__(self, initial_filter_state):
        super().__init__(initial_filter_state)


class TestAbstractFilterInitialState(unittest.TestCase):
    def test_constructor_copies_initial_filter_state(self):
        initial_state = _MutableState()

        filter_under_test = _ConcreteFilter(initial_state)
        initial_state.values.append(2.0)

        self.assertIsNot(filter_under_test.filter_state, initial_state)
        self.assertEqual(filter_under_test.filter_state.values, [1.0])


if __name__ == "__main__":
    unittest.main()
