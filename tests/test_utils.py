import pytest

from tinygpt.utils import DType, parse_value


def test_DType():

    # cast() method
    for value in [-2, -1, 0, 1, 2]:
        assert DType.bool.cast(value) == (value != 0)
        assert DType.float32.cast(value) == float(value)
        assert DType.float32.cast((value != 0)) == float((value != 0))
        assert DType.int32.cast(value) == int(value)
        assert DType.int32.cast((value != 0)) == int((value != 0))

    # deduce_dtype() function
    for int_val in range(-5, 5):
        assert DType.deduce_dtype(float(int_val)) is DType.float32

    for int_val in range(-5, 5):
        assert DType.deduce_dtype(int_val) is DType.int32

    for bool_val in [True, False]:
        assert DType.deduce_dtype(bool_val) is DType.bool

    for not_valid_input in [[], (), None]:
        with pytest.raises(RuntimeError, match="Could not infer dtype of type"):
            DType.deduce_dtype(not_valid_input)


def test_parse_value():
    # Ints
    for org_val in [-2, -1, 0, 1, 2]:
        value = parse_value(f"{org_val}")
        assert isinstance(value, int), "Failed on int"
        assert value == org_val, "Value mismatch on int"

    # Floats, including scientific notation
    for org_val in [-2.5, -1.0, 0.0, 1.5, 2.1, 3.5e-2, -1.2e3]:
        value = parse_value(f"{org_val}")
        assert isinstance(value, float), "Failed on float"
        assert value == org_val, "Value mismatch on float"

    # Bools
    for org_val, str_val in zip([True, False], ["True", "False"]):
        value = parse_value(str_val)
        assert isinstance(value, bool), "Failed on bool"
        assert value is org_val, "Value mismatch on bool"

    # Empty list
    value = parse_value("[]")
    assert isinstance(value, list), "Failed on empty list"
    assert value == [], "Value mismatch on empty list"

    # Nested lists of each value, including empty nested list
    nested_lists = [
        "[1, 2, [3, 4]]",
        "[[True, False], []]",
        "[[], [[-1.2, 2.3], [3.45e-2]]]",
        "[[], []]",
        "[[True], [False, [True]], [[[]]]]"
    ]
    expected_values = [
        [1, 2, [3, 4]],
        [[True, False], []],
        [[], [[-1.2, 2.3], [3.45e-2]]],
        [[], []],
        [[True], [False, [True]], [[[]]]]
    ]

    for str_val, exp_val in zip(nested_lists, expected_values):
        value = parse_value(str_val)
        assert isinstance(value, list), "Failed on nested list"
        assert value == exp_val, "Value mismatch on nested list"