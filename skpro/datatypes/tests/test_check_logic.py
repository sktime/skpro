from skpro.datatypes._table._check import _check_list_of_dict_table


def test_check_list_of_dict_table_detects_invalid_types():
    invalid_data = [{"a": 1}, {"a": [99]}]

    check_result, msg, _ = _check_list_of_dict_table(invalid_data, return_metadata=True)

    assert check_result is False
    assert "not a primitive type" in msg