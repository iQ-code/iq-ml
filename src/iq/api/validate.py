def integer(number, min_value, max_value, name=None, b_return_repr=False):
    number = int(number)
    if number < min_value or number > max_value:
        if name is not None:
            raise ValueError(f'The parameter {name} is {number}, which is out of range: ({min_value}, {max_value})')
        else:
            raise ValueError(f'Number: {number} is out of range: ({min_value}, {max_value})')
    if b_return_repr:
        return repr(number)
    else:
        return number


def string(obj, name=None):
    if isinstance(obj, str):
        return obj
    else:
        if name is not None:
            raise ValueError(f"The parameter '{name}' must be a string. Got:\n{obj}")
        else:
            raise ValueError(f"Expected a string. Got:\n{obj}")


def real(number, min_value=None, max_value=None, name=None, b_return_repr=False):
    number = float(number)
    if min_value is not None and number < min_value:
        if name is not None:
            raise ValueError(f"The parameter {name} is {number}, which is less than its minimum value: {min_value}")
        else:
            raise ValueError(f"Number: {number} is less than its minimum value: {min_value}")
    if max_value is not None and number > max_value:
        if name is not None:
            raise ValueError(f"The parameter {name} is {number}, which is larger than its maximum value: {max_value}")
        else:
            raise ValueError(f"Number: {number} is larger than its maximum value: {max_value}")
    if b_return_repr:
        return repr(number)
    else:
        return number


def dictionary(obj, name=None, b_return_repr=False):
    if not isinstance(obj, dict):
        if name is not None:
            raise ValueError(f"The parameter '{name}' is must be a dictionary. Got:\n{obj}")
        else:
            raise ValueError(f"Expected a dictionary. Got:\n{obj}")
    obj = dict(obj)
    if b_return_repr:
        return repr(obj)
    else:
        return obj
