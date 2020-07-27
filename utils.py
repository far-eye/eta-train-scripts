import os


def get_var(variable_name: object, default: object = None) -> object:
    """
    method to raise exception if some environment variable is not set
    :param variable_name: the environment variable being searched for
    :param default: default value of the environment var
    :return: the value of the variable
    """
    val = os.environ.get(variable_name, default)
    if val is None:
        raise EnvironmentError(
            'Please set the environment variable {0}'.format(variable_name))
    return val
