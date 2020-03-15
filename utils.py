# Dropout RNN training utilities

def filter_parameters(param_gen, prefix, postfix):
    """Filter parameters by names
    -Args:
       param_gen: named_parameters
       prefix: prefix of parameter names
       postfix: postfix of parameter names"""

    filtered = filter(lambda named_params: named_params[0].startswith(prefix), param_gen)
    filtered = filter(lambda named_params: named_params[0].endswith(postfix), filtered)
    return map(lambda named_params: named_params[1], filtered)


def weight_coefficient(length_scale, precision, dropout_rate, N):
    """Calculate the coefficient of dropout rnn layer weight"""
    return 0.5 * length_scale**2 * dropout_rate / precision / N


def bias_coefficient(length_scale, precision, N):
    """Calculate the coefficient of dropout rnn layer bias"""
    return 0.5 * length_scale**2 / precision / N
