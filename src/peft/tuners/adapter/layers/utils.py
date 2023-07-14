from contextlib import contextmanager
import warnings

def to_power_of_2(n: int):
    """
    As of pytorch 2.0.1 ffts for 16 bit are only supported for powers of 2.
    This convenience method returns the smallest power of 2 >= n.
    """
    return 1<<(n-1).bit_length()



@contextmanager
def AdviceFormat():
    old_format = warnings.formatwarning
    try:
        def custom_format(message, category, filename, lineno, line=None):
            return f'{filename}:{lineno} {category.__name__}: {message}\n'
        warnings.formatwarning = custom_format
        yield
    finally:
        warnings.formatwarning = old_format
    