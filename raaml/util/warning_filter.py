import warnings
from contextlib import contextmanager
from sklearn.exceptions import ConvergenceWarning

@contextmanager
def filter_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield  # run the code inside the with-block

    
"""@contextmanager
def filter_warnings():
    
    original_filters = warnings.filters[:]

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    yield
    
    warnings.filters = original_filters"""
        
