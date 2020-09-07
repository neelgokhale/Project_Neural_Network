"""
Created by Neel Gokhale at 2020-09-07
File logging.py from project Project_NN_From_Scratch
Built using PyCharm

"""

from functools import wraps


def my_logger(original_function):
    import logging
    from datetime import datetime
    logging.basicConfig(filename=f'assets/logs/{original_function.__name__}.log', level=logging.INFO)

    @wraps(original_function)
    def wrapper(*args, **kwargs):
        logging.info(f'<{datetime.now().strftime("%H:%M:%S")}> Ran with args: {args} and kwargs: {kwargs}')
        return original_function(*args, **kwargs)

    return wrapper


def my_timer(original_function):
    import time

    @wraps(original_function)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = original_function(*args, **kwargs)
        t2 = time.time() - t1
        print(f'{original_function.__name__} ran in {t2} seconds')
        return result
    return wrapper
