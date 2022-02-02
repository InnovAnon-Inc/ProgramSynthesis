#! /usr/bin/env python3

import numpy as np

def is_in_list(array_to_check, list_np_arrays): return np.any(np.all(array_to_check == list_np_arrays))

