import numpy as np
import pandas as pd


def get_region(row:str) -> str:
    if type(row) == str:
        arr = row.split(',')
        for elem_idx in range(len(arr)):
            if arr[elem_idx].strip() == 'Россия':
                res = arr[elem_idx-1].strip()
                if type(res) == str:
                    return res
                    break
                else:
                    return np.nan
    else:
        return row
    

def get_city(row:str) -> str:
    if type(row) == str:
        return row.split(',')[2].strip()
    else:
        return row