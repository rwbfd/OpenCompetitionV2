# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "ind_cols.h":
    vector[long] get_ind_cols(double*, const long, const long)

def get_ind_col(df):
    matrix = np.concatenate(np.ones(df.shape[0]), df)

    cdef np.ndarray[double, ndim=2, mode='fortran'] arg= np.asfortranarray(matrix, type=np.float64)

    ind_col_vec = get_ind_cols(&arg[0,0], df.shape[0], df.shape[1])

    result = list()
    for i in ind_col_vec:
        result.append(df.columns[i-1])
    return result