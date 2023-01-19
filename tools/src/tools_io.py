"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the tool .py file including toolkits for vtu files

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
import numpy as np
import vtk
from .vtktools import *


def get_clean_vtu_file(filename):
    "Removes fields and arrays from a vtk file, leaving the coordinates/connectivity information."
    vtu_data = vtu(filename)
    clean_vtu = vtu()
    clean_vtu.ugrid.DeepCopy(vtu_data.ugrid)
    fieldNames = clean_vtu.GetFieldNames()
    # remove all fields and arrays from this vtu
    for field in fieldNames:
        clean_vtu.RemoveField(field)
        fieldNames = clean_vtu.GetFieldNames()
        vtkdata = clean_vtu.ugrid.GetCellData()
        arrayNames = [
            vtkdata.GetArrayName(i) for i in range(
                vtkdata.GetNumberOfArrays())]
    for array in arrayNames:
        vtkdata.RemoveArray(array)
    return clean_vtu


# read in vtus -----------------------------------------------------------

def get_nNodes_from_vtu(filename):

    # assuming the simulations are run on a fixed mesh, so the number of nodes
    # is the same for each time level
    representative_vtu = vtu(filename)
    coordinates = representative_vtu.GetLocations()
    nNodes = coordinates.shape[0]

    return nNodes


def get_POD_functions(snapshots_matrix, nPOD, cumulative_tol, nNodes):

    nrows, ncols = snapshots_matrix.shape

    if nrows > ncols:
        SSmatrix = np.dot(snapshots_matrix.T, snapshots_matrix)
    else:
        SSmatrix = np.dot(snapshots_matrix, snapshots_matrix.T)
        print('WARNING - CHECK HOW THE BASIS FUNCTIONS ARE CALCULATED WITH THIS METHOD')

    print(
        'SSmatrix',
        SSmatrix.shape,
        'snapshots_matrix',
        snapshots_matrix.shape)
    eigvalues, v = np.linalg.eigh(SSmatrix)
    eigvalues = eigvalues[::-1]
    # get rid of small negative eigenvalues (there shouldn't be any as the eigenvalues of a real, symmetric
    # matrix are non-negative, but sometimes very small negative values do
    # appear)
    eigvalues[eigvalues < 0] = 0
    s_values = np.sqrt(eigvalues)
    nAll = len(eigvalues)

    if nPOD == -1:
        # truncation - the number of POD basis functions is chosen by seeing when the percentage
        # of cumulative information captured by a certain number of POD basis functions reaches a
        # user-defined tolerance
        # see https://onlinelibrary.wiley.com/doi/10.1002/nme.6681 for
        # a brief explanation of truncation (section 3.2)

        # calculate cumulative information captured by the POD basis functions
        cumulative_info = np.zeros(len(eigvalues))
        for j in range(len(eigvalues)):
            if j == 0:
                cumulative_info[j] = eigvalues[j]
            else:
                cumulative_info[j] = cumulative_info[j - 1] + eigvalues[j]

        cumulative_info = cumulative_info / cumulative_info[-1]

        nPOD = sum(cumulative_info <= cumulative_tol)  # tolerance
    elif nPOD == -2:
        # keeping all the POD basis functions
        nPOD = nAll
    # else:
    #    the user has specified nPOD

    print("retaining", nPOD, "basis functions of a possible", nAll)

    basis_functions = np.zeros((2 * nNodes, nPOD))
    for j in reversed(range(nAll - nPOD, nAll)):
        Av = np.dot(snapshots_matrix, v[:, j])
        basis_functions[:, nAll - j - 1] = Av / np.linalg.norm(Av)

    write_sing_values(s_values)

    return s_values, basis_functions


def write_sing_values(s_values):
    f = open('singular_values.dat', "w+")
    f.write('# index, s_values, normalised s_values, cumulative energy \n')
    # f.write('# field: %s\n' % field[k])

    running_total = np.zeros((len(s_values)))
    running_total[0] = s_values[0] * s_values[0]

    for i in range(1, len(s_values)):
        running_total[i] = running_total[i - 1] + s_values[i] * s_values[i]
    total = running_total[-1]

    for i in range(len(s_values)):
        f.write(
            '%d %g %g %18.10g \n' %
            (i,
             s_values[i],
             s_values[i] /
             s_values[0],
             running_total[i] /
             total))
    f.close()

    return
