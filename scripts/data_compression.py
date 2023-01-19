"""
Copyright: Donghu Guo

Author: Donghu Guo

DDescription: this is the python script to compress the nodes data into POD coefficient

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
from sklearn.decomposition import PCA
import numpy as np
import joblib
import os

for c in range(1, 9):
    # *************values setting***********
    root_path = '..data/' # set the root path where data is stored
    dataset_type = 'train'  # train or test

    p_tot = 0.9999999999999999
    p = 0.95  #0.925
    # *************values setting***********


    # Load matrix
    all_values = joblib.load(
        root_path +
        'Cotrace_fixed_720_cases/case{}_npys/case{}_'.format(c, c) +
        dataset_type +
        '_snapshot_matrix.pkl')

    # p_tot -> proportion of the variance we want to keep
    # Can’t be set to 1, because 1 means that only 1 component will be left
    # Create an instance of the PCA model
    pca = PCA(p_tot)
    train_pca = pca.fit_transform(all_values)
    X_recovered = pca.inverse_transform(train_pca)
    np.allclose(X_recovered, all_values)

    # Find the cumulative sum of the explained variance ratio to plot
    cumsum_eig = np.cumsum(pca.explained_variance_ratio_)
    d_tot = pca.n_components_
    # Find the number of principle components
    # d = np.argmax(cumsum_eig >= p) + 1
    d = 120
    print("initial number of components = ", d_tot)
    print("number of components after PCA = ", d)

    pca_compress = PCA(n_components=d)
    X_pca = pca_compress.fit_transform(all_values)
    X_recovered = pca_compress.inverse_transform(X_pca)
    print(np.allclose(X_recovered, all_values))
    print(X_pca.shape)

    joblib.dump(
        X_pca,
        root_path +
        'Cotrace_fixed_720_cases/case{}_npys/case{}_'.format(c, c) +
        dataset_type +
        '_pod_coefficients_{}.pkl'.format(d))
    joblib.dump(
        pca_compress,
        root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_'.format(c, c) + dataset_type + '_pca_compress_{}.pkl'.format(d))
