#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Credit to https://github.com/BorisMuzellec/MissingDataOT
"""


import os
import pandas as pd

import wget

DATASETS = ["qsar_biodegradation", "wine_quality_white",
            "concrete_compression", "parkinsons" "yeast",
            "ionosphere", "blood_transfusion", "breast_cancer_diagnostic",
            "connectionist_bench_vowel", "climate_model_crashes"]


def dataset_loader(dataset):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes
    datasets are located in './datasets'. If the called for dataset is not in
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.

    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS, f"Dataset not supported: {dataset}"

    if dataset in DATASETS:
        if dataset == 'qsar_biodegradation':
            X = fetch_qsar_biodegradation()['data']
        elif dataset == 'parkinsons':
            X = fetch_parkinsons()['data']
        elif dataset == 'concrete_compression':
            X = fetch_concrete_compression()['data']
        elif dataset == 'ionosphere':
            X = fetch_ionosphere()['data']
        elif dataset == 'blood_transfusion':
            X = fetch_blood_transfusion()['data']
        elif dataset == 'breast_cancer_diagnostic':
            X = fetch_breast_cancer_diagnostic()['data']
        elif dataset == 'connectionist_bench_vowel':
            X = fetch_connectionist_bench_vowel()['data']
        elif dataset == 'wine_quality_white':
            X = fetch_wine_quality_white()['data']

        return X


def fetch_parkinsons():
    if not os.path.isdir('datasets/parkinsons'):
        os.mkdir('datasets/parkinsons')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        wget.download(url, out='datasets/parkinsons/')

    with open('datasets/parkinsons/parkinsons.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=0)
        Xy = {}
        Xy['data'] = df.values[:, 1:].astype('float')
        Xy['target'] = df.values[:, 0]

    return Xy

def fetch_concrete_compression():
    if not os.path.isdir('datasets/concrete_compression'):
        os.mkdir('datasets/concrete_compression')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out='datasets/concrete_compression/')

    with open('datasets/concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-2]
        Xy['target'] = df.values[:, -1]

    return Xy

def fetch_ionosphere():
    if not os.path.isdir('datasets/ionosphere'):
        os.mkdir('datasets/ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out='datasets/ionosphere/')

    with open('datasets/ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy

def fetch_qsar_biodegradation():
    if not os.path.isdir('datasets/qsar_biodegradation'):
        os.mkdir('datasets/qsar_biodegradation')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out='datasets/qsar_biodegradation/')

    with open('datasets/qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy

def fetch_blood_transfusion():
    if not os.path.isdir('datasets/blood_transfusion'):
        os.mkdir('datasets/blood_transfusion')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out='datasets/blood_transfusion/')

    with open('datasets/blood_transfusion/transfusion.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy

def fetch_breast_cancer_diagnostic():
    if not os.path.isdir('datasets/breast_cancer_diagnostic'):
        os.mkdir('datasets/breast_cancer_diagnostic')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out='datasets/breast_cancer_diagnostic/')

    with open('datasets/breast_cancer_diagnostic/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] = df.values[:, 1]

    return Xy

def fetch_connectionist_bench_vowel():
    if not os.path.isdir('datasets/connectionist_bench_vowel'):
        os.mkdir('datasets/connectionist_bench_vowel')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out='datasets/connectionist_bench_vowel/')

    with open('datasets/connectionist_bench_vowel/vowel-context.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy

def fetch_wine_quality_white():
    if not os.path.isdir('datasets/wine_quality_white'):
        os.mkdir('datasets/wine_quality_white')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out='datasets/wine_quality_white/')

    with open('datasets/wine_quality_white/winequality-white.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy
