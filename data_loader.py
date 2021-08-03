import numpy as np
import os
import shutil
import urllib.request
import zipfile
import glob
from scipy.io import loadmat

def __downloadDataset():
    """Download the dataset into current working directory."""
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    path = 'dataset/activity'
    zippath = 'dataset/activity.zip'
    if not os.path.exists(path):
        os.mkdir(path)
        urllib.request.urlretrieve('https://www.dlr.de/kn/Portaldata/27/Resources/dokumente/041_Abteilung_COS/Activity_DataSet.zip')

        os.unlink(zippath)        

        for src in glob.glob('dataset/activity/ARS DLR Data Set/*'):
            shutil.move(src, 'dataset/activity')
        shutil.rmtree('dataset/activity/ARS DLR Data Set') 







def getDataset():
    __downloadDataset()
    datasetV1 = loadmat('dataset/activity/ARS_DLR_DataSet.mat')
    datasetV2 = loadmat('dataset/activity/ARS_DLR_DataSet_V2.mat')
    dataset = datasetV1.copy()
    dataset.update(datasetV2)

    del dataset['__header__']
    del dataset['__version__']
    del dataset['__globals__']
    del datasetV1
    del datasetV2
    return dataset








def dataset_preparation(dataset):
    dataset_keys = list(dataset.keys())
    dataset_values = list(dataset.values())
    tot_measurements = len(dataset_keys)
    del dataset_keys
    X_data = []
    labels = np.array([])
    for measure in range(tot_measurements):
        IMU_measurements = dataset_values[measure][0][0]
        direction_cosine = dataset_values[measure][0][1]

        #Removing the time element since we are not interested in that
        IMU_measurements = IMU_measurements[:, 1:]
        direction_cosine = direction_cosine[:, 1:]

        activities = dataset_values[measure][0][2][0]
        measurements_indexes = dataset_values[measure][0][3][0] - 1 #Matlab starts indexes from 1. we want the indexes that starts from 0
        for idx, act in enumerate(activities):
            if act == 'TRANSUP' or act == 'TRANSDW' or act == 'TRNSACC' or act == 'TRNSDCC' or act == 'TRANSIT':
                continue
            elif act == 'JUMPBCK' or act == 'JUMPFWD' or act == 'JUMPVRT':
                act = 'JUMPING'
            elif act == 'WALKDWS' or act == 'WALKUPS':
                act = 'WALKING'
            labels = np.append(labels, ''.join(act))
            index1 = measurements_indexes[2*idx]
            index2 = measurements_indexes[2*idx+1]
            for i in range(index1, index2):
                #ax = IMU_measurements[i][1]
                #ay = IMU_measurements[i][2]
                #az = IMU_measurements[i][3]
                ax = IMU_measurements[i][0]
                ay = IMU_measurements[i][1]
                az = IMU_measurements[i][2]
                acc = np.array([ax, ay, az])
                #Direction cosine matrix
                #Cb = np.array([[direction_cosine[i][1], direction_cosine[i][4], direction_cosine[i][7]],
                            #[direction_cosine[i][2], direction_cosine[i][5], direction_cosine[i][8]],
                            #[direction_cosine[i][3], direction_cosine[i][6], direction_cosine[i][9]]])
                
                Cb = np.array([[direction_cosine[i][0], direction_cosine[i][3], direction_cosine[i][6]],
                            [direction_cosine[i][1], direction_cosine[i][4], direction_cosine[i][7]],
                            [direction_cosine[i][2], direction_cosine[i][5], direction_cosine[i][8]]])
                #Transforming the acceleration in global frame
                an = np.matmul(Cb, acc)
                #IMU_measurements[i][1] = an[0]
                #IMU_measurements[i][2] = an[1]
                #IMU_measurements[i][3] = an[2]
                IMU_measurements[i][0] = an[0]
                IMU_measurements[i][1] = an[1]
                IMU_measurements[i][2] = an[2]
            measure_set = IMU_measurements[index1:index2]
            X_data.append(measure_set)
    del dataset_values
    return (np.array(X_data), labels)