import pandas as pd
import numpy as np
import sklearn.metrics as skm
import os
import re

__author__ = "Gyubin Son"
__copyright__ = "Copyright 2018, Daewoo Shipbuilding & Marine Engineering Co., Ltd."
__credits__ = ["Minsik Park", "Jaeyun Jeong", "Heejeong Choi"]
__version__ = "1.0"
__maintainer__ = "Gyubin Son"
__email__ = "gyubin_son@korea.ac.kr"
__status__ = "Develop"

PRED_RESULTS_DIR = './pred_results/'


def get_sorted_file_names():
    """
    directory 내의 파일들을 model의 번호순으로 정렬
    ex) model1_pred.csv, model34_pred.csv, model99_pred.csv, ...
    
    Args:
        None
    
    Returns:
        번호 순으로 정렬된 csv 파일들의 이름이 담긴 list
    
    Raises:
        FileNotFoundError: dir_name 디렉토리가 존재하지 않을 때
    """
    if not os.path.exists(PRED_RESULTS_DIR):
        raise FileNotFoundError('Directory "{}" does not exist.'.format(PRED_RESULTS_DIR))
        
    file_names = os.listdir(PRED_RESULTS_DIR)
    file_names = sorted(file_names,
                        key=lambda x: int(re.findall(r'[0-9]+', x)[0]))
    return file_names


def get_acc(file_path):
    """
    매개변수로 받은 예측 데이터의 accuracy return
    
    Args:
        file_path: 예측 데이터의 경로
    
    Returns:
        accuracy 값(0~1) return
    
    Raises:
        FileNotFoundError: 예측 데이터가 존재하지 않을 때
    """
    file_path = PRED_RESULTS_DIR + file_path
    if not os.path.isfile(file_path):
        raise FileNotFoundError('File "{}" does not exist.'.format(file_path))
        
    data = pd.read_csv(file_path)
    y_true = data['y_true'].values
    y_pred = data['y_pred'].values
    acc = skm.accuracy_score(y_true, y_pred)
    return acc


def select_top_k_of_part(file_names, top_k=30, upper_bound_epoch=100):
    """
    Upper bound epoch 까지의 예측 데이터 중 accuracy를 기준으로 상위 k개 return
    
    Args:
        file_names: 예측 데이터의 파일 이름들이 저장되어있는 리스트
        top_k: accuracy 기준 상위 몇 개를 뽑을 것인지 지정
        upper_bound_epoch: epoch 몇 회까지 학습한 것으로 계산할 것인지 지정
    
    Returns:
        상위 k개의 예측 데이터 파일 이름들이 담긴 list return
    
    Raises:
        None
    """
    result = sorted(file_names[:upper_bound_epoch],
                    key=get_acc,
                    reverse=True)
    return result[:top_k]


def majority_vote(file_names):
    """
    선택된 예측 데이터를 합쳐서 majority voting을 최종 ensemble 예측치 return
    
    Args:
        file_names: 예측 데이터의 파일 이름들이 저장되어있는 리스트
        
    Returns:
        최종 ensemble 예측 결과가 저장된 numpy arrary
    
    Raises:
        None
    """
    result = np.array([])
    for fn in file_names:
        fn = PRED_RESULTS_DIR + fn
        y_pred = pd.read_csv(fn)['y_pred'].values
        y_pred = y_pred.reshape(-1, 1)
        if len(result) == 0:
            result = y_pred
        else:
            result = np.hstack((result, y_pred))
    
    indices = np.argmax(result, axis=1)
    pred_ensemble = result[np.arange(len(indices)), indices]
    return pred_ensemble


def get_metrics(y_pred):
    """
    필요한 Metrics(f1, recall, precision, bcr, acc)를 출력 및 return
    
    Args:
        y_pred: y hat
    
    Returns:
        f1, recall, precision, bcr, acc 값이 담긴 dictionary
    
    Raises:
        None
    """
    y_true = pd.read_csv(PRED_RESULTS_DIR + 'model1_pred.csv')['y_true'].values
    cm = skm.confusion_matrix(y_true, y_pred)
    TN, FN, TP, FP = cm[0, 0], cm[1, 0], cm[1, 1], cm[0, 1]
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / (1/recall + 1/precision)
    acc = (TN+TP) / (TN+FN+TP+FP)
    tpr = recall
    tnr = TN / (TN + FP)
    bcr = np.sqrt(tpr * tnr)
    print('f1:\t{:.4f}\nrecall:\t{:.4f}\nprec:\t{:.4f}\nbcr:\t{:.4f}\nacc:\t{:.4f}'
          .format(f1, recall, precision, bcr, acc))
    
    result = {'f1': f1, 'recall': recall,
              'precision': precision, 'bcr': bcr, 'acc': acc}
    return result


def ensemble_a_of_b(top_k=30, upper_bound_epoch=100):
    """
    위 함수들을 모아서 쉽게 ensemble 할 수 있는 모음 함수

    Args:
        top_k: accuracy 기준 상위 몇 개를 뽑을 것인지 지정
        upper_bound_epoch: epoch 몇 회까지 학습한 것으로 계산할 것인지 지정

    Returns:
        결과 metric

    Raises:
        None
    """
    sorted_file_names = get_sorted_file_names()
    selected_file_names = select_top_k_of_part(sorted_file_names, top_k=top_k, upper_bound_epoch=upper_bound_epoch)
    pred_ensemble = majority_vote(selected_file_names)
    metrics_result = get_metrics(pred_ensemble)

