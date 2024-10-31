import h5py
import numpy as np
import optuna
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from config import *
from utils import masked_mae_cal, masked_mre_cal, masked_rmse_cal
import sys
import random

MISSING_PERCENTAGE = [30,40,50,60,70,80,90]
DATASET='USGSOhioRIver'           #choose from USGSMuddyFK, USGSOhioRiver
MODEL = 'MICE_kNN'              #choose from mean, locf, kNNImputer, MICE_XXX
NUM_ITERATIONS = 10             #For MICE_XXX
CONVERGENCE_THRESHOLD = 1e-4    #For MICE_XXX


class MICEImputation:
    def __init__(self, X, missing_mask):
        self.X = X
        self.missing_mask = missing_mask

    def init_imputation(self):
        imputer = SimpleImputer(strategy='mean')
        initial_imputations = imputer.fit_transform(self.X)
        return initial_imputations

    def regression_imputation(self,imputations,target_col_idx,missing_mask,model):
        #X_train, y_train and X_test are nomenclatures
        mice_mask = (1-missing_mask).astype(bool)
        target_missing_mask = mice_mask[:,target_col_idx]
        X_train = imputations[~target_missing_mask,:]
        y_train = imputations[~target_missing_mask,target_col_idx]
        X_train = np.delete(X_train,target_col_idx,axis=1)
        X_val = imputations[target_missing_mask,:]
        X_val = np.delete(X_val,target_col_idx,axis=1)
        if X_train.shape[0]>0:
            y_train = y_train.ravel()
            model.fit(X_train,y_train)
            predictions = model.predict(X_val)
            imputations[target_missing_mask,target_col_idx] = predictions
        return imputations

def read_files(folder,mode,mdp:int=20):
    file_path = f"{folder}/{mdp}_datasets.h5"
    with h5py.File(file_path) as hf:
        X = hf[mode]['X'][:]
        if mode == 'test' or mode=="val":
            X_hat = hf[mode]['X_hat'][:]
            indicating_mask = hf[mode]['indicating_mask'][:]
            return X,X_hat,indicating_mask
        return X


def mean_replacement(X_hat,X,indicating_mask):
    arr = np.copy(X_hat)
    col_mean = np.nanmean(arr,axis=0)
    idx = np.where(np.isnan(arr))
    arr[idx] = np.take(col_mean,idx[1])
    X_pred = arr
    error = error_calculation_naive(X_pred,np.nan_to_num(X),indicating_mask)
    return error

def locf(X_hat,X,indicating_mask):
    axis = 1
    arr = np.copy(X_hat)
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim == i else np.newaxis
                               for dim in range(len(arr.shape))])]
           for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    arr = arr[tuple(slc)]
    X_pred = np.nan_to_num(arr)
    error = error_calculation_naive(X_pred,np.nan_to_num(X),indicating_mask)
    return error

def knnimputer(X_hat,X,indicating_mask):
    arr = np.copy(X_hat)
    #From sqlite
    if DATASET == 'USGSMuddyFK':
        n_neighbors = 84
        weights = 'uniform'
    elif DATASET == 'USGSOhioRiver':
        n_neighbors = 200
        weights = 'uniform'
    else:
        raise ValueError
    imputer = KNNImputer(n_neighbors=n_neighbors,weights=weights)
    X_pred = imputer.fit_transform(arr)
    error = error_calculation_naive(X_pred,np.nan_to_num(X),indicating_mask)
    return error

def mice(X_hat,X,indicating_mask):
    #Print statements to debug
    # print(np.count_nonzero(np.isnan(X)))
    original_missing = np.isnan(X).astype(int)
    mask = original_missing + indicating_mask
    #Print statements to debug
    # print(np.sum(mask>1))
    missing_mask = 1-mask
    weights = "distance"
    n_neighbors = 4
    metric = "manhattan"
    if DATASET == "USGSMuddyFK":
        algorithm = "kd_tree"
    else:
        algorithm = "brute"
    #define model with sqlite file
    model = KNeighborsRegressor(weights=weights,
                                n_neighbors=n_neighbors,
                                algorithm=algorithm,
                                metric=metric, n_jobs=-3)
    miceimputation = MICEImputation(X_hat,missing_mask)
    imputations = miceimputation.init_imputation()
    for iteration in range(NUM_ITERATIONS):
        prev_imp = imputations.copy()
        for j in range(X.shape[1]):
            imputations = miceimputation.regression_imputation(imputations,j,missing_mask,model)
            change_in_imputations = np.max(np.abs(imputations-prev_imp),axis=0)
            if np.all(change_in_imputations<CONVERGENCE_THRESHOLD):
                break
    error = error_calculation_naive(imputations,np.nan_to_num(X),indicating_mask)
    return error



def error_calculation_naive(X_pred,X,indicating_mask):
    return [masked_rmse_cal(torch.from_numpy(X_pred),torch.from_numpy(X),torch.from_numpy(indicating_mask)),
            masked_mae_cal(torch.from_numpy(X_pred),torch.from_numpy(X),torch.from_numpy(indicating_mask)),
            masked_mre_cal(torch.from_numpy(X_pred),torch.from_numpy(X),torch.from_numpy(indicating_mask))]



if __name__=="__main__":
    file_stdout = sys.stdout
    test_printouts = f'output/{DATASET}/Naive_Result.txt'
    for mdp in MISSING_PERCENTAGE:
        RMSE_collector, MAE_collector, MRE_collector = [], [], []
        for seed in get_random_seeds():
            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            X, X_hat, indicating_mask = read_files(DATASET,'test',mdp)
            X, X_hat, indicating_mask = [x.reshape(x.shape[0] * x.shape[1], x.shape[2]) for x in
                                         (X, X_hat, indicating_mask)]
            # [rmse,mae,mre] = mean_replacement(X_hat,X,indicating_mask)
            # [rmse,mae,mre] = locf(X_hat,X,indicating_mask)
            # [rmse,mae,mre] = knnimputer(X_hat,X,indicating_mask)
            [rmse,mae,mre] = mice(X_hat,X,indicating_mask)
            RMSE_collector.append(rmse)
            MAE_collector.append(mae)
            MRE_collector.append(mre)
        mean_mae, mean_rmse, mean_mre = (np.mean(MAE_collector), np.mean(RMSE_collector), np.mean(MRE_collector))
        std_mae, std_rmse, std_mre = (np.std(MAE_collector), np.std(RMSE_collector), np.std(MRE_collector))
        with open(test_printouts,'a') as f:
            sys.stdout = f
            print(
                f"\n{DATASET}:\n",
                f"{MODEL}:\n",
                f"Missing percentage: {mdp}%\n"
                f"MAE={mean_mae:.4f} ± {std_mae:.4f}, "
                f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f}, "
                f"MRE={mean_mre:.4f} ± {std_mre:.4f}, "
            )
        sys.stdout = file_stdout
