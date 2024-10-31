import numpy as np
import torch
import optuna
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from utils import masked_mae_cal
from naive_models import read_files

DATASET = 'USGSMuddyFK'           #choose from USGSMuddyFK, USGSOhioRiver
MODEL = 'MICE_RF'               #choose from kNNImputer,  MICE_RF, MICE_XGBR, MICE_kNN, MICE_HGBR
NUM_ITERATIONS = 25
CONVERGENCE_THRESHOLD = 1e-4
MISSING_PERCENTAGE = 20

def introduce_missingness_train():
    #Load split and scaled train data
    X = read_files(DATASET,'train',MISSING_PERCENTAGE)
    l,m,n = X.shape[0],X.shape[1], X.shape[2]
    X = X.reshape(-1)
    indices = np.where(~np.isnan(X))[0].tolist()
    indices = np.random.choice(
        indices,
        int(len(indices) * MISSING_PERCENTAGE/100),
        replace=False
    ).astype(int)
    X_hat = np.copy(X)
    X_hat[indices] = np.nan
    missing_mask = (~np.isnan(X_hat)).astype(np.float32)    #Gives 1 if observed and 0 otherwise
    indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32) #Gives 1 if artificially missing 0 otherwise
    X, X_hat, missing_mask, indicating_mask = [x.reshape(l*m, n) for x in
                                               (X, X_hat, missing_mask, indicating_mask)]
    return X, X_hat, missing_mask, indicating_mask


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

def mice_run(trial):
    if MODEL == 'MICE_kNN':
        n_neighbors = trial.suggest_categorical('n_neighbors', [i for i in range(1, 201)])
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev'])
        model = KNeighborsRegressor(weights=weights,
                                    n_neighbors=n_neighbors,
                                    algorithm=algorithm,
                                    metric=metric)
    elif MODEL == 'MICE_XGBR':
        lr = trial.suggest_categorical('learning_rate', [0.05, 0.1, 0.3])
        n_estimators = trial.suggest_categorical('n_estimators', [100, 500, 1000])
        max_depth = trial.suggest_categorical('max_depth', [3, 5, 8])
        min_child_weight = trial.suggest_categorical('min_child_weight', [1, 3, 5])
        gamma = trial.suggest_categorical('gamma', [0, 0.1, 0.3])
        subsample = trial.suggest_categorical('subsample', [0.6, 0.8, 1.0])
        colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0])
        model = XGBRegressor(n_estimators=n_estimators,
                             subsample = subsample,
                             min_child_weight = min_child_weight,
                             learning_rate = lr,
                             max_depth = max_depth,
                             gamma = gamma,
                             colsample_bytree = colsample_bytree)
    elif MODEL == 'MICE_RF':
        n_estimators = trial.suggest_categorical('n_estimators',[50, 100, 150, 200, 250, 300, 500])
        max_features = trial.suggest_categorical('max_features',[0.2, 0.4, 0.6, 0.8, 1.0])
        max_depth = trial.suggest_categorical('max_depth',[4, 8, 12, 16, None])
        max_samples = trial.suggest_categorical('max_samples',[0.5, 0.65, 0.75, 0.85, 1.0])
        min_samples_split = trial.suggest_categorical('min_samples_split',[2, 5, 10])
        min_samples_leaf = trial.suggest_categorical('min_samples_leaf',[1, 2, 4])
        model = RandomForestRegressor(n_estimators = n_estimators,
                                      max_features=max_features,
                                      max_depth = max_depth,
                                      max_samples = max_samples,
                                      bootstrap = True,
                                      min_samples_split = min_samples_split,
                                      min_samples_leaf = min_samples_leaf)
    else:
        raise ValueError
    X, X_hat, missing_mask, indicating_mask = introduce_missingness_train()
    miceimputation = MICEImputation(X,missing_mask)
    imputations = miceimputation.init_imputation()
    for iteration in range(NUM_ITERATIONS):
        prev_imp = imputations.copy()
        for j in range(X.shape[1]):
            imputations = miceimputation.regression_imputation(imputations,j,missing_mask,model)
            change_in_imputations = np.max(np.abs(imputations-prev_imp),axis=0)
            if np.all(change_in_imputations<CONVERGENCE_THRESHOLD):
                break
    return masked_mae_cal(torch.from_numpy(imputations),torch.from_numpy(np.nan_to_num(X)),torch.from_numpy(indicating_mask))


#Check
def knnimputer(trial):
    X, X_hat, indicating_mask = read_files(DATASET,'val',20)
    X, X_hat, indicating_mask = [x.reshape(x.shape[0]*x.shape[1],x.shape[2]) for x in (X,X_hat,indicating_mask)]
    arr = np.copy(X_hat)
    n_neighbors = trial.suggest_categorical('n_neighbors',[i for i in range(1,201)])
    weights = trial.suggest_categorical('weights',['uniform','distance'])
    imputer = KNNImputer(n_neighbors=n_neighbors,weights=weights)
    X_pred = imputer.fit_transform(arr)
    return masked_mae_cal(torch.from_numpy(X_pred),torch.from_numpy(np.nan_to_num(X)),torch.from_numpy(indicating_mask))



if __name__=="__main__":
    if MODEL == 'kNNImputer':
        search_space = {
                'n_neighbors':[i for i in range(1,201)],
                'weights':['uniform','distance']
                }
        study = optuna.create_study(direction='minimize',sampler=optuna.samplers.GridSampler(search_space=search_space),storage=f"sqlite:///{MODEL}.sqlite3",study_name=f"{DATASET}")
        study.optimize(knnimputer,n_jobs=1)
    else:
        study = optuna.create_study(direction='minimize',sampler=optuna.samplers.RandomSampler(),storage=f"sqlite:///{MODEL}.sqlite3",study_name=f"{DATASET}")
        study.optimize(mice_run, n_jobs=1,n_trials=100)
    print("Study statistics: ")
    print(" Number of finished trials: ",len(study.trials))
    print("Best trial")
    trial = study.best_trial
    print(" Value: ",trial.value)
    print("Params: ")
    for key,value in trial.params.items():
        print("   {}: {}".format(key,value))
