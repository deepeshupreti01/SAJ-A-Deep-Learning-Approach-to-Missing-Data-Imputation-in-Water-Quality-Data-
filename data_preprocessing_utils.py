import numpy as np
import h5py
import os
from config import preprocessing_info

def splitscale(path,splitdata,eps:float=1e-9):
    data = np.genfromtxt(path, delimiter=',',missing_values='NaN',filling_values=np.nan)
    train, val, test = data[:splitdata[0]], data[splitdata[0]:splitdata[1]], data[splitdata[1]:]
    train_mean, train_std = np.nanmean(train,axis=0),np.nanstd(train,axis=0)
    train, val, test = [(x-train_mean)/(train_std+eps) for x in (train,val,test)]
    return train, val, test


def window_truncate(feature_vectors,seq_len,sliding_len=None):
    sliding_len = seq_len if sliding_len is None else sliding_len
    total_len = feature_vectors.shape[0]
    start_indices = np.asarray(range(total_len//sliding_len))*sliding_len
    if total_len-start_indices[-1]*sliding_len<seq_len:
        start_indices = start_indices[:-1]
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx:idx+seq_len])
    return np.asarray(sample_collector).astype('float32')


def add_artificial_mask(X,artificial_missing_rate,mode:str):
    assert mode in ['train','val','test'], 'The provided mode is not available'
    assert (0<artificial_missing_rate<1),'Artificial missing rate provided is not within range (0,1)'
    sample_num, seq_len, feature_num = X.shape
    if mode == "train":
        #You only add missingness if MIT is present, otherwise you don't need to add any value here
        data_dict = {
            'X':X
        }
    else:
        X = X.reshape(-1)
        indices = np.where(~np.isnan(X))[0].tolist()
        indices = np.random.choice(
            indices,
            int(len(indices) * artificial_missing_rate),
            replace=False
        ).astype(int)
        X_hat = np.copy(X)
        X_hat[indices] = np.nan
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        indicating_mask = ((~np.isnan(X_hat))^(~np.isnan(X))).astype(np.float32)
        data_dict = {
            'X':X.reshape([sample_num,seq_len,feature_num]),
            'X_hat':X_hat.reshape([sample_num,seq_len,feature_num]),
            'missing_mask':missing_mask.reshape([sample_num,seq_len,feature_num]),
            'indicating_mask':indicating_mask.reshape([sample_num,seq_len,feature_num])
        }
    return data_dict

def saving_to_h5(config,data_dict):
    def save_each_set(handle,mode,data):
        single_set = handle.create_group(mode)
        single_set.create_dataset('X',data=data['X'].astype(np.float32))
        if mode in ['val','test']:
            single_set.create_dataset('X_hat',data=data['X_hat'].astype(np.float32))
            single_set.create_dataset('missing_mask',data=data['missing_mask'].astype(np.float32))
            single_set.create_dataset('indicating_mask',data=data['indicating_mask'].astype(np.float32))
    saving_dir = config['saving_dir']
    dataset_name = f"{int(config['artificial_missing_rate']*100)}_datasets.h5"
    saving_path = os.path.join(saving_dir,dataset_name)
    with h5py.File(saving_path,'w') as hf:
        save_each_set(hf,'train',data_dict['train'])
        save_each_set(hf,'val',data_dict['val'])
        save_each_set(hf,'test',data_dict['test'])

def dataset_generating(config):
    train,val,test = splitscale(config['path'],config['splitdata'])
    train,val,test = [window_truncate(x,config['seq_len']) for x in (train,val,test)]
    train_dict,val_dict,test_dict = [add_artificial_mask(x,config['artificial_missing_rate'],y) for (x,y) in ((train,"train"),(val,"val"),(test,"test"))]
    processed_data = {
        'train':train_dict,
        'val':val_dict,
        'test':test_dict
    }
    saving_to_h5(config,processed_data)
    print('Saved Successfully!!!')

if __name__=="__main__":
    config = preprocessing_info()
    dataset_generating(config)