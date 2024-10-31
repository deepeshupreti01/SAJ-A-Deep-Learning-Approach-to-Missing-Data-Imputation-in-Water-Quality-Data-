import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

def parse_delta(masks,seq_len,feature_num):
  deltas = []
  for h in range(seq_len):
    if h==0:
      deltas.append(np.zeros(feature_num))
    else:
      deltas.append(np.ones(feature_num)+(1-masks[h]*deltas[-1]))
  return np.asarray(deltas)



class LoadTrainDataset(Dataset):
    def __init__(self,path,seq_len,feature_num,model_type,masked_imputation_task):
        super().__init__()
        self.masked_imputation_task = masked_imputation_task
        if masked_imputation_task:
            self.artificial_missing_rate = 0.2
        with h5py.File(path,'r') as hf:
            self.X = hf['train']['X'][:]
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.masked_imputation_task:
            X = X.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(
                indices,
                int(len(indices) * self.artificial_missing_rate),
                replace=False
            ).astype(int)
            X_hat = np.copy(X)
            X_hat[indices] = np.nan
            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)
            X = np.nan_to_num(X)
            X_hat = np.nan_to_num(X_hat)
            X, X_hat, missing_mask, indicating_mask = [x.reshape(self.seq_len,self.feature_num) for x in (X,X_hat,missing_mask,indicating_mask)]
            if self.model_type in ["SAITS", "SAJ"]:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X_hat.astype("float32")),
                    torch.from_numpy(missing_mask.astype("float32")),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(indicating_mask.astype("float32")),
                )
            elif self.model_type in ["BRITS"]:
                forward = {
                    "X_hat": X_hat,
                    "missing_mask": missing_mask,
                    "deltas": parse_delta(missing_mask, self.seq_len, self.feature_num),
                }

                backward = {
                    "X_hat": np.flip(forward["X_hat"], axis=0).copy(),
                    "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
                }
                backward["deltas"] = parse_delta(
                    backward["missing_mask"], self.seq_len, self.feature_num
                )
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(forward["X_hat"].astype("float32")),
                    torch.from_numpy(forward["missing_mask"].astype("float32")),
                    torch.from_numpy(forward["deltas"].astype("float32")),
                    torch.from_numpy(backward["X_hat"].astype("float32")),
                    torch.from_numpy(backward["missing_mask"].astype("float32")),
                    torch.from_numpy(backward["deltas"].astype("float32")),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(indicating_mask.astype("float32")),
                )
            else:
                assert ValueError,f"No model of type {self.model_type}"
        else:
            missing_mask = (~np.isnan(X)).astype(np.float32)
            X = np.nan_to_num(X)
            if self.model_type in ['SAITS', "SAJ"]:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(missing_mask.astype('float32'))
                )
            elif self.model_type in ['BRITS']:
                forward = {
                    "X": X,
                    "missing_mask": missing_mask,
                    "deltas": parse_delta(missing_mask, self.seq_len, self.feature_num),
                }

                backward = {
                    "X": np.flip(forward["X"], axis=0).copy(),
                    "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
                }
                backward["deltas"] = parse_delta(
                    backward["missing_mask"], self.seq_len, self.feature_num
                )
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(forward["X"].astype("float32")),
                    torch.from_numpy(forward["missing_mask"].astype("float32")),
                    torch.from_numpy(forward["deltas"].astype("float32")),
                    torch.from_numpy(backward["X"].astype("float32")),
                    torch.from_numpy(backward["missing_mask"].astype("float32")),
                    torch.from_numpy(backward["deltas"].astype("float32")),
                )
            else:
                assert ValueError,f"No model of type {self.model_type}"
        return sample


class LoadValTestDataset(Dataset):
    def __init__(self,path,mode,seq_len,feature_num,model_type):
        assert mode in ['val','test'],f'No mode by the name of {mode}'
        with h5py.File(path,'r') as hf:
            self.X = hf[mode]['X'][:]
            self.X_hat = hf[mode]['X_hat'][:]
            self.missing_mask = hf[mode]['missing_mask'][:]
            self.indicating_mask = hf[mode]['indicating_mask'][:]
        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ['SAITS',"SAJ"]:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X_hat[idx].astype("float32")),
                torch.from_numpy(self.missing_mask[idx].astype("float32")),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.indicating_mask[idx].astype("float32")),
            )
        elif self.model_type in ['BRITS']:
            forward = {
                "X_hat": self.X_hat[idx],
                "missing_mask": self.missing_mask[idx],
                "deltas": parse_delta(self.missing_mask[idx], self.seq_len, self.feature_num),
            }

            backward = {
                "X_hat": np.flip(forward["X_hat"], axis=0).copy(),
                "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
            }
            backward["deltas"] = parse_delta(
                backward["missing_mask"], self.seq_len, self.feature_num
            )
            sample = (
                torch.tensor(idx),
                torch.from_numpy(forward["X_hat"].astype("float32")),
                torch.from_numpy(forward["missing_mask"].astype("float32")),
                torch.from_numpy(forward["deltas"].astype("float32")),
                torch.from_numpy(backward["X_hat"].astype("float32")),
                torch.from_numpy(backward["missing_mask"].astype("float32")),
                torch.from_numpy(backward["deltas"].astype("float32")),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.indicating_mask[idx].astype("float32")),
            )
        else:
            assert ValueError, f"No model of type {self.model_type}"

        return sample



class UnifiedDataloader:
    def __init__(self,path,seq_len,feature_num,model_type,batch_size,num_workers,masked_imputation_task:bool=False):
        self.path = path
        self.seq_len = seq_len
        self.feature_num = feature_num
        assert model_type in ['SAITS','BRITS',"SAJ"],f'No model by the name of {model_type}'
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.masked_imputation_task = masked_imputation_task

    def get_trainvaltest_dataloader(self,mode:str):
        assert mode in ['train','val','test'],'Model should have train, val or test mode set'
        if mode == 'train':
            dataset = LoadTrainDataset(self.path,self.seq_len,self.feature_num,self.model_type,self.masked_imputation_task)
            dataloader = DataLoader(dataset,self.batch_size,True,num_workers=self.num_workers)
        else:
            dataset = LoadValTestDataset(self.path,mode,self.seq_len,self.feature_num,self.model_type)
            dataloader = DataLoader(dataset,self.batch_size,False,num_workers=self.num_workers)
        return dataloader