import numpy as np
import torch
import random
from utils import masked_mae_cal
from saits import build_SAITS
from brits import build_BRITS
from dataloader import UnifiedDataloader
from config import *
import optuna
from optuna.trial import TrialState

RANDOM_SEED = 120
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.random.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)



def get_ds(config,mode):
    unified_dataloader = UnifiedDataloader(config['dataset_path'],config['seq_len'],config['feature_num'],config['model_type'],config['batch_size'],config['num_workers'],True)
    return unified_dataloader.get_trainvaltest_dataloader(mode)

def prepare_result_for_brits(ret,X_holdout,indicting_mask):
    imputation_loss = masked_mae_cal(ret['imputed_data'],X_holdout,indicting_mask)
    return {
        'consistency_loss':ret['consistency_loss'],
        'reconstruction_loss':ret['reconstruction_loss'],
        'imputation_loss':imputation_loss
    }



def prepare_result_for_saits(tilde_X1,tilde_X2,tilde_X3,Xc,X,X_holdout,missing_mask,indicating_mask):
    reconstruction_loss =0.
    reconstruction_loss+=masked_mae_cal(tilde_X1,X,missing_mask)
    reconstruction_loss+=masked_mae_cal(tilde_X2,X,missing_mask)
    reconstruction_loss+=masked_mae_cal(tilde_X3,X,missing_mask)
    reconstruction_loss/=3
    imputation_loss = masked_mae_cal(tilde_X3,X_holdout,indicating_mask)
    return {
        "reconstruction_loss":reconstruction_loss,
        "imputation_loss":imputation_loss
    }


def result_process(config,result):
    result['total_loss'] = torch.tensor(0.,device=config['device'])
    if config["model_type"]=="BRITS":
        result['total_loss'] = (result['consistency_loss']*config['consistency_loss_wt'])
    result['reconstruction_loss'] = (result['reconstruction_loss']*config['reconstruction_loss_wt'])
    result['imputation_loss'] = (result['imputation_loss']*config['imputation_loss_wt'])
    result['total_loss'] += result['imputation_loss']
    if config['ORT']:
        result['total_loss']+=result['reconstruction_loss']
    return result

def get_model(config, trial):
    if config['model_type']=="SAITS":
        d_model = trial.suggest_categorical('d_model',[64, 128, 256, 512])
        d_v = trial.suggest_categorical('d_v',[32,64,128,256,512])
        d_ff = trial.suggest_categorical('d_ff',[128, 256, 512,1024,2048])
        N = trial.suggest_categorical('N',[1,2,3,4,5,6,7,8])
        n_inner_groups = trial.suggest_categorical('n_inner_groups',[1])
        h = trial.suggest_categorical('h',[2,4,8])
        d_k = d_model // h
        dropout = trial.suggest_categorical('dropout',[0,0.1,0.2,0.3,0.4,0.5])
        attn_dropout = trial.suggest_categorical('attn_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        model = build_SAITS(N,n_inner_groups,config['seq_len'],config['feature_num'],d_model,d_ff,h,d_k,d_v,dropout,attn_dropout).to(config['device'])
        return model,N

    elif config['model_type']=="BRITS":
        hidden_size = trial.suggest_categorical('hidden_size',[32,64,128,256,512])
        dropout = trial.suggest_categorical('dropout',[0,0.1,0.2,0.3,0.4,0.5])
        model = build_BRITS(config['seq_len'],config['feature_num'],hidden_size,dropout,True,config['device']).to(config['device'])
        return model

    else:
        return ValueError, 'No model by that name'

def brits_run_train(model,data,config):
    indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, indicating_mask = map(lambda x:x.to(config['device']),data)
    input_data = {
        'forward':{'X':X,'missing_mask':missing_mask,'deltas':deltas},
        'backward':{'X':back_X,'missing_mask':back_missing_mask,'deltas':back_deltas}
    }
    ret_f = model._forward(input_data)
    ret_b = model._reverse(model._backward(input_data))
    ret = model.merge_ret(ret_f,ret_b)
    return ret,X_holdout,indicating_mask

def brits_run_val(model,data,config):
    indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, indicating_mask = map(lambda x: x.to(config['device']), data)
    input_data = {
        'forward':{'X':X,'missing_mask':missing_mask,'deltas':deltas},
        'backward':{'X':back_X,'missing_mask':back_missing_mask,'deltas':back_deltas}
    }
    ret_f = model._forward(input_data)
    ret_b = model._reverse(model._backward(input_data))
    ret = model.merge_ret(ret_f,ret_b)
    return ret, X_holdout,indicating_mask



def saits_run(model,data,config):
    indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x:x.to(config['device']),data)
    inputs = {
        "X":X,
        "missing_mask":missing_mask,
        "X_holdout":X_holdout,
        "indicating_mask":indicating_mask
    }
    Xc, [tilde_X1, tilde_X2, tilde_X3] = model.impute(inputs)
    return tilde_X1,tilde_X2,tilde_X3,Xc,X,X_holdout,missing_mask,indicating_mask




def train_val(trial):
    config = get_info()
    train_dataloader,val_dataloader = get_ds(config,'train'),get_ds(config,'val')
    if config['model_type'] in ["SAITS"]:
        model,N = get_model(config,trial)
    elif config['model_type'] in ["BRITS"]:
        model = get_model(config,trial)
    else:
        return ValueError,"No model by that name"
    optimizer = trial.suggest_categorical('optimizer', ["adam", "adamw"])
    lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
    if optimizer=='adam':
        optimize = torch.optim.Adam(model.parameters(),lr=lr)
    elif optimizer=='adamw':
        optimize = torch.optim.AdamW(model.parameters(),lr=lr)
    else:
        raise ValueError
    best_val_loss = np.inf
    early_stopping = config['earlystopping']
    current_stopping = early_stopping
    states_to_consider = (TrialState.COMPLETE,)
    trials_to_consider = trial.study.get_trials(deepcopy=False,states=states_to_consider)
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value
    for epoch in range(config['epochs']):
        print(f"Epoch: {epoch+1}")
        running_loss = 0.
        model.train()
        for idx,data in enumerate(train_dataloader):
            if config["model_type"] in ["SAITS"]:
                tilde_X1, tilde_X2, tilde_X3,Xc, X, X_holdout, missing_mask, indicating_mask = saits_run(model,data,config)
                result = prepare_result_for_saits(tilde_X1,tilde_X2,tilde_X3,Xc,X,X_holdout,missing_mask,indicating_mask)
            elif config["model_type"] == "BRITS":
                ret, X_holdout, indicating_mask = brits_run_train(model,data,config)
                result = prepare_result_for_brits(ret,X_holdout,indicating_mask)
            else:
                return ValueError ,'No model by that name'
            result = result_process(config,result)
            running_loss += result['total_loss'].item()
            optimize.zero_grad()
            result['total_loss'].backward()
            optimize.step()
        model.eval()
        evalX_collector, evalMask_collector,imputation_collector = [],[],[]
        with torch.no_grad():
            for idx,data in enumerate(val_dataloader):
                if config["model_type"] in ["SAITS"]:
                    tilde_X1, tilde_X2, tilde_X3,Xc, X, X_holdout, missing_mask, indicating_mask = saits_run(model,data,config)
                elif config['model_type'] == "BRITS":
                    ret,X_holdout,indicating_mask = brits_run_val(model,data,config)
                    Xc = ret['imputed_data']
                else:
                    return ValueError, 'No model by that name'
                evalX_collector.append(X_holdout)
                evalMask_collector.append(indicating_mask)
                imputation_collector.append(Xc)
            evalX_collector = torch.concat(evalX_collector)
            evalMask_collector = torch.concat(evalMask_collector)
            imputation_collector = torch.concat(imputation_collector)
            imputation_MAE = masked_mae_cal(imputation_collector,evalX_collector,evalMask_collector)
            val_loss = imputation_MAE.cpu().numpy().mean()
            print(val_loss)
            if np.isnan(val_loss):
                print('NaN value for validation loss.. Terminating..')
                break
            trial.report(val_loss,epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                current_stopping = early_stopping
            else:
                current_stopping-=1
            if current_stopping==0:
                print('Early Stopping')
                break

    return best_val_loss

if __name__=="__main__":
    config = get_info()
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.RandomSampler(seed=RANDOM_SEED),storage=f"sqlite:///{config['best_model_dir'].split('/')[1]}_{config['model_type']}.sqlite3",study_name=f"{config['model_type']}",load_if_exists=True)
    for trial in study.trials:
        if trial.state == TrialState.FAIL:
            study.enqueue_trial(trial.params)
    study.optimize(train_val,n_trials=76,n_jobs=8)
    print("Study statistics: ")
    print(" Number of finished trials: ",len(study.trials))
    failed_trials = sum(1 for trial in study.get_trials() if trial.state == TrialState.FAIL)
    complete_trials = sum(1 for trial in study.get_trials() if trial.state == TrialState.COMPLETE)
    pruned_trials = sum(1 for trial in study.get_trials() if trial.state == TrialState.PRUNED)
    print(" Number of failed trials: ",failed_trials)
    print(" Number of completed trials: ", complete_trials)
    print(" Number of pruned trials: ", pruned_trials)
    print("Best trial")
    trial = study.best_trial
    print(" Value: ",trial.value)
    print("Params: ")
    for key,value in trial.params.items():
        print("   {}: {}".format(key,value))