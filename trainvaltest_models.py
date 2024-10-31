import sys
import time
import numpy as np
import random
import torch
import csv
from saits import build_SAITS
from saj import build_SAJ
from brits import build_BRITS
from dataloader import UnifiedDataloader
from config import *
from utils import masked_mae_cal,masked_mre_cal,masked_rmse_cal

def get_ds(config,mode):
    unified_dataloader = UnifiedDataloader(config['dataset_path'],config['seq_len'],config['feature_num'],config['model_type'],config['batch_size'],config['num_workers'],True)
    return unified_dataloader.get_trainvaltest_dataloader(mode)

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


def prepare_result_for_brits(ret,X_holdout,indicting_mask):
    imputation_loss = masked_mae_cal(ret['imputed_data'],X_holdout,indicting_mask) if X_holdout is not None else torch.tensor(0.)
    return {
        'consistency_loss':ret['consistency_loss'],
        'reconstruction_loss':ret['reconstruction_loss'],
        'imputation_loss':imputation_loss
    }


def prepare_result_for_saj(tilde_X,Xc,X,X_holdout,missing_mask,indicating_mask):
    reconstruction_loss = 0.
    reconstruction_loss += masked_mae_cal(tilde_X,X,missing_mask)
    imputation_loss = masked_mae_cal(Xc,X_holdout,indicating_mask)

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

def get_model(config,hyperparameters):
    if config['model_type']=="SAITS":
        d_k = hyperparameters['d_model'] // hyperparameters['h']
        model = build_SAITS(hyperparameters['N'],hyperparameters['n_inner_groups'],config['seq_len'],config['feature_num'],hyperparameters['d_model'],hyperparameters['d_ff'],hyperparameters['h'],d_k,hyperparameters['d_v'],hyperparameters['dropout'],hyperparameters['attn_dropout']).to(config['device'])
        return model

    elif config['model_type'] == 'SAJ':
        model = build_SAJ(config['seq_len'],config['feature_num'],hyperparameters['d_model'],hyperparameters['d_v'],hyperparameters['d_inner'],hyperparameters['d_conv'],hyperparameters['h'],hyperparameters['N'],hyperparameters['dropout'],hyperparameters['conv_dropout'],hyperparameters['attn_dropout']).to(config['device'])
        return model
    elif config['model_type']=="BRITS":
        model = build_BRITS(config['seq_len'],config['feature_num'],hyperparameters['hidden_size'],hyperparameters['dropout'],True,config['device']).to(config['device'])
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


def saj_run(model,data,config,N):
    indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x:x.to(config['device']),data)
    first_rep = model.firstrep(X, missing_mask,config['device'])
    tilde_X1 = model.tilde_X1(X, missing_mask,config['device'])
    attn = model.sajloop.layers[N-1].self_attention_block.attention_scores
    eta = model.weighted_combination_block(attn)
    Xc = model.compile(eta, first_rep)
    return tilde_X1,Xc,X,X_holdout,missing_mask,indicating_mask





def train_val(config,hyperparameters,seed):
    train_dataloader,val_dataloader = get_ds(config,'train'),get_ds(config,'val')
    if config['model_type'] in ["SAITS","SAJ"]:
        model = get_model(config,hyperparameters)
        N = hyperparameters['N']
    elif config['model_type'] in ["BRITS"]:
        model = get_model(config,hyperparameters)
    else:
        return ValueError,"No model by that name"
    optimizer = hyperparameters['optimizer']
    lr = hyperparameters['lr']
    if optimizer=='adam':
        optimize = torch.optim.Adam(model.parameters(),lr=lr)
    elif optimizer=='adamw':
        optimize = torch.optim.AdamW(model.parameters(),lr=lr)
    else:
        raise ValueError
    best_val_loss = np.inf
    early_stopping = config['earlystopping']
    current_stopping = early_stopping
    run_result = {'epochs': [], 'train_loss': [], 'val_loss': []}
    for epoch in range(config['epochs']):
        print(f"Epoch: {epoch+1}")
        running_loss = 0.
        model.train()
        run_result['epochs'].append(epoch+1)
        for idx,data in enumerate(train_dataloader):
            if config["model_type"] in ["SAITS"]:
                tilde_X1, tilde_X2, tilde_X3, Xc, X, X_holdout, missing_mask, indicating_mask = saits_run(model, data,config)
                result = prepare_result_for_saits(tilde_X1, tilde_X2, tilde_X3, Xc, X, X_holdout, missing_mask, indicating_mask)
            elif config["model_type"] in ["SAJ"]:
                tilde_X1, Xc, X, X_holdout, missing_mask, indicating_mask = saj_run(model,data,config,N)
                result = prepare_result_for_saj(tilde_X1,Xc,X,X_holdout,missing_mask,indicating_mask)
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
        run_result['train_loss'].append(running_loss/len(train_dataloader))
        model.eval()
        evalX_collector, evalMask_collector,imputation_collector = [],[],[]
        val_loss_collector = []
        with torch.no_grad():
            for idx,data in enumerate(val_dataloader):
                if config["model_type"] in ["SAITS"]:
                    tilde_X1, tilde_X2, tilde_X3,Xc, X, X_holdout, missing_mask, indicating_mask = saits_run(model,data,config)
                elif config["model_type"] in ["SAJ"]:
                    tilde_X1, Xc, X, X_holdout, missing_mask, indicating_mask = saj_run(model, data,config,N)
                elif config['model_type'] == "BRITS":
                    ret,X_holdout,indicating_mask = brits_run_val(model,data,config)
                    Xc = ret['imputed_data']
                else:
                    return ValueError, 'No model by that name'
                if not val_loss_collector:
                    evalX_collector.append(X_holdout)
                    evalMask_collector.append(indicating_mask)
                    imputation_collector.append(Xc)
            if not val_loss_collector:
                evalX_collector = torch.concat(evalX_collector)
                evalMask_collector = torch.concat(evalMask_collector)
                imputation_collector = torch.concat(imputation_collector)
                imputation_MAE = masked_mae_cal(imputation_collector,evalX_collector,evalMask_collector)
                val_loss = imputation_MAE.cpu().numpy().mean()
            else:
                val_loss = np.asarray(val_loss_collector).mean()
        run_result['val_loss'].append(val_loss)
        print(val_loss)
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            current_stopping = early_stopping
            print('Saving the model.....')
            torch.save(model.state_dict(),f"{config['best_model_dir']}/{config['model_type']}_{seed}.pt")
        else:
            current_stopping-=1
        if current_stopping==0:
            print('Early Stopping')
            break
    if config['save_train_vs_val_loss']:
        with open(f"{config['best_model_dir'].split('/')[1]}_train_loss_vs_val_loss.csv",'w',newline='') as f:
            writer = csv.writer(f)
            header = list(run_result.keys())
            writer.writerow(header)
            for epoch, train_loss, val_loss in zip(run_result['epochs'],run_result['train_loss'],run_result['val_loss']):
                writer.writerow([epoch,train_loss,val_loss])
            print('Saved')


def test_model(config,hyperparameters,seed):
    if config['model_type'] in ["SAITS","SAJ"]:
        model = get_model(config,hyperparameters)
        N = hyperparameters['N']
    elif config['model_type'] in ["BRITS"]:
        model = get_model(config,hyperparameters)
    else:
        return ValueError,"No model by that name"
    model.load_state_dict(torch.load(f"{config['best_model_dir']}/{config['model_type']}_{seed}.pt"))
    test_dataloader = get_ds(config,'test')
    start = time.time()
    model.eval()
    evalX_collector, evalMask_collector, imputation_collector =[],[],[]
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            if config["model_type"] in ["SAITS"]:
                tilde_X1, tilde_X2, tilde_X3, Xc, X, X_holdout, missing_mask, indicating_mask = saits_run(model, data, config)
            elif config["model_type"] in ["SAJ"]:
                tilde_X1, Xc, X, X_holdout, missing_mask, indicating_mask = saj_run(model, data, config, N)
            elif config['model_type'] == "BRITS":
                ret, X_holdout, indicating_mask = brits_run_val(model, data, config)
                Xc = ret['imputed_data']
            else:
                return ValueError, 'No model by that name'
            evalX_collector.append(X_holdout)
            evalMask_collector.append(indicating_mask)
            imputation_collector.append(Xc)
        evalX_collector = torch.concat(evalX_collector)
        evalMask_collector = torch.concat(evalMask_collector)
        imputation_collector = torch.concat(imputation_collector)
        return evalX_collector, evalMask_collector, imputation_collector,time.time()-start,model


if __name__=="__main__":
    model_config = get_info()
    file_stdout = sys.stdout
    result_storage = get_result()
    test_printouts = result_storage['test_result']
    mae_collector = []
    rmse_collector = []
    mre_collector = []
    time_collector = []
    hyperparameters = get_hpo_results()[model_config['model_type']]
    for seed in get_random_seeds():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # train_val(model_config, hyperparameters,seed)

       # After saving the best model, use the model to test on different missing percentage by uncommenting the below lines of code
    #
        evalX_collector, evalMask_collector,imputation_collector,time_diff,model = test_model(model_config,hyperparameters,seed)
        time_collector.append(time_diff)
        mae = masked_mae_cal(imputation_collector,evalX_collector,evalMask_collector)
        rmse = masked_rmse_cal(imputation_collector,evalX_collector,evalMask_collector)
        mre = masked_mre_cal(imputation_collector,evalX_collector,evalMask_collector)
        mae_collector.append(mae.item())
        rmse_collector.append(rmse.item())
        mre_collector.append(mre.item())
    mean_mae, mean_rmse, mean_mre = (np.mean(mae_collector),np.mean(rmse_collector),np.mean(mre_collector))
    std_mae, std_rmse, std_mre = (np.std(mae_collector),np.std(rmse_collector),np.std(mre_collector))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_config['store_result_in_notepad']:
        with open(test_printouts,'a') as f:
            sys.stdout = f
            print(
                f"\n{model_config['best_model_dir'].split('/')[1]}:\n"
                f"Missing percentage: {int(preprocessing_info()['artificial_missing_rate']*100)}%\n"
                f"Averaged {model_config['model_type']}: {num_params} parameters\n"
                f"MAE={mean_mae:.4f} ± {std_mae:.4f}, "
                f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f}, "
                f"MRE={mean_mre:.4f} ± {std_mre:.4f}, "
                f"Average Inference time={np.mean(time_collector):.2f} seconds"
            )
        sys.stdout = file_stdout
