import sys
import numpy as np
import random
import torch
from ablation_saj import build_SAJ
from dataloader import UnifiedDataloader
from ablation_config import *
from utils import masked_mae_cal,masked_mre_cal,masked_rmse_cal



def get_ds(config,mode):
    unified_dataloader = UnifiedDataloader(config['dataset_path'],config['seq_len'],config['feature_num'],'SAJ',config['batch_size'],config['num_workers'],config['MIT'])
    return unified_dataloader.get_trainvaltest_dataloader(mode)



def prepare_result_for_saj(tilde_X,Xc,X,X_holdout,missing_mask,indicating_mask):
    reconstruction_loss = 0.
    reconstruction_loss += masked_mae_cal(tilde_X,X,missing_mask)
    imputation_loss = masked_mae_cal(Xc,X_holdout,indicating_mask) if X_holdout is not None else torch.tensor(0.)

    return {
        "reconstruction_loss":reconstruction_loss,
        "imputation_loss":imputation_loss
    }



def result_process(config,result):
    result['total_loss'] = torch.tensor(0.,device=config['device'])
    result['reconstruction_loss'] = (result['reconstruction_loss']*config['reconstruction_loss_wt'])
    result['imputation_loss'] = (result['imputation_loss']*config['imputation_loss_wt'])
    if config['MIT']:
        result['total_loss'] += result['imputation_loss']
    if config['ORT']:
        result['total_loss']+=result['reconstruction_loss']
    return result

def get_model(config):
    model = build_SAJ(config['seq_len'],config['feature_num']).to(config['device'])
    return model





def saj_run(model,data,config,mode='train',N=6):
    if not config['MIT'] and mode =='train':
        indices, X, missing_mask = map(lambda x: x.to(config['device']),data)
        X_holdout, indicating_mask = None, None
    else:
        indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x:x.to(config['device']),data)
    first_rep = model.firstrep(X, missing_mask, config['ltm'], config['device'])
    tilde_X1 = model.tilde_X1(X, missing_mask, config['ltm'], config['relu'], config['device'])
    attn = model.sajloop.layers[N-1].self_attention_block.attention_scores
    eta = model.weighted_combination_block(attn)
    Xc = model.compile(eta, first_rep)
    return tilde_X1,Xc,X,X_holdout,missing_mask,indicating_mask





def train_val(config):
    train_dataloader,val_dataloader = get_ds(config,'train'),get_ds(config,'val')
    model = get_model(config)
    optimize = torch.optim.Adam(model.parameters(),lr=0.00038)
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
            tilde_X1, Xc, X, X_holdout, missing_mask, indicating_mask = saj_run(model,data,config)
            result = prepare_result_for_saj(tilde_X1,Xc,X,X_holdout,missing_mask,indicating_mask)
            result = result_process(config,result)
            running_loss += result['total_loss'].item()
            optimize.zero_grad()
            result['total_loss'].backward()
            optimize.step()
        run_result['train_loss'].append(running_loss/len(train_dataloader))
        model.eval()
        evalX_collector, evalMask_collector,imputation_collector = [],[],[]
        with torch.no_grad():
            for idx,data in enumerate(val_dataloader):
                tilde_X1, Xc, X, X_holdout, missing_mask, indicating_mask = saj_run(model, data, config,'val')
                evalX_collector.append(X_holdout)
                evalMask_collector.append(indicating_mask)
                imputation_collector.append(Xc)
            evalX_collector = torch.concat(evalX_collector)
            evalMask_collector = torch.concat(evalMask_collector)
            imputation_collector = torch.concat(imputation_collector)
            imputation_MAE = masked_mae_cal(imputation_collector,evalX_collector,evalMask_collector)
            val_loss = imputation_MAE.cpu().numpy().mean()
        run_result['val_loss'].append(val_loss)
        print(val_loss)
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            current_stopping = early_stopping
            print('Saving the model.....')
            torch.save(model.state_dict(),f"relu_{config['relu']}_ltm_{config['ltm']}_MIT_{config['MIT']}.pt")
        else:
            current_stopping-=1
        if current_stopping==0:
            print('Early Stopping')
            break


def test_model(config):
    model = get_model(config)
    model.load_state_dict(torch.load(f"relu_{config['relu']}_ltm_{config['ltm']}_MIT_{config['MIT']}.pt"))
    test_dataloader = get_ds(config,'test')
    model.eval()
    evalX_collector, evalMask_collector, imputation_collector =[],[],[]
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            tilde_X1, Xc, X, X_holdout, missing_mask, indicating_mask = saj_run(model, data, config,'val')
            evalX_collector.append(X_holdout)
            evalMask_collector.append(indicating_mask)
            imputation_collector.append(Xc)
        evalX_collector = torch.concat(evalX_collector)
        evalMask_collector = torch.concat(evalMask_collector)
        imputation_collector = torch.concat(imputation_collector)
        return evalX_collector, evalMask_collector, imputation_collector,model


if __name__=="__main__":
    model_config = configuration_saj()
    test_printouts = 'ablation_study.txt'
    file_stdout = sys.stdout
    mae_collector = []
    rmse_collector = []
    mre_collector = []
    time_collector = []
    for seed in get_random_seeds():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_val(model_config)

       # After saving the best model, use the model to test on different missing percentage by uncommenting the below lines of code
    #
        evalX_collector, evalMask_collector,imputation_collector,model = test_model(model_config)
        mae = masked_mae_cal(imputation_collector,evalX_collector,evalMask_collector)
        rmse = masked_rmse_cal(imputation_collector,evalX_collector,evalMask_collector)
        mre = masked_mre_cal(imputation_collector,evalX_collector,evalMask_collector)
        mae_collector.append(mae.item())
        rmse_collector.append(rmse.item())
        mre_collector.append(mre.item())
    mean_mae, mean_rmse, mean_mre = (np.mean(mae_collector),np.mean(rmse_collector),np.mean(mre_collector))
    std_mae, std_rmse, std_mre = (np.std(mae_collector),np.std(rmse_collector),np.std(mre_collector))
    with open(test_printouts,'a') as f:
        sys.stdout = f
        print(
            "\n'USGSMuddyFK:\n"
            "Missing percentage: 20%\n"
            f"Averaged SAJ with relu {model_config['relu']}, ltm {model_config['ltm']}, MIT {model_config['MIT']}:\n"
            f"MAE={mean_mae:.4f} ± {std_mae:.4f}, "
            f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f}, "
            f"MRE={mean_mre:.4f} ± {std_mre:.4f}, "
        )
    sys.stdout = file_stdout
