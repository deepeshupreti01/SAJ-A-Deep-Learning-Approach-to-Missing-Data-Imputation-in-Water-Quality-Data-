import h5py
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import csv
import torch
from brits import build_BRITS
from saj import build_SAJ
from saits import build_SAITS
from dataloader import UnifiedDataloader
from trainvaltest_models import saits_run, saj_run, brits_run_val
from config import *
from datetime import datetime, timedelta

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def training_vs_validation(file_path,epoch_limit:25):
    '''
    file_path: Provide path to csv file that will be used for plotting training loss vs validation loss
    epoch_limit: Provide number of epochs that will be plotted
    This will automatically save the plot in 'Plot' folder
    '''
    #To read epochs, train_loss and val_loss
    df = pd.read_csv(file_path)
    #initialize plot and configure figure size
    plt.figure(figsize=(11,6))
    #Plot training loss with marker of o in continuous line fashion of blue color and label it as training loss. Change the size of marker by markersize
    plt.plot(df['epochs'], df['train_loss'], marker = 'o', linestyle = '-', color='b', label='Training Loss',markersize = 3)
    plt.plot(df['epochs'],df['val_loss'],marker='o', linestyle = '-', color = 'r', label='Validation Loss',markersize = 3)
    #Label the epochs and fontsize and fontweight
    plt.xlabel('Epochs',fontsize=12,fontweight='bold')
    plt.ylabel('Mean Absolute Error',fontsize=12,fontweight='bold')
    #To change the main bounding box (formed by axis) of the graph
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.3)
    #Start from 1 in epochs and 0 to 0.8 in loss axis (Change according to dataset)
    ax.set_xlim(left=1,right=epoch_limit)
    ax.set_ylim(bottom=0,top=0.8)
    #Increase the ticks in the axis
    ax.tick_params(axis='both',which='major',labelsize=5)
    #Increase font size of the label numbers
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    #For legend
    plt.legend(fontsize=10)
    #For grids
    # plt.grid()  #For grid in both X and Y
    # plt.grid(True,axis='y',which='both',linestyle='-',linewidth=0.5) #For grid, perpendicular to Y axis
    plt.savefig(f'Plot/Training_vs_Validation_MAE_USGSMuddyFK_{epoch_limit}.png',dpi=1200, bbox_inches='tight')
    plt.show()

def actual_vs_imputed(config,hyperparameters,seed):
    model_saj = build_SAJ(config['seq_len'], config['feature_num'], hyperparameters['SAJ']['d_model'], hyperparameters['SAJ']['d_v'],
                      hyperparameters['SAJ']['d_inner'], hyperparameters['SAJ']['d_conv'], hyperparameters['SAJ']['h'], hyperparameters['SAJ']['N'],
                      hyperparameters['SAJ']['dropout'], hyperparameters['SAJ']['conv_dropout'], hyperparameters['SAJ']['attn_dropout']).to(config['device'])
    model_saits = build_SAITS(hyperparameters['SAITS']['N'], hyperparameters['SAITS']['n_inner_groups'], config['seq_len'],
                        config['feature_num'], hyperparameters['SAITS']['d_model'], hyperparameters['SAITS']['d_ff'],
                        hyperparameters['SAITS']['h'], hyperparameters['SAITS']['d_model']//hyperparameters['SAITS']['h'], hyperparameters['SAITS']['d_v'], hyperparameters['SAITS']['dropout'],
                        hyperparameters['SAITS']['attn_dropout']).to(config['device'])
    model_brits = build_BRITS(config['seq_len'], config['feature_num'], hyperparameters['BRITS']['hidden_size'],
                        hyperparameters['BRITS']['dropout'], True, config['device']).to(config['device'])
    model_saj.load_state_dict(torch.load(f"{config['best_model_dir']}/SAJ_{seed}.pt"))
    model_saits.load_state_dict(torch.load(f"{config['best_model_dir']}/SAITS_{seed}.pt"))
    model_brits.load_state_dict(torch.load(f"{config['best_model_dir']}/BRITS_{seed}.pt"))
    # batch_idx, feature_idx = 0, 1 #for discharge in USGSMuddyFK
    batch_idx, feature_idx = 1, 6 #for DO in USGSOhioRiver
    for model in ['SAITS','BRITS','SAJ']:
        unified_dataloader = UnifiedDataloader(config['dataset_path'],config['seq_len'],config['feature_num'],model,config['batch_size'],config['num_workers'],True)
        test_dataloader = unified_dataloader.get_trainvaltest_dataloader('test')
        for i,data in enumerate(test_dataloader):
            if model == 'SAITS':
                _,_,tilde_X3,_,X,X_holdout,missing_mask,indicating_mask = saits_run(model_saits, data, config)
                saits_Xc = missing_mask * X + (1 - missing_mask) * tilde_X3
            elif model == 'SAJ':
                _,tilde_X,_,_,_,_ = saj_run(model_saj, data, config, hyperparameters['SAJ']['N'])
                saj_Xc = missing_mask * X + (1 - missing_mask) * tilde_X
            else:
                brits_ret, _, _ = brits_run_val(model_brits, data, config)
                tilde_X = brits_ret['imputed_data']
                brits_Xc = missing_mask * X + (1 - missing_mask) * tilde_X
            if i == batch_idx:
              break
  #  #For Discharge
  #   X_holdout = X_holdout[batch_idx,4:12,feature_idx].detach().cpu().numpy()
  #   saj_Xc = saj_Xc[batch_idx,4:12,feature_idx].detach().cpu().numpy()
  #   saits_Xc = saits_Xc[batch_idx,4:12,feature_idx].detach().cpu().numpy()
  #   brits_Xc = brits_Xc[batch_idx,4:12,feature_idx].detach().cpu().numpy()
    #For DO
    X_holdout = X_holdout[batch_idx,:12,feature_idx].detach().cpu().numpy()
    saj_Xc = saj_Xc[batch_idx,:12,feature_idx].detach().cpu().numpy()
    saits_Xc = saits_Xc[batch_idx,:12,feature_idx].detach().cpu().numpy()
    brits_Xc = brits_Xc[batch_idx,:12,feature_idx].detach().cpu().numpy()
    df = pd.read_csv(config['plot_actual_vs_imputed'],header=None)
    train_mean_for_feature = df.iloc[:preprocessing_info()['splitdata'][0], feature_idx+1].mean()
    train_std_for_feature = df.iloc[:preprocessing_info()['splitdata'][0], feature_idx+1].std()
    X_holdout = X_holdout * train_std_for_feature + train_mean_for_feature
    saj_Xc = saj_Xc * train_std_for_feature + train_mean_for_feature
    saits_Xc = saits_Xc * train_std_for_feature + train_mean_for_feature
    brits_Xc = brits_Xc * train_std_for_feature + train_mean_for_feature
    # #For discharge
    # date_time = df.iloc[config['datetime_data_for_graph']+4:config['datetime_data_for_graph']+12, 0]
    # date_time = pd.to_datetime(date_time)
    #For DO
    date_time = df.iloc[config['datetime_data_for_graph']:config['datetime_data_for_graph']+12, 0]
    date_time = pd.to_datetime(date_time)
    #initialize plot and configure figure size
    fig, ax = plt.subplots(figsize=(11,6))
    plt.plot(date_time, X_holdout, label='Ground Data', color='black', zorder = 2)
    plt.plot(date_time, saj_Xc, label='SAJ', color='green', zorder = 1)
    plt.plot(date_time, saits_Xc, label='SAITS', color='blue', zorder = 1)
    plt.plot(date_time, brits_Xc, label='BRITS', color='red', zorder = 1)
    #Label the DateTime and fontsize and fontweight
    plt.xlabel('DateTime',fontsize=12,fontweight='bold')
    # plt.ylabel('Discharge (ft3/s)',fontsize=12,fontweight='bold')
    plt.ylabel('Dissolved Oxygen (mg/l)',fontsize=12,fontweight='bold')
    plt.xlim(min(date_time),max(date_time))

    #set parts to grey to indicate missingness
    # #For Discharge
    # ax.axvspan(min(date_time),min(date_time)+timedelta(minutes=15),color='white',alpha=0.3)
    # ax.axvspan(min(date_time)+timedelta(minutes=15),min(date_time)+timedelta(minutes=45),color='grey',alpha=0.3)
    # ax.axvspan(min(date_time)+timedelta(minutes=45),min(date_time)+timedelta(minutes=60),color='white',alpha=0.3)
    # ax.axvspan(min(date_time)+timedelta(minutes=60),min(date_time)+timedelta(minutes=90),color='grey',alpha=0.3)
    # ax.axvspan(min(date_time)+timedelta(minutes=90),min(date_time)+timedelta(minutes=135),color='white',alpha=0.3)
    #For DO
    ax.axvspan(min(date_time),min(date_time)+timedelta(minutes=60),color='grey',alpha=0.3)
    ax.axvspan(min(date_time)+timedelta(minutes=60),min(date_time)+timedelta(minutes=75),color='white',alpha=0.3)
    ax.axvspan(min(date_time)+timedelta(minutes=75),min(date_time)+timedelta(minutes=150),color='grey',alpha=0.3)
    ax.axvspan(min(date_time)+timedelta(minutes=150),min(date_time)+timedelta(minutes=165),color='white',alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    #To change the main bounding box (formed by axis) of the graph
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.3)
    #Increase the ticks in the axis
    plt.gca().tick_params(axis='both',which='major',labelsize=5)
    #Increase font size of the label numbers
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #For legend
    plt.legend(fontsize = 10)
    #For grids
    # plt.grid()  #For grid in both X and Y
    # plt.grid(True,axis='y',which='both',linestyle='-',linewidth=0.5) #For grid, perpendicular to Y axis
    # plt.savefig('Plot/ActualvsImputed_MAE_Discharge_USGSMuddyFK.png',dpi=1200, bbox_inches='tight')
    plt.savefig('Plot/ActualvsImputed_MAE_DO_USGSOhioRiver.png',dpi=1200,bbox_inches='tight') #For DO
    plt.show()

if __name__=="__main__":
    training_vs_validation('USGSMuddyFK_train_loss_vs_val_loss.csv',100)
    # config = get_info()
    # hyperparameters = get_hpo_results()
    # actual_vs_imputed(config,hyperparameters,seed=3)

