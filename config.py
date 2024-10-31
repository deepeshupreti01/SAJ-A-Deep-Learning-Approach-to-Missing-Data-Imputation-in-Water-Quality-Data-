import torch

def preprocessing_info():
    return {
        'path':'preprocessed_data/USGSOhioRiver.csv',
        'splitdata':[245482,300085,350688],     #[245482,300085,350688] for USGSOhioRiver, [98256,112296,140352] for USGSMuddyFK
        'seq_len':24,  # USGSOhioRiver data and USGSMuddyFK
        'artificial_missing_rate':0.5,    #50% missing in both Ohio and Muddy
        'saving_dir':'USGSOhioRiver'
    }

def get_random_seeds():
    #pi = 3.14159265
    return [3,14,15,92,65]


def get_hpo_results():
    dataset = get_info()['best_model_dir'].split('/')[1]
    if dataset == "USGSOhioRiver":
        return {
            'SAITS': {
                'd_model': 512,
                'd_v': 64,
                'd_ff': 2048,
                'N': 7,
                'h': 4,
                'n_inner_groups': 1,
                'dropout': 0,
                'attn_dropout': 0.1,
                'optimizer': 'adamw',
                'lr': 0.00019471587821872846
            },
            'SAJ': {
                'd_model': 64,
                'd_v': 64,
                'd_conv': 16,
                'd_inner': 256,
                'N': 6,
                'h': 8,
                'dropout': 0,
                'conv_dropout':0.1,
                'attn_dropout':0,
                'optimizer': 'adam',
                'lr': 0.00038

            },
            'BRITS': {
                'hidden_size': 256,
                'dropout': 0.5,
                'optimizer': 'adam',
                'lr': 0.0020552078037207556
            }
        }
    elif dataset == 'USGSMuddyFK':
        return {
            'SAITS': {
                'd_model': 512,
                'd_v': 512,
                'd_ff': 512,
                'N': 4,
                'h': 2,
                'n_inner_groups': 1,
                'dropout': 0,
                'attn_dropout': 0,
                'optimizer': 'adam',
                'lr': 0.00012590551987234013
            },
            'SAJ': {
                'd_model': 64,
                'd_v': 64,
                'd_conv': 16,
                'd_inner': 256,
                'N': 6,
                'h': 8,
                'dropout': 0,
                'conv_dropout':0.1,
                'attn_dropout':0,
                'optimizer': 'adam',
                'lr': 0.00038

            },
            'BRITS': {
                'hidden_size': 512,
                'dropout': 0.2,
                'optimizer': 'adam',
                'lr': 0.006024755519894964
            }
            }
    else:
        raise ValueError



def get_info():
    return {
        #Model_info
        'model_type': 'BRITS',
        #Directory info
        'dataset_path': f"{preprocessing_info()['saving_dir']}/{int(preprocessing_info()['artificial_missing_rate']*100)}_datasets.h5",
        #Hyperparameters and values
        "seq_len":24, #24 for USGSOhioRiver, 24 for USGSMuddyFK
        "batch_size":64,
        "feature_num":8, #8 for USGSOhioRiver, 6 for USGSMuddyFK
        'num_workers':1,
        # Training info
        "epochs":100,
        "earlystopping":20,
        'device':'cuda' if torch.cuda.is_available() else "cpu",
        'ORT':True,
        #Loss tuning
        'imputation_loss_wt':1.,
        'reconstruction_loss_wt':1.,
        'consistency_loss_wt':1.,
        #Best model saving dir
        'best_model_dir':f'best_models/{preprocessing_info()["saving_dir"]}',
        #Save train and val loss
        'save_train_vs_val_loss':False,
        'store_result_in_notepad':True,
        #Graph
        'plot_actual_vs_imputed':'preprocessed_data/USGSOhioRiver_with_date.csv',
        'datetime_data_for_graph': 301645   #301645 for DO of USGSOhioRiver 50% missingness 112296 USGSMuddyFK 50% missingness     Add 1 to search in excel
    }

def get_result():
    folder  = get_info()['best_model_dir'].split('/')[1]
    return {
        'test_result' : f'output/{folder}/output.txt',
    }