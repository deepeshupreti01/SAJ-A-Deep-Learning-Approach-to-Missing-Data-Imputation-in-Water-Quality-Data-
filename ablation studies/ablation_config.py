import torch
def configuration_saj():
    return {
        #dataloader configuration
        'dataset_path':r'C:\Users\ASUS\Desktop\Impuation\USGSMuddyFK\20_datasets.h5',
        'seq_len':24,
        'feature_num':6, #8 for USGSOhioRiver and 6 for USGSMuddyFK
        'batch_size':64,
        'num_workers':1,
        #Test of different methodologies that could be involved
        'relu':False,
        'ltm':False, #if false then diagonal masking
        'MIT':True,
        'ORT':True,
        #model configuration
        'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs':100,
        'earlystopping':20,
        'reconstruction_loss_wt':1.,
        'imputation_loss_wt':1.
    }

def get_random_seeds():
    #pi = 3.14159265
    return [3,14,15,92,65]