import pandas as pd

def quarter_hourly(input_path:str,with_date:bool):
    df = pd.read_csv(input_path,delimiter='\t')
    columns_to_remove = [col for col in df.columns if len(col.split('_')) == 2 and col.split('_')[1] == 'cd' or len(col.split('_')) == 3]
    df = df.drop(['site_no']+columns_to_remove,axis=1)
    df = df.drop(index=0).reset_index(drop=True)
    columns_original_name = {
      '60130_00065':'Gage height,ft',
      '60131_00060':'Discharge, ft3/s',
      '60133_00010':'Temperature, Celsius',
      '60134_00300':'Dissolved Oxygen, mg/l',
      '60135_00400':'pH,standard units',
      '60136_00095':'Specific conductance, microsiemens/cm at 25 deg Celsius'
    }
    df = df.rename(columns=columns_original_name)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.astype('float')
    quarterhourly_data = df.resample('15T').mean()
    quarterhourly_data = quarterhourly_data.reset_index()
    quarterhourly_data.set_index('datetime',inplace=True)
    if with_date:
        quarterhourly_data.to_csv("../../preprocessed_data/USGSMuddyFK_with_date.csv",index=True,header=False)
    else:
        quarterhourly_data.to_csv("../../preprocessed_data/USGSMuddyFK.csv",index=False,header=False)
    print('Saved Successfully')


if __name__=="__main__":
    quarter_hourly('data.txt',with_date=True)