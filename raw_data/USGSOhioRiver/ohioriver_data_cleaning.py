import pandas as pd

def quarter_hourly(input_path:str,with_date:bool):
    df = pd.read_csv(input_path,delimiter='\t')
    columns_to_remove = [col for col in df.columns if len(col.split('_')) == 2 and col.split('_')[1] == 'cd' or len(col.split('_')) == 3]
    df = df.drop(['323512_00065','site_no','220312_00065']+columns_to_remove,axis=1)
    df = df.drop(index=0).reset_index(drop=True)
    columns_original_name = {
    '166869_00060': 'Discharge',
    '268332_63680': 'Turbidity',
    '60629_00065': 'Gage height, Headwater',
    '60634_00045': 'Precipitation, inches',
    '60636_99133': 'Nitrate plus nitrite, insitu',
    '60638_00010': 'Temperature, water',
    '60639_00095': 'Specific conductance',
    '60640_00300': 'Dissolved Oxygen',
    '60641_00400':'pH'
    }
    df = df.rename(columns=columns_original_name)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.replace('Eqp',float('nan'))
    df = df.astype('float')
    quarterhourly_data = df.resample('15T').mean()
    quarterhourly_data = quarterhourly_data.reset_index()
    quarterhourly_data.set_index('datetime',inplace=True)
    if with_date:
        quarterhourly_data.to_csv("../../preprocessed_data/USGSOhioRiver_with_date.csv",index=True,header=False)
    else:
        quarterhourly_data.to_csv("../../preprocessed_data/USGSOhioRiver.csv",index=False,header=False)
    print('Saved Successfully')


if __name__=="__main__":
    quarter_hourly('data.txt',with_date=False)