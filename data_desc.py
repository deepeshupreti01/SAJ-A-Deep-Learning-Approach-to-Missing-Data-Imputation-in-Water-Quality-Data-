import pandas as pd
import sys
from config import preprocessing_info

def data_description(config):
  path = config['path']
  file_description_path = 'data_desc_and_stats/description_data.txt'
  train_data, val_data, _ = config['splitdata']
  df = pd.read_csv(path,header=None)
  dataset = path.split('/')[1].split('_')[0]
  if dataset == 'USGSOhioRiver':
    column_list = ['DateTime', 'Discharge (ft3/s)', 'Turbidity (FNU)', 'Gage height (ft)', 'Nitrate plus nitrite insitu (ml/l as nitrogen)', 'Temperature (Cels)', 'Specific conductance (microsiemens/cm at 25 deg cels)', 'DO, (mg/l)', 'pH(standard units)']
  elif dataset == 'USGSMuddyFK':
    column_list = ['DateTime', 'Gage Height (ft)', 'Discharge (ft3/s)', 'Temp (Cels)', 'DO (mg/l)', 'pH (standard units)', 'Specific conductance (microsiemens/cm at 25 deg Cels)']
  else:
    raise ValueError ('No dataset by that name')
  df.columns = column_list
  origin = 'USGS'
  data_type = 'downloaded type .FILE extension'
  source_of_download = 'https://waterdata.usgs.gov/nwis/qw'
  start_date = df.iloc[0,0]
  end_date = df.iloc[-1,0]
  no_of_rows = len(df)
  del column_list[0]
  no_of_features = len(column_list)
  data_recording_freq = '15 mins'
  seq_len = 24
  with open(file_description_path,'a') as f:
    ori_stdout = sys.stdout
    sys.stdout = f
    print(f'\t{dataset}\n')
    print('------------------------------------\n')
    print(f'origin: {origin}\n')
    print(f'data type: {data_type}\n')
    print(f'source of download: {source_of_download}\n')
    if dataset == 'USGSOhioRiver':
      print(f'Station number: USGS 03612600 OHIO RIVER AT OLMSTED, IL\n')
      print(f'Decimal Latitude: 37.17921867\n')
      print(f'Decimal Longitude: -89.058404	\n')
    elif dataset == 'USGSMuddyFK':
      print(f'Station number: USGS 03293530 MUDDY FK AT MOCKINGBIRD VALLEY RD AT LOUISVILLE,KY\n')
      print(f'Decimal Latitude: 38.27645917\n')
      print(f'Decimal Longitude: -85.693573	\n')
    else:
      raise ValueError('Dataset not found/available')
    print(f'start date: {start_date}\n')
    print(f'end date: {end_date}\n')
    print(f'no of rows: {no_of_rows}\n')
    print(f'sequence length: {seq_len}\n')
    print(f'interval: {data_recording_freq}\n')
    print(f'no of features: {no_of_features}\n')
    print(f'Parameters: {column_list}\n')
    df_wo_date = df.drop(columns=['DateTime']).astype('float64')
    print(f'Overall Missing Percentage: {round(df_wo_date.isna().sum().sum()/(no_of_rows*no_of_features)*100,3)}\n')
    print(f'Training data: From {df.iloc[0,0]} To {df.iloc[train_data,0]}\n')
    print(f'Validation data: {df.iloc[train_data,0]} To {df.iloc[val_data,0]}\n')
    print(f'Test data: {df.iloc[val_data,0]} To {df.iloc[-1,0]}\n')
    print('Remarks provided on data for each parameters:\n')
    print('A  Approved for publication -- Processing and review completed.\n')
    print('P  Provisional data subject to revision.\n')
    print('e  Value has been estimated.\n')
    print('------------------------------------\n')
    sys.stdout = ori_stdout
  results = []
  for cols in df_wo_date.columns:
    stats = {
    'Parameter': cols,
    'Min': round(df[cols].min(),3),
    'Max': round(df[cols].max(),3),
    'Mean': round(df[cols].mean(),3),
    'Std Dev': round(df[cols].std(),3),
    'Skewness': round(df[cols].skew(),3),
    'Kurtosis': round(df[cols].kurtosis(),3),
    'Missing %': round(df[cols].isna().sum()/len(df)*100,3)
    }
    results.append(stats)
  pd.DataFrame(results).to_csv(f"data_desc_and_stats/{dataset}_stats.csv")
  print('Content details saved!!!')

if __name__ == '__main__':
  config = preprocessing_info()
  data_description(config)

