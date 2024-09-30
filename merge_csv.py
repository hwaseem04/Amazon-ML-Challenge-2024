import pandas as pd
import os
from natsort import natsorted

pred_folder = './inference_csv'
csv_files = [f for f in os.listdir(pred_folder) if f.endswith('.csv')]

# Sort the files in natural order
csv_files = natsorted(csv_files)


dfs = [pd.read_csv(os.path.join(pred_folder, file)) for file in csv_files]
final_pred = pd.concat(dfs, ignore_index=True)

test = pd.read_csv('./dataset/test.csv')
# Replace the index of test with the index of final_pred
final_pred['index'] = test['index']

final_pred.to_csv('reindex_pred.csv', index=False)
