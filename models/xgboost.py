# This code is used train an NN model for each column

import sys
import time
import copy
import pickle
import numpy as np
import netCDF4 as nc
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

from functions import read_nc, location_range, load_variables, load_label
       
location = sys.argv[1]
n_estimators = sys.argv[2]
max_depth = sys.argv[3]
print(location, n_estimators, max_depth)
savefile = '/ceph-data/cmx/ERA5_15_19/major_revision_experiments/xgb_results/model_{}_{}_{}_pos35.pickle'.format(location, str(n_estimators), str(max_depth))

lon_st, lon_ed = location_range(location=location)
loc = [30, 0, lon_st, lon_ed]

label = load_label(lon_st=lon_st, lon_ed=lon_ed)
label_x1 = label[np.newaxis,1:-1:1,:,:]
print(label_x1.shape)
label_x2 = label[np.newaxis,0:-2:1,:,:]

data_x = load_variables(loc, location, lon_st, lon_ed)
data_x = data_x[:,2::,:,:]


print("After delete first two t data_x has the shape: ", data_x.shape)
data = np.vstack((data_x, label_x1, label_x2)).reshape(33,-1)
label = label[2::,:,:]
label = label.reshape(-1)
del data_x
print("Data has the shape: ", data.shape)
print("Label has the shape: ", label.shape)
data = np.transpose(data)
print("After transpose, data has the shape: ", data.shape)
print("After transpose, label has the shape: ", label.shape)

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

interval = data.shape[0] // 5
train_x, train_y = data[0:interval*3,:], label[0:interval*3]
val_x, val_y = data[interval*3:interval*4,:], label[interval*3:interval*4]
test_x, test_y = data[interval*4::,:], label[interval*4::]
print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape)

start = time.time()
xgb = XGBClassifier(n_estimators=int(n_estimators), nthread=8, max_depth=int(max_depth), scale_pos_weight = 3.5, reg_lambda=0.001)
xgb.fit(train_x, train_y)
end = time.time()
duration = end - start
print('Training time: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

predict_val = xgb.predict(val_x)
predict_test = xgb.predict(test_x)
print("train score: ", xgb.score(train_x, train_y))
print("val score: ", xgb.score(val_x, val_y))
print("test score: ", xgb.score(test_x, test_y))

tn, fp, fn, tp = confusion_matrix(test_y, predict_test).ravel()
print(tn, fp, fn, tp)
f1 = f1_score(test_y, predict_test, average='macro')
P = precision_score(test_y, predict_test, average='macro')
R = recall_score(test_y, predict_test, average='macro')
print(f1,P,R)

with open(savefile,'wb') as f:
    pickle.dump(xgb, f)
