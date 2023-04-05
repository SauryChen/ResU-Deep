# This code is used to construct a global model.
# The dataloader should be modified since the large size of variables.
# The file should be opened every time when reading the data.
# Time ranges from 2015-2019, JJA

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import classification_report
import copy


from Unet_res_up import Unet_Res_up
from read_nc_15_19 import load_variables

locs = [
    [30, 0, 64, 134],
    [30, 0, 162, 232],
    [30, 0, 240, 310],
    [30, 0, 290, 360]
]

batch_size = 64
time_length = 11040 # should be modified according to shape[0] of .nc file
labels = np.load('/ceph-data/cmx/TRMM/deep_label.npy')

# load data
class ncdata(torch.utils.data.Dataset):
    def __init__(self, locs, time_length, flag, labels, transform = None):
        super(ncdata, self).__init__()
        self.locs = locs
        self.time_length = time_length
        self.transform = transform
        self.labels = labels
        
        self.t_loc = []
        
        self.interval = self.time_length // 3
        self.interval3 = int(self.interval / 5 * 3)
        self.interval4 = int(self.interval / 5 * 4)
        
        if flag == 'TRAIN':
            for t in range(2, self.interval3):
                for loc in locs:
                    self.t_loc.append({"time": t,
                                    "location": loc})
        elif flag == 'VAL':
            for t in range(self.interval3+2, self.interval4):
                for loc in locs:
                    self.t_loc.append({"time": t,
                                    "location": loc})
        elif flag == 'TEST':
            for t in range(self.interval4+2, self.interval):
                for loc in locs:
                    self.t_loc.append({"time": t,
                                    "location": loc})            
        else:
            print("Error: Wrong flag in dataloader.")
        
    def __len__(self):
        return len(self.t_loc)
    
    def __getitem__(self, index):
        t_loc_ref = self.t_loc[index]
        t = t_loc_ref['time']
        loc = t_loc_ref['location']
        
        data_x = load_variables(loc, t)
        
        #label_x1 = load_label(loc, t-1)
        #label_x2 = load_label(loc, t-2)
        #label = load_label(loc, t)
        label_x1 = labels[t-1, :, (loc[2]*4):(loc[3]*4):1]
        label_x2 = labels[t-2, :, (loc[2]*4):(loc[3]*4):1]
        label = labels[t, :, (loc[2]*4):(loc[3]*4):1]
        data = np.dstack((data_x, label_x1, label_x2)) # should be (121, 280, 33)
        
        channel_mean = []
        channel_std = []
        for ch in range(0,data.shape[2]):
            channel_mean.append(np.mean(data[:,:,ch]))
            channel_std.append(np.std(data[:,:,ch]))
        
        #print(channel_mean)
        #print(channel_std)
        if(channel_std[10] == 0): # ssrd
            channel_std[10] = 1
        
        if self.transform == None:
            self.trans = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(tuple(channel_mean), tuple(channel_std))
                                        ])
        data = self.trans(data)
        
        return data, label

def create_dataloader(locs, time_length, flag, labels, batch_size):
    datasource = ncdata(locs=locs, time_length=time_length, flag=flag, labels = labels, transform=None)
    dataloader = DataLoader(datasource, batch_size=batch_size, shuffle = False, drop_last=False)
    return dataloader


def train(net, epoch, optimizer):
    net.train()
    loss = 0

    for batch_idx, (features, labels) in enumerate(trainloader):
        
        features = features.float()
        labels = labels.float()
        
        features = features.to(device)
        labels = labels.to(device)
        
        features, labels = Variable(features), Variable(labels)
        optimizer.zero_grad()
        
        pred = net(features)
        
        #loss = F.binary_cross_entropy_with_logits(pred, labels)
        
        # BCE loss
        #loss_func = torch.nn.BCELoss(reduction = 'mean')
        #labels = labels.unsqueeze(1)
        #loss = loss_func(pred, labels)

        #loss = weighted_BCE(pred, labels, reduction = 'mean')

        # Focal loss
        #loss_func = FocalLoss(alpha=0.25, gamma=2, size_average=True)
        #loss = loss_func(pred, labels)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{: .4f}'.format(
            epoch, batch_idx*len(features), len(trainloader.dataset), \
            100*batch_idx/len(trainloader), loss))

    return net

def validation(net, total_validation_loss):
    net.eval()
    validation_loss = 0
    count = 0
    with torch.no_grad():
        for (features, labels) in valloader:
            count += 1
            
            labels = labels.float()
            features = features.float()
            features = features.to(device)
            labels = labels.to(device)

            features, labels = Variable(features), Variable(labels)
            pred_validation = net(features)

            #test_loss += F.binary_cross_entropy_with_logits(pred, labels).item()
            
            # BCE loss
            #loss_func = nn.BCELoss(reduction = 'sum')
            #labels = labels.unsqueeze(1)
            #validation_loss += loss_func(pred_validation, labels).item()
            #loss = weighted_BCE(pred, labels, reduction = 'mean')
            
        validation_loss /= len(valloader.dataset)
        total_validation_loss.append(validation_loss)
        print('Validation set: Average loss: {:.4f}'.format(validation_loss))
        print('\n')
    return validation_loss, total_validation_loss

def test(model, epoch):
    model.eval()
    count = 0
    test_loss = 0
    with torch.no_grad():
        for (features, labels) in testloader:
            count += 1
            features = features.float()
            features = features.to(device)
            labels = labels.float()
            labels = labels.to(device)
            
            features, labels = Variable(features), Variable(labels)
            pred_test = model(features)
            
            #test_loss += F.binary_cross_entropy_with_logits(pred, labels).item()
            
            # BCE loss
            #loss_func = nn.BCELoss(reduction = 'sum')
            #labels = labels.unsqueeze(1)
            #test_loss += loss_func(pred_test, labels).item()
            #loss = weighted_BCE(pred, labels, reduction = 'mean')
            
            # to CPU
            pred_test = pred_test.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            
            if (count == 1):
                total_pred_test = pred_test
                total_label = labels
            elif (count > 1):
                total_pred_test = np.concatenate((total_pred_test,pred_test),axis=0)
                total_label = np.concatenate((total_label, labels), axis=0)

        test_loss /= len(testloader.dataset)
        print('Test set: Average loss: {:.4f}'.format(test_loss))
    
    print("Converting to numpy has the shape: ", total_pred_test.shape)
    np.save('/ceph-data/cmx/ERA5_15_19/major_revision_experiments/global_model_results/total_pred_test_{}.npy'.format(str(epoch)), total_pred_test)
    return total_pred_test, total_label

def class_metric(model, epoch):
    print(">>>>>>>>>> test with sklearn classification report. <<<<<<<<<<")
    total_pred_test, total_label = test(model, epoch)
    total_label = total_label.reshape(-1)
    
    file = '/ceph-data/cmx/ERA5_15_19/major_revision_experiments/global_model_results/classification_report_{}.txt'.format(str(epoch))
    fc = open(file, 'a')
    for threshold in np.arange(0.05, 1, 0.05):
        total_pred_class = copy.deepcopy(total_pred_test)
        print(threshold)
        print("********************************", file = fc)
        print(threshold, file = fc)
        np.place(total_pred_class, total_pred_class < threshold, 0)
        np.place(total_pred_class, total_pred_class >= threshold, 1)

        
        sk_Y_test = total_label
        sk_pred_class = total_pred_class.reshape(-1)
        target_names = ['class 0', 'class 1']
        print(classification_report(sk_Y_test, sk_pred_class, target_names = target_names), file = fc)
    fc.close()
    

cuda_id = sys.argv[1]
trainloader = create_dataloader(locs=locs, time_length=time_length, flag='TRAIN', labels = labels, batch_size=batch_size)
valloader = create_dataloader(locs=locs, time_length=time_length, flag='VAL', labels = labels, batch_size=batch_size)
testloader = create_dataloader(locs=locs, time_length=time_length, flag='TEST', labels = labels, batch_size=batch_size)

#train_data, train_label = next(iter(trainloader))
#print("After dataloader, ", train_data.size(), train_label.size())

device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
model = Unet_Res_up(in_ch = 33, out_ch = 1)
model = model.to(device)
print('device name, (device): ', torch.cuda.get_device_name(0), device)

# train
optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum = 0.8)
print(optimizer)
total_validation_loss = []
best_model = None
best_loss = 10000
best_epoch = 0

for epoch in range(1,100):
    model = train(model, epoch, optimizer)
    validation_loss, total_validation_loss = validation(model, total_validation_loss)

    if validation_loss < best_loss:
        best_model = copy.deepcopy(model)
        best_loss = validation_loss
        best_epoch = epoch
        
print("Best epoch is Epoch {}".format(best_epoch))
np.save("/ceph-data/cmx/ERA5_15_19/major_revision_experiments/global_model_results/total_validation_loss.npy", total_validation_loss)
checkpoint_path = "/ceph-data/cmx/ERA5_15_19/major_revision_experiments/global_model_results/best_model_{}".format(best_epoch)
state = {'model':best_model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':best_epoch}
torch.save(state, checkpoint_path)

#assert best_model != model, 'best_model == last model'
class_metric(best_model, best_epoch)
