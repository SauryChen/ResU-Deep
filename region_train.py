# import packages
import sys
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import classification_report
import copy

# import tested models and loss functions
#from models.Unet import Unet
from losses.focalloss import FocalLoss
from losses.weightedBCE import weighted_BCE
#from models.Unet_up import Unet_up
from models.Unet_Res_up import Unet_Res_up
#from models.Unet_res import Unet_res

class CustomDataset_hist(torch.utils.data.Dataset):
    def __init__(self, dataset, label, transform = None):
        self.data = dataset
        self.labels = label
        self.transform = transform       
    def __len__(self):
        #return len(self.data.shape[0])
        return len(self.data)   
    def __getitem__(self, index):
        #data = torch.cat([self.data[index,:,:,:] , self.labels[index-1,:,:,:], self.labels[index-2,:,:,:], self.labels[index-3,:,:,:]], dim = 0)
        data = torch.cat([self.data[index,:,:,:] , self.labels[index-1,:,:,:], self.labels[index-2,:,:,:]], dim = 0)
        labels = self.labels[index,:,:,:]
        data = self.transform(data)
        return data, labels

# load dataset
def read_nc(filename, variable, level, loc):
    # lat: 30 - 0
    # lon: -180 - 180
    # time: 3 hour resolution
    # read .nc file
    # dim = (time, lat, lon) OR (time, level, lat, lon)
    # filename = {cape [cape], cin [cin], cloud [tciw, tclw, tcwv], cp [cp], moisture [p84.162] , \
    #    rh [r, (300,500,700)], sh [q, (300,500,700), soil_boundary [blh (boundary layer height), \
    #    surface temperature [st], surlh [slhf], sursh [sshf], temperature [t, (300,500,700)], \
    #    u_wind [u, (300,500,700,925)], vertical_velocity [w, (500,700)], v_wind [v, (300,500,700,925)],\
    #    inst_moisture_flux, solar radiation}
    lat_st = loc[0]
    lat_ed = loc[1]
    lon_st = loc[2]
    lon_ed = loc[3]
        
    dir_path = '/ceph-data/cmx/ERA5_15_19/'
        
    file = dir_path + filename + '.nc'
    data_file = nc.Dataset(file)
    
    if (level == -2):
        print(f'********** READING FILE[VARIABLE]: {filename}[{variable}][{level}] **********')
        data_a = data_file[variable][0,:,(lon_st*4):(lon_ed*4+1):1]
        print(data_a.shape)
        return data_a

    elif (level == -1):
        print(f'********** READING FILE[VARIABLE]: {filename}[{variable}][{level}] **********')
        data_a = data_file[variable][::3,:,(lon_st*4):(lon_ed*4+1):1]
        print(data_a.shape)
        return data_a
    
    elif (level == 0):
        if (filename == 'rh' or filename == 'sh' or filename == 'temperature' or filename == 'u_wind' or filename == 'v_wind'):
            print(f'********** READING FILE[VARIABLE]: {filename}[{variable}][{level}] **********')
            data_a = data_file[variable][::3, 0, :, (lon_st*4):(lon_ed*4+1):1]
            print(f'********** READING FILE[VARIABLE]: {filename}[{variable}][{level+1}] **********')
            data_b = data_file[variable][::3, 1, :, (lon_st*4):(lon_ed*4+1):1]
            print(f'********** READING FILE[VARIABLE]: {filename}[{variable}][{level+2}] **********')
            data_c = data_file[variable][::3, 2, :, (lon_st*4):(lon_ed*4+1):1]
            print(data_a.shape, data_b.shape, data_c.shape)
            return data_a, data_b, data_c
        
        elif (filename == 'vertical_velocity'):
            print(f'********** READING FILE[VARIABLE]: {filename}[{variable}][{level}] **********')
            data_a = data_file[variable][::3, 0, :, (lon_st*4):(lon_ed*4+1):1]
            print(f'********** READING FILE[VARIABLE]: {filename}[{variable}][{level+1}] **********')
            data_b = data_file[variable][::3, 1, :, (lon_st*4):(lon_ed*4+1):1]
            print(data_a.shape, data_b.shape)
            return data_a, data_b
        
    else:
        print("Error: Wrong Document!")
        
def location_range(location):
    if location == 'A':
        xticklabel = ['-116', '-106', '-96', '-86', '-76', '-66', '-56']
        batch_size = 16
        lon_st = 64
        lon_ed = 134
        print("Retrieving America.")
    elif location == 'B':
        xticklabel = ['-18', '-8', '2', '12', '22', '32', '42', '52']
        batch_size = 16
        lon_st = 162
        lon_ed = 232
        print("Retrieving Africa.")
    elif location == 'C':
        xticklabel = ['60', '70', '80', '90', '100', '110', '120', '130']
        batch_size = 16
        lon_st = 240
        lon_ed = 310
        print("Retrieving Asia.")
    elif location == 'D':
        xticklabel = ['130', '140', '150', '160', '170', '180']
        batch_size = 16
        lon_st = 290
        lon_ed = 360
        print("Retriving Marine.")
    else:
        print("Wrong location.")
    print("batch_size = ", batch_size)
    return lon_st, lon_ed, xticklabel, batch_size

def load_variables(loc, location, lon_st, lon_ed):
    numpy_file = '/ceph-data/cmx/ERA5_15_19/' + 'surface_temperature_' + location + '_new.npy'
    start = time.time()
    geopotential = read_nc('geopotential', 'z', -2, loc)
    geoheight = geopotential / 9.80665
    cape = read_nc('cape', 'cape', -1, loc)
    cin = read_nc('cin', 'cin', -1, loc)
    tciw = read_nc('cloud', 'tciw', -1 , loc)
    tclw = read_nc('cloud', 'tclw', -1 , loc)
    tcwv = read_nc('cloud', 'tcwv', -1 , loc)
    #p84_162 = read_nc('moisture', 'p84.162', -1 , loc)
    ie = read_nc('inst_moist', 'ie', -1, loc)
    blh = read_nc('soil_boundary', 'blh', -1, loc)
    ssrd = read_nc('solar_radiation', 'ssrd', -1, loc)
    st = np.load(numpy_file)
    surlh = read_nc('surlh', 'slhf', -1, loc)
    sursh = read_nc('sursh', 'sshf', -1, loc)
    rh_300, rh_500, rh_700 = read_nc('rh', 'r', 0, loc)
    sh_300, sh_500, sh_700 = read_nc('sh', 'q', 0, loc)
    t_300, t_500, t_700 = read_nc('temperature', 't', 0, loc)
    u_300, u_500, u_700 = read_nc('u_wind', 'u', 0, loc)
    v_300, v_500, v_700 = read_nc('v_wind', 'v', 0, loc)
    u_925 = read_nc('wind_925', 'u', -1, loc)
    v_925 = read_nc('wind_925', 'v', -1, loc)
    ver_500, ver_700 = read_nc('vertical_velocity', 'w', 0, loc)
    end = time.time()
    duration = end - start
    print('Reading Data: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    # prepare dataset:
    geoheight = geoheight.reshape(-1)
    geoheight = np.tile(geoheight, 3680)
    geoheight = geoheight.reshape(3680, 121, -1)
    data_x = np.concatenate((#cape, cin, tciw, tclw, tcwv, p84_162, blh, st, surlh, sursh, ie, ssrd,\
                              cape, cin, tciw, tclw, tcwv, blh, st, surlh, sursh, ie, ssrd,\
                              rh_300, rh_500, rh_700, \
                              sh_300, sh_500, sh_700, \
                              t_300, t_500, t_700, \
                              u_300, u_500, u_700, u_925,\
                              v_300, v_500, v_700, v_925,\
                              ver_500, ver_700, geoheight
                        ), axis = 1)
    print("variable dataset has the shape: ", data_x.shape)
    if (location == 'D'):
        data = data_x.reshape(3680, 31, 121, (lon_ed*4-lon_st*4)) #[time, channel, height, width]
    else:
        data = data_x.reshape(3680, 31, 121, (lon_ed*4-lon_st*4 + 1)) #[time, channel, height, width]
    return data

def load_label(lon_st, lon_ed):

    label = np.load('/ceph-data/cmx/TRMM/deep_label.npy')
    label = label[:,:,(lon_st*4):(lon_ed*4+1):1]

    #print("Data has the shape: ", data.shape)
    print("Label has the shape: ", label.shape)
    return label
def count_label(Y):
    unique, counts = np.unique(Y, return_counts = True)
    return (dict(zip(unique, counts)), counts[0], counts[1], counts[1] / counts[0])
def customed_data(data, label):
    train_val_cut = int(data.shape[0] / 5 * 3)
    val_test_cut = int(data.shape[0] / 5 * 4)
    X_train = data[0:train_val_cut,:,:,:]
    Y_train = label[0:train_val_cut,:,:]
    X_val = data[train_val_cut:val_test_cut, :,:,:]
    Y_val = label[train_val_cut:val_test_cut,:,:]
    X_test = data[val_test_cut:, :,:,:]
    Y_test = label[val_test_cut:,:,:]
    print(X_train.shape, Y_train.shape, X_val.shape, X_test.shape, X_train.shape[0] / X_val.shape[0])
    print("Y_train label count: ", count_label(Y_train))
    print("Y_val label count: ", count_label(Y_val))
    print("Y_test label count: ", count_label(Y_test))

    channel_mean = []
    channel_std = []
    for i in range(0,31):
        channel_std.append(np.std(X_train[:,i,:,:]))
        channel_mean.append(np.mean(X_train[:,i,:,:]))

    # if use historical deep data
    Y_mean = np.mean(Y_train[:,:,:])
    Y_std = np.std(Y_train[:,:,:])
    for i in range(0,2):
        # use two historical deep labels, t-1 and t-2
        channel_mean.append(Y_mean)
        channel_std.append(Y_std)

    channel_std = np.array(channel_std)
    channel_mean = np.array(channel_mean)

    print("Channel mean: ", channel_mean)
    print("Channel mean shape: ", channel_mean.shape)
    print("Channel std: ", channel_std)
    print("Channel std shape: ", channel_std.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, channel_mean, channel_std
def numpytotensor(X,Y):
    X_t = torch.from_numpy(X).type(torch.FloatTensor)
    Y_t = (torch.from_numpy(Y.reshape(X.shape[0], 1, X.shape[2], X.shape[3]))).type(torch.FloatTensor)
    
    return X_t, Y_t
def dataloader(X_train, Y_train, X_val, Y_val, X_test, Y_test, channel_mean, channel_std, batch_size):
    X_train_t, Y_train_t = numpytotensor(X_train, Y_train)
    X_val_t, Y_val_t = numpytotensor(X_val, Y_val)
    X_test_t, Y_test_t = numpytotensor(X_test, Y_test)
    print("Tensor size: ", X_train_t.size(), Y_train_t.size(), X_val_t.size(), Y_val_t.size(), X_test_t.size(), Y_test_t.size())
    
    trans = transforms.Normalize(channel_mean, channel_std)
    train_source = CustomDataset_hist(X_train_t, Y_train_t, transform = trans)
    val_source = CustomDataset_hist(X_val_t, Y_val_t, transform = trans)
    test_source = CustomDataset_hist(X_test_t, Y_test_t, transform = trans)

    dataloaders = {
    'train_set': DataLoader(train_source, batch_size = batch_size, shuffle = False, drop_last = False),
    'val_set': DataLoader(val_source, batch_size = batch_size, shuffle = False, drop_last = False),
    'test_set': DataLoader(test_source, batch_size = batch_size, shuffle = False, drop_last = False)
    }

    train_features, train_labels = next(iter(dataloaders['train_set']))
    test_features, test_labels = next(iter(dataloaders['test_set']))
    print("After dataloader: ", train_features.size(), train_labels.size(), test_features.size(), test_labels.size())

    return dataloaders


def train(net, epoch, optimizer, dataloaders):
    net.train()
    loss = 0

    for batch_idx, (features, labels) in enumerate(dataloaders['train_set']):

        features = features.to(device)
        labels = labels.to(device)
        
        features, labels = Variable(features), Variable(labels)
        optimizer.zero_grad()
        
        pred = net(features)
        
        #loss = F.binary_cross_entropy_with_logits(pred, labels)
        
        # BCE loss
        #loss_func = torch.nn.BCELoss(reduction = 'mean')
        #loss = loss_func(pred, labels)

        # Focal loss
        #loss_func = FocalLoss(alpha=0.25, gamma=2, size_average=True)
        #loss = loss_func(pred, labels)
        
        # weighted BCE
        loss = weighted_BCE(pred, labels, reduction = 'mean')
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{: .4f}'.format(
            epoch, batch_idx*len(features), len(dataloaders['train_set'].dataset), \
            100*batch_idx/len(dataloaders['train_set']), loss))

    return net

def validation(net, total_validation_loss, dataloaders):
    net.eval()
    validation_loss = 0
    count = 0
    with torch.no_grad():
        for (features, labels) in dataloaders['val_set']:
            count += 1
            
            #labels = labels.float()
            #features = features.float()
            features = features.to(device)
            labels = labels.to(device)

            features, labels = Variable(features), Variable(labels)
            pred_validation = net(features)

            #test_loss += F.binary_cross_entropy_with_logits(pred, labels).item()
            
            # BCE loss
            loss_func = nn.BCELoss(reduction = 'sum')
            validation_loss += loss_func(pred_validation, labels).item()

            
        validation_loss /= len(dataloaders['val_set'].dataset)
        total_validation_loss.append(validation_loss)
        print('Validation set: Average loss: {:.4f}'.format(validation_loss))
        print('\n')
    return validation_loss, total_validation_loss

def test(model, dataloaders, epoch):
    model.eval()
    count = 0
    test_loss = 0
    with torch.no_grad():
        for (features, labels) in dataloaders['test_set']:
            count += 1
            #features = features.float()
            features = features.to(device)
            #labels = labels.float()
            labels = labels.to(device)
            
            features, labels = Variable(features), Variable(labels)
            pred_test = model(features)
            
            #test_loss += F.binary_cross_entropy_with_logits(pred, labels).item()
            
            # BCE loss
            loss_func = nn.BCELoss(reduction = 'sum')
            test_loss += loss_func(pred_test, labels).item()

            
            # to CPU
            pred_test = pred_test.cpu().detach().numpy()
            
            if (count == 1):
                total_pred_test = pred_test
            elif (count > 1):
                total_pred_test = np.concatenate((total_pred_test,pred_test),axis=0)

        test_loss /= len(dataloaders['test_set'].dataset)
        print('Test set: Average loss: {:.4f}'.format(test_loss))
    
    print("Converting to numpy has the shape: ", total_pred_test.shape)
    np.save('/home/cmx/deep_conv_prec/{}/{}/total_pred_test_{}.npy'.format(save_dir, location, epoch), total_pred_test)
    return total_pred_test

## save average fields:
def save_average_field_subplot(data_array, title):
    if (title == 'label'):
        avg = np.mean(data_array[:,:,:], axis = 0)
        np.save('/home/cmx/deep_conv_prec/{}/{}/avg_field_label.npy'.format(save_dir, location), avg)
    else:
        for i in range(0,31):
            avg = np.mean(data_array[:,i,:,:], axis = 0)
            np.save('/home/cmx/deep_conv_prec/{}/{}/avg_field_{}.npy'.format(save_dir, location, variable_list[i]), avg.compressed())

## test with sklearn classification report.
def class_metric(model, dataloaders, epoch):
    print(">>>>>>>>>> test with sklearn classification report. <<<<<<<<<<")
    total_pred_test = test(model, dataloaders, epoch)
    file = '/home/cmx/deep_conv_prec/{}/{}/classification_report_{}.txt'.format(save_dir, location, str(epoch))
    fc = open(file, 'a')
    for threshold in np.arange(0.05, 1, 0.05):
        total_pred_class = copy.deepcopy(total_pred_test)
        print(threshold)
        print("********************************", file = fc)
        print(threshold, file = fc)
        np.place(total_pred_class, total_pred_class < threshold, 0)
        np.place(total_pred_class, total_pred_class >= threshold, 1)

        sk_Y_test = Y_test.reshape(-1)
        sk_pred_class = total_pred_class.reshape(-1)
        target_names = ['class 0', 'class 1']
        print(classification_report(sk_Y_test, sk_pred_class, target_names = target_names), file = fc)
    fc.close()



location = sys.argv[1] # A, B, C, D
status = sys.argv[2] # train, fine_tune
cuda_id = sys.argv[3]
fine_tune_epoch = sys.argv[4]
end_epoch = int(sys.argv[5]) # for fine_tune
save_dir = sys.argv[6]

variable_list = ['cape', 'cin', 'tciw', 'tclw', 'tcwv', 'blh', 'st', 'surlh', 'sursh', 'ssrd', 'rh_300', 'rh_500', 'rh_700', 'sh_300', 'sh_500', 'sh_700', 't_300', 't_500', 't_700', 'u_300', 'u_500', 'u_700', 'u_925', 'v_300', 'v_500', 'v_700', 'v_925', 'ver_500', 'ver_700', 'geoheight']
lon_st, lon_ed, xticklabel, batch_size = location_range(location)
loc = [30, 0, lon_st, lon_ed]
data = load_variables(loc, location, lon_st, lon_ed)
label = load_label(lon_st, lon_ed)
X_train, Y_train, X_val, Y_val, X_test, Y_test, channel_mean, channel_std = customed_data(data, label)
dataloaders = dataloader(X_train, Y_train, X_val, Y_val, X_test, Y_test, channel_mean, channel_std, batch_size)


device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
model = Unet_Res_up(in_ch = 33, out_ch = 1)
#model = Unet(in_ch = 33, out_ch = 1)
model = model.to(device)
print('device name, (device): ', torch.cuda.get_device_name(0), device)

if (status == 'train'):
    print(">>>>>>>>>> Save average field. <<<<<<<<<<")
    #save_average_field_subplot(label, 'label')
    #save_average_field_subplot(data, 'data')

    print(">>>>>>>>>> Train. <<<<<<<<<<")
    # lr = 0.01 before. train with epoch (1,401)
    optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum = 0.8)
    print(optimizer)
    total_validation_loss = []
    best_model = None
    best_loss = 10000
    best_epoch = 0

    for epoch in range(1,200):
        model = train(model, epoch, optimizer, dataloaders)
        validation_loss, total_validation_loss = validation(model, total_validation_loss, dataloaders)

        if validation_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = validation_loss
            best_epoch = epoch
        
        if (epoch % 10 == 0):
            checkpoint_path = "/home/cmx/deep_conv_prec/{}/{}/{}.pth".format(save_dir, location, str(epoch))
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, checkpoint_path)
            np.save('/home/cmx/deep_conv_prec/{}/{}/val_loss_{}.npy'.format(save_dir, location, str(epoch)), np.array(total_validation_loss))
            total_pred_test = test(model, dataloaders, epoch)
        

    print("Best epoch is Epoch {}".format(best_epoch))
    checkpoint_path = "/home/cmx/deep_conv_prec/{}/{}/best.pth".format(save_dir, location)
    state = {'model':best_model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':best_epoch}
    torch.save(state, checkpoint_path)
    
    #assert best_model != model, 'best_model == last model'
    class_metric(best_model, dataloaders, best_epoch)

elif (status == 'fine_tune'):
    print(">>>>>>>>>> Fine tune. <<<<<<<<<<")
    checkpoint_path = "/home/cmx/deep_conv_prec/{}/{}/{}.pth".format(save_dir, location, str(fine_tune_epoch))
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.8)

    optimizer.load_state_dict(checkpoint['optimizer'])  # if want to use train's optimizer, not reduce the lr
    print(optimizer)
    start_epoch = checkpoint['epoch']
    total_validation_loss = np.load('/home/cmx/deep_conv_prec/{}/{}/val_loss_{}.npy'.format(save_dir, location, str(fine_tune_epoch)))
    total_validation_loss =  total_validation_loss.tolist()
    print('Successfully Load Epoch {} '.format(start_epoch))
    print('Length of total validation loss: ', len(total_validation_loss))
    
    for epoch in range(start_epoch + 1, end_epoch+1):
        train(model, epoch, optimizer, dataloaders)
        total_validation_loss = validation(model, total_validation_loss, dataloaders)
        if (epoch == end_epoch):
            checkpoint_path = "/home/cmx/deep_conv_prec/{}/{}/{}.pth".format(save_dir, location, str(epoch))
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, checkpoint_path)
            np.save('/home/cmx/deep_conv_prec/{}/{}/val_loss_{}.npy'.format(save_dir, location, str(epoch)), np.array(total_validation_loss))
            class_metric(model, dataloaders, epoch)
