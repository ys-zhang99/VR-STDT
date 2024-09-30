from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import Tensor
import numpy as np
import os
from src.utils.scaler import StandardScaler,MinMaxScaler

def get_dataloader(datapath, batch_size, input_dim, mode='train'):
    '''
    get data loader from preprocessed data
    '''
    data = {}
    processed = {}
    results = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        if category == 'train':
            data['y1_' + category] = cat_data['y_']

    scalers = []
    for i in range(input_dim):
        #scalers.append(StandardScaler(mean=data['x_train'][..., i].mean(),std=data['x_train'][..., i].std()))
        scalers.append(MinMaxScaler(min=data['x_train'][..., i].min(),
                                    max=data['x_train'][..., i].max()))
    scalers2 = []
    scalers2.append(scalers[3])

    # Data format
    for category in ['train', 'val', 'test']:
        # normalize the target series (generally, one kind of series)
        for i in range(input_dim):
            data['x_' + category][..., i] = scalers[i].transform(data['x_' + category][..., i])
        
        data['y_' + category][..., 0] = scalers2[0].transform(data['y_' + category][..., 0])
        if category == 'train':
            data['y1_' + category][...,0] = scalers2[0].transform(data['y1_' + category][..., 0])
            new_x = Tensor(data['x_' + category])
            new_y = Tensor(data['y_' + category])
            new_y1 = Tensor(data['y1_' + category])
            processed[category] = TensorDataset(new_x,new_y,new_y1)
        else:  
            new_x = Tensor(data['x_' + category])
            new_y = Tensor(data['y_' + category])
            processed[category] = TensorDataset(new_x, new_y)

    results['train_loader'] = DataLoader(processed['train'], batch_size, shuffle=True)
    results['val_loader'] = DataLoader(processed['val'], batch_size, shuffle=False)
    results['test_loader'] = DataLoader(processed['test'], batch_size, shuffle=False)

    print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
                                                   len(results['val_loader'].dataset),
                                                   len(results['test_loader'].dataset)))
    results['scalers'] = scalers2
    return results

def check_device(device=None):
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)

def get_num_nodes(dataset):
    d = {'ft_new': 1,'ft_new_PM10':1, 'ft_new_CO2':1,'ft_new_TVOC':1,'ft_new_PM1.0':1,'ft_new_nohis':1,'ft_new_nohis_120':1,'ft_new_notime':1,'ft_new_nopatch':1,'ft_new_nodouble':1,'ft_new_simplified':1,
         'ft_new_0.1a':1,'ft_new_0.2a':1,'ft_new_0.3a':1,'ft_new_0a':1,'ft_new_0.05a':1,'ft_new_72':1,'ft_new_24':1,'ft_new_nopatch72':1,'ft_new_nopatch24':1,'ft_new_nochannel':1,'ft_new_airformer':1
         ,'ft_data':9,'ft_data_airformer':9,'air_data':8,'air_data_airformer':8,'ft_new_last':1,'ft_new_last0a':1,'ft_new_last0.1a':1,'ft_new_last0.3a':1,'ft_new_last_nopatch120':1,'ft_new_last_nopatch72':1
         ,'ft_new_last_24':1,'ft_new_last_72':1,'ft_new_last_nodouble':1,'ft_new_last_noloss':1,'ft_new_last48':1}
    assert dataset in d.keys()
    return d[dataset]
