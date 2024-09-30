import logging
import os
import time
from typing import Optional, List, Union
import matplotlib as mpl
import matplotlib

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam

from src.utils.logging import get_logger
from src.utils import metrics as mc
from src.utils.metrics import masked_mae,masked_mse
from src.base.sampler import RandomSampler
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import pandas as pd
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
plt.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['Times New Roman']   #设置简黑字体
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
#matplotlib.use('Agg')

class BaseTrainer():
    def __init__(
            self,
            model: nn.Module,
            adj_mat,
            filter_type: str,
            data,
            aug: float,
            base_lr: float,
            steps,
            lr_decay_ratio,
            log_dir: str,
            n_exp: int,
            save_iter: int = 300,
            clip_grad_value: Optional[float] = None,
            max_epochs: Optional[int] = 1000,
            patience: Optional[int] = 1000,
            device: Optional[Union[torch.device, str]] = None,
    
    ):
        super().__init__()

        self._logger = get_logger(
            log_dir, __name__, 'info_{}.log'.format(n_exp), level=logging.INFO)
        if device is None:
            print("`device` is missing, try to train and evaluate the model on default device.")
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)

        self._model = model
        self.model.to(self._device)
        self._logger.info("the number of parameters: {}".format(self.model.param_num(self.model.name))) 

        self._adj_mat = adj_mat
        self._filter_type = filter_type
        self._aug = aug
        self._loss_fn = masked_mae
        self._mse = masked_mse
        self._base_lr = base_lr
        self._optimizer = torch.optim.AdamW(self.model.parameters(), base_lr)
        self._lr_decay_ratio = lr_decay_ratio
        self._steps = steps
        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(self.optimizer,
                                             steps,
                                             gamma=lr_decay_ratio)
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        self._n_exp = n_exp
        self._data = data
        self._supports = None
        
        if aug > 0:
            self._sampler = RandomSampler(adj_mat, filter_type)
        
        self._supports = self._calculate_supports(adj_mat, filter_type)
        assert(self._supports is not None)

    @property
    def model(self):
        return self._model

    @property
    def supports(self):
        return self._supports

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn
    @property
    def mse(self):
        return self._mse
    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _inverse_transform(self, tensors: Union[Tensor, List[Tensor]]):
        n_output_dim = 1
        def inv(tensor, scalers):
            for i in range(n_output_dim):
                tensor[..., i] = scalers[i].inverse_transform(tensor[..., i])
            return tensor

        if isinstance(tensors, list):
            return [inv(tensor, self.data['scalers']) for tensor in tensors]
        else:
            return inv(tensors, self.data['scalers'])

    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_{}.pt'.format(n_exp)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return True

    def load_model(self, epoch, save_path, n_exp):
        filename = 'final_model_{}.pt'.format(n_exp)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
        return True

    def early_stop(self, epoch, best_loss):
        self.logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch, best_loss))
        np.savetxt(os.path.join(self.save_path, 'val_loss_{}.txt'.format(self._n_exp)), [best_loss], fmt='%.4f', delimiter=',')

    def _calculate_supports(self, adj_mat, filter_type):
        return None

    def train_batch(self, X, label, y1, iter):
        '''
        the training process of a batch
        '''        
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports
        self.optimizer.zero_grad()
        pred,y_ = self.model(X, supports)
        pred, label,y_,y1 = self._inverse_transform([pred, label,y_,y1])

        loss = self.loss_fn(pred, label, 0.0) + self.loss_fn(y_, y1, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()
    """ def train_batch(self, X, label, iter):
        '''
        the training process of a batch
        '''        
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports
        self.optimizer.zero_grad()
        pred= self.model(X, supports)
        pred, label= self._inverse_transform([pred, label])

        loss = self.loss_fn(pred, label, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item() """

    def train(self):
        '''
        the training process
        '''       
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, (X, label,y1) in enumerate(self.data['train_loader']):
                X, label,y1 = self._check_device([X, label,y1])
                train_losses.append(self.train_batch(X, label, y1,  iter))
                """ iter += 1
                if iter != None:
                    if iter % self._save_iter == 0: # iteration needs to be checked
                        val_loss = self.evaluate()
                        message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} '.format(epoch,
                                    self._max_epochs,
                                    iter,
                                    np.mean(train_losses),
                                    val_loss)
                        self.logger.info(message)

                        if val_loss < np.min(val_losses): 
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp)
                            self._logger.info(
                                'Val loss decrease from {:.4f} to {:.4f}, '
                                'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                            val_losses.append(val_loss)
                            saved_epoch = epoch """
            val_loss = self.evaluate()
            message = 'Epoch [{}/{}]  train_mae: {:.4f}, val_mae: {:.4f} '.format(epoch,
                        self._max_epochs,
                        np.mean(train_losses),
                        val_loss)
            self.logger.info(message)

            if val_loss < np.min(val_losses): 
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch             
            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch,
                                 self._max_epochs,
                                 iter,
                                 np.mean(train_losses),
                                 val_loss,
                                 new_lr,
                                 (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses): # error saving criterion
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

    def evaluate(self):
        '''
        model evaluation
        '''
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (X, label) in enumerate(self.data['val_loader']):
                X, label = self._check_device([X, label])
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        mae = self.loss_fn(preds, labels, 0.0).item()
        mse = 1
        return mae

    def test_batch(self, X, label):
        '''
        the test process of a batch
        '''
        pred,_ = self.model(X, self.supports)
        pred, label = self._inverse_transform([pred, label])
        return pred, label

    def test(self, epoch, mode='test'):
        '''
        test process
        '''
        self.load_model(epoch, self.save_path, self._n_exp)

        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (X, label) in enumerate(self.data[mode + '_loader']):
                X, label = self._check_device([X, label])
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        print(labels.shape,preds.shape)
        show_label = labels[:,23,0,0].reshape(labels.shape[0]) #[B,T,N,C]
        show_pred = preds[:,23,0,0].reshape(preds.shape[0])
        """ test=pd.DataFrame(data=show_pred)#将数据放进表格
        test.to_csv('./data/ft_data_origin/33.csv') #数据存入csv,存储位置及文件名称 """
        x_axis = range(0,len(show_label))
        error_ = np.abs((labels.flatten()-preds.flatten())/(labels.flatten()))
        error2_ = np.abs((labels.flatten()-preds.flatten())/(labels.flatten().mean()))

        error = np.abs((show_label-show_pred)/(show_label))
        error2 = np.abs((show_label-show_pred)/(show_label.mean()))
        x_error = range(0,len(error))

        colors = [(34 / 255, 82 / 255, 127 / 255), (240 / 255, 70 / 255, 67 / 255)]
        fig=plt.figure(1,figsize=(10,6),dpi=500)
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.5])
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x_axis,show_label,"-",color=colors[0],label="Truth", linewidth=1)
        ax1.plot(x_axis,show_pred,'-',color=colors[1],label="Prediction",linewidth=1)
        ax1.set_xlabel("Time", fontdict={'family': 'Times New Roman', 'size': 15})
        ax1.set_ylabel("Value", fontdict={'family': 'Times New Roman', 'size': 15})
        # 设置x轴刻度标签
        plt.xticks(fontproperties='Times New Roman', size=15)
        plt.yticks(fontproperties='Times New Roman', size=15)
        #ax1.set_xlim(-10, 1526)
        # plt.ylim(0, 0.66)
        ax1.legend(loc="upper right", fontsize='x-large', prop={'family': 'Times New Roman', 'size': 15},
                    frameon=False, markerscale=1, handlelength=1)
        # 添加第二个子图，占20%
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(x_error, error, color=colors[0], linewidth=1,label='Relative error')
        ax2.set_ylabel('Relative error', fontdict={'family': 'Times New Roman', 'size': 15})
        ax2.set_xlabel("Time", fontdict={'family': 'Times New Roman', 'size': 15})
        plt.yticks(fontproperties='Times New Roman', size=15)  # [2,5,10,15,20,25,],
        plt.xticks(fontproperties='Times New Roman', size=15)
        ax2.set_ylim(0,1)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        fig.align_ylabels()
        plt.legend()

        plt.tight_layout()


        """ fig = plt.figure(figsize=(12,5))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams.update({"font.size":15})
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_xlabel(...,fontsize=15)
        ax1.set_ylabel(...,fontsize=15)
        ax1.set_title(...,fontsize=15)
        plt.title('Virtual sensor PM2.5 prediction')
        plt.xlabel('Time/s')
        plt.ylabel('PM2.5')
        plt.plot(x_axis,show_label,label='real')
        plt.plot(x_axis,show_pred,label='pred')
        plt.legend(['Real','Predict']) """
        """ x_data1 = []
        x_data2=[]
        y_data =[]
        y_data2=[]

        for i in range(30):
            x_data1.append(x_axis[i])
            y_data.append(show_pred[i])
            plt.cla()
            plt.ylim((0,300))
            plt.title('Virtual sensor PM2.5 prediction')
            plt.xlabel('Time/s')
            plt.ylabel('PM2.5')
            plt.plot(x_data2,y_data2,label='real')
            plt.plot(x_data1,y_data,label='pred',)
            plt.legend(['Real','Predict'])
            plt.pause(0.1)

        for i in range(0,len(x_axis)-30):
            x_data1.append(x_axis[i+30])
            x_data2.append(x_axis[i])
            y_data.append(show_pred[i+30])
            y_data2.append(show_label[i])

            #fig = plt.figure(figsize=(12,5))
            plt.cla()
            plt.ylim((0,300))
            plt.title('Virtual sensor PM2.5 prediction')
            plt.xlabel('Time/s')
            plt.ylabel('PM2.5')
            plt.plot(x_data2,y_data2,label='real')
            plt.plot(x_data1,y_data,label='pred',)
            plt.legend(['Real','Predict'])
            plt.pause(0.1) """
        #plt.draw()
        """ ax2 = fig.add_subplot(1,2,2)
        error = np.abs((show_label-show_pred)/(show_label))
        error2 = np.abs((show_label-show_pred)/(show_label.mean()))
        #error=np.delete(error, np.where(error > 0.5)[0], axis=0)
        x_error = range(0,len(error))
        ax2.set_xlabel(...,fontsize=15)
        ax2.set_ylabel(...,fontsize=15)
        ax2.set_title(...,fontsize=15)
        plt.title('prediction relative error')
        plt.ylim(0,1)
        plt.plot(x_error,error)
        plt.legend(['Relative error'])
        #plt.plot(x,acc_list,label='误差')
        plt.xlabel('Time/s')
        plt.ylabel('Relative error') """
        plt.show()
        corr,_= pearsonr(show_label, show_pred)
        #print("error:",error.mean(),error.max())
        print("corr:",corr)
        mse = torch.mean((show_pred-show_label)**2)
        mae = torch.mean(torch.absolute(show_pred-show_label))
        mse_ = torch.mean((preds.flatten()-labels.flatten())**2)
        mae_ = torch.mean(torch.absolute(preds.flatten()-labels.flatten()))
        print("mae:{} mse:{}".format(mae,mse))
        print("ARD:{} MAPE:{}".format(error2.mean(),error.mean()))

        if self.model.horizon == 8: 
            amae_day = []
            armse_day = []

            for i in range(0, self.model.horizon, 8):
                pred = preds[:, i: i + 8]
                real = labels[:, i: i + 8]
                metrics = mc.compute_all_metrics(pred, real, 0.0)
                amae_day.append(metrics[0])
                armse_day.append(metrics[1])

            log = '0-7 (1-24h) Test MAE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(amae_day[0], armse_day[0]))

            results = pd.DataFrame(columns=['Time','Test MAE', 'Test RMSE'], index=range(4))
            Time_list=['1-24h','25-48h','49-72h', 'SuddenChange']
            for i in range(3):
                results.iloc[i, 0]= Time_list[i]
                results.iloc[i, 1]= amae_day[i]
                results.iloc[i, 2]= armse_day[i]
            
        else:
            print('The output length is not 24!!!')

        mask_sudden_change = mc.sudden_changes_mask(labels, datapath = './data/ft_data', null_val = 0.0, threshold_start = 75, threshold_change = 20)
        results.iloc[3, 0] = Time_list[3]
        sc_mae, sc_rmse = mc.compute_sudden_change(mask_sudden_change, preds, labels, null_value = 0.0)
        results.iloc[3, 1:] = [sc_mae, sc_rmse]
        log = 'Sudden Changes MAE: {:.4f},  RMSE: {:.4f}'
        print(log.format(sc_mae, sc_rmse))
        results.to_csv(os.path.join(self.save_path, 'metrics_{}.csv'.format(self._n_exp)), index = False)


    def save_preds(self, epoch):
        '''
        save prediction results
        '''
        self.load_model(epoch, self.save_path, self._n_exp)

        for mode in ['train', 'val', 'test']:
            labels = []
            preds = []
            inputs = []
            with torch.no_grad():
                self.model.eval()
                for _, (X, label) in enumerate(self.data[mode + '_loader']):
                    X, label = self._check_device([X, label])
                    pred, label = self.test_batch(X, label)
                    labels.append(label.cpu())
                    preds.append(pred.cpu())
                    inputs.append(X.cpu())
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            inputs = torch.cat(inputs, dim=0)

            np.save(os.path.join(self.save_path, mode + '_preds.npy'), preds)
            np.save(os.path.join(self.save_path, mode + '_labels.npy'), labels)
