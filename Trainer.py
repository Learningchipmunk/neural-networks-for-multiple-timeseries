import os
import glob
import gc
import torch
import copy
import numpy as np
import pandas as pd

from tqdm import tqdm

def pearsonr(x, y):
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))

    ## If r_den is 0, then r_num is also 0, to avoid 0/0, we add 1 to r_den
    r_den = torch.where(r_den == 0, torch.tensor(1.0, device=r_den.device), r_den)

    r = r_num / r_den
    return r#.abs() We actually want the pearson correlation to be the closet to 1: so we minimize -r


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, coef_mse=1.0, coef_corr=1.0, corr_on_main_targ=False, save_best_model=False, save_path=None, load_model_from_path=False):
        self.model      = model
        self.best_model = None #Stores best model on validation set
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.coef_mse  = coef_mse
        self.coef_corr = coef_corr
        self.corr_on_main_targ = corr_on_main_targ
        self.save_path = save_path
        self.save_best_model = save_best_model
        self.load_model_from_path = load_model_from_path
        self.best_val_loss = float('inf')# Init best val loss to inf

        ## Creates the save path if needed
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        ## Loads the model for training if needed
        if(self.load_model_from_path):
            self.model = self.load_model(model)

    def load_model(self, model):
        if self.save_path is None:
            raise ValueError("save_path must be provided when loading a model from path.")
        if not os.path.exists(self.save_path):
            raise ValueError(f"Path {self.save_path} does not exist.")
        if os.path.isdir(self.save_path):
            files = [f for f in os.listdir(self.save_path) if f.endswith('.pt')]
            if len(files) == 0:
                raise ValueError(f"No .pt files found in {self.save_path}.")
            elif len(files) == 1:
                model_path = os.path.join(self.save_path, files[0])
                model.load_state_dict(torch.load(model_path))
            else:
                print(f"Multiple .pt files found in {self.save_path}. Please choose one:")
                for i, f in enumerate(files):
                    print(f"{i+1}: {f}")
                choice = int(input())
                while choice < 1 or choice > len(files):
                    print(f"Invalid choice. Please choose a number between 1 and {len(files)}:")
                    choice = int(input())
                model_path = os.path.join(self.save_path, files[choice-1])
                model.load_state_dict(torch.load(model_path))
            
            metrics = glob.glob(os.path.join(self.save_path, '[m,M]etric*[t,T]rain*.csv'))
            if len(metrics) == 0:
                print(f"No training metrics files found in {self.save_path}. Best `val_loss` will be inf.")
            elif len(metrics) == 1:
                csv_path = metrics[0]
                csv = pd.read_csv(csv_path)
                val_loss_col = [col for col in csv.columns if 'val' in col.lower() and 'loss' in col.lower()][0]
                self.best_val_loss = csv[val_loss_col].min()
                print(f"Metric file found, best `{val_loss_col}` is {self.best_val_loss:.4f}.")
            else:
                print(f"Multiple metric files found in {self.save_path}. Please choose one:")
                for i, f in enumerate(metrics):
                    print(f"{i+1}: {f}")
                choice = int(input())
                while choice < 1 or choice > len(metrics):
                    print(f"Invalid choice. Please choose a number between 1 and {len(metrics)}:")
                    choice = int(input())
                csv_path = metrics[choice-1]
                csv = pd.read_csv(csv_path)
                val_loss_col = [col for col in csv.columns if 'val' in col.lower() and 'loss' in col.lower()][0]
                self.best_val_loss = csv[val_loss_col].min()
                print(f"Metric file found, best `{val_loss_col}` is {self.best_val_loss:.4f}.")
            print(f"Loaded model from {model_path}")

        return model.to(self.device)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def compute_loss(self, outputs, padded_labels, masks_inputs):
        # MSE only on non-padded labels
        mse = self.criterion(outputs * masks_inputs, padded_labels * masks_inputs)

        ## Corr with all targets; adjust as needed
        # corr = pearsonr(outputs[:,:,0].view(-1)[masks_inputs.nonzero()], padded_labels[:,:,0].view(-1)[masks_inputs.nonzero()])#If you want only to evaluate the primary target
        # corr = pearsonr(outputs.view(-1)[masks_inputs.nonzero()], padded_labels.view(-1)[masks_inputs.nonzero()])
        if self.corr_on_main_targ:#If you want only to compute and maximize the loss on the primary target
            corr = pearsonr(outputs[masks_inputs.squeeze(-1)!=0][:, 0], padded_labels[masks_inputs.squeeze(-1)!=0][:, 0])
        else:
            corr = pearsonr(outputs[masks_inputs.squeeze(-1)!=0], padded_labels[masks_inputs.squeeze(-1)!=0])


        ## Compute loss as a weighted sum of MSE and Corr
        loss = mse*self.coef_mse - corr*self.coef_corr

        return loss, mse, corr
    
    def unpack_batch(self, batch):
        padded_inputs = batch[0].to(device=self.device)
        padded_labels = batch[1].to(device=self.device)
        masks_inputs = batch[2].to(device=self.device)
        if(len(batch)>3):
            segment_masks = batch[3].to(device=self.device)
        else:
            segment_masks = None

        return padded_inputs, padded_labels, masks_inputs, segment_masks
    
    # Training loop
    def train_on_batch(self, batch):
        padded_inputs, padded_labels, masks_inputs, segment_masks = self.unpack_batch(batch)

        self.optimizer.zero_grad()

        outputs = self.model(padded_inputs / 4.0, segment_masks)

        # target_weight_softmax = None
        #random_weights = torch.rand(padded_labels.shape[-1], device=device)
        #target_weight_softmax = F.softmax(random_weights)

        loss, _mse, _corr = self.compute_loss(outputs, padded_labels, masks_inputs)
        loss.backward()
        self.optimizer.step()
        return loss.item(), _mse.item(), _corr.item()

    def evaluate_on_batch(self, batch, model=None):
        if model is None:
            model = self.model

        padded_inputs, padded_labels, masks_inputs, segment_masks = self.unpack_batch(batch)

        outputs = model(padded_inputs / 4.0, segment_masks)
        loss, _mse, _corr = self.compute_loss(outputs, padded_labels, masks_inputs)

        return loss.item(), _mse.item(), _corr.item(), outputs[masks_inputs.squeeze(-1)!=0].squeeze().detach().cpu().numpy()
    
    def ModelPredict(self, data_loader, model=None):
        '''If `model` is None, then the best model available is used.
        '''

        # Gives the choice to the user to use a certain model or the best model
        if model is None:
            if self.best_model is None:
                model = self.model
            else:
                model = self.best_model

        # Predicting using model
        model.eval()

        total_loss = []
        total_corr = []
        preds      = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                loss, _mse, _corr, _preds = self.evaluate_on_batch(batch, model)
                total_loss.append(loss)
                total_corr.append(_corr)
                preds.append(_preds)
            loss = np.mean(total_loss)
            corr = np.mean(total_corr)
            print(f"Loss: {loss:.4f} | Corr: {corr:.4f}")
        
        preds = np.concatenate(preds)

        return preds, loss, corr


    def train_model(self, num_epochs, early_stopping_patience=0):
        epochs_without_improvement = 0
        self.model.train()

        # Initialize dictionary to store training and validation metrics
        metrics = {'epoch': [], 'train_loss': [], 'train_corr': [], 'val_loss': [], 'val_corr': []}

        for epoch in range(num_epochs):
            total_loss = []
            total_corr = []
            print(f"\nEPOCH: {epoch+1}/{num_epochs}")
            for batch in tqdm(self.train_loader):
                loss, _mse, _corr = self.train_on_batch(batch)
                total_loss.append(loss)
                total_corr.append(_corr)
            train_loss = np.mean(total_loss)
            train_corr = np.mean(total_corr)
            print(f"Train Loss: {train_loss:.4f} | Train Corr: {train_corr:.4f}")
            metrics['epoch'].append(epoch+1)
            metrics['train_loss'].append(train_loss)
            metrics['train_corr'].append(train_corr)

            self.model.eval()
            with torch.no_grad():

                total_loss = []
                total_corr = []
                for batch in tqdm(self.val_loader):
                    loss, _mse, _corr, _preds = self.evaluate_on_batch(batch)
                    total_loss.append(loss)
                    total_corr.append(_corr)
                val_loss = np.mean(total_loss)
                val_corr = np.mean(total_corr)
                print(f"Val Loss: {val_loss:.4f} | Val Corr: {val_corr:.4f}")
                metrics['val_loss'].append(val_loss)
                metrics['val_corr'].append(val_corr)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model)
                    if self.save_best_model:
                        self.save_model(self.best_model, self.save_path+"best_model.pt")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping after {epoch+1} epochs")
                        break

            # Save metrics as pandas dataframe
            df_metrics = pd.DataFrame(metrics)
            df_metrics.to_csv(self.save_path+"metrics_during_training.csv", index=False)


            torch.cuda.empty_cache()
            _ = gc.collect()

        self.save_model(self.model, self.save_path+"model_last_iter.pt")

        print("\n Training finished!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Model saved at {self.save_path}best_model.pt")

        return self.model

    def get_best_model(self):
        return self.best_model

