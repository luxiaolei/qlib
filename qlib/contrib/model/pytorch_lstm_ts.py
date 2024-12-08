# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division, print_function

import atexit
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.weight import Reweighter
from ...log import get_module_logger
from ...model.base import Model
from ...model.utils import ConcatDataset
from ...utils import get_or_create_path


class LSTM(Model):
    """LSTM Model

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs
    ):
        super().__init__()
        
        # Add version-compatible memory management
        self.memory_status = {
            'peak_allocated': 0,
            'current_allocated': 0
        }
        
        def track_memory():
            if torch.cuda.is_available():
                try:
                    # Try newer PyTorch versions
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        self.memory_status['peak_allocated'] = torch.cuda.max_memory_allocated()
                        self.memory_status['current_allocated'] = torch.cuda.memory_allocated()
                except AttributeError:
                    # Fallback for older PyTorch versions
                    self.logger.warning("Advanced CUDA memory tracking not available in this PyTorch version")
                    try:
                        self.memory_status['current_allocated'] = torch.cuda.memory_allocated()
                    except:
                        pass
                
        # Register memory tracking without reset
        atexit.register(track_memory)
        
        # Add safety checks
        if d_feat <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("Invalid model dimensions")

        # Set logger.
        self.logger = get_module_logger("LSTM")
        self.logger.info("LSTM pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        if torch.cuda.is_available() and GPU >= 0:
            self.device = torch.device(f"cuda:{GPU}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        self.logger.info(
            "LSTM parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Ensure model is created on CPU first
        self.LSTM_model = LSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        
        # Then safely move to device
        try:
            self.LSTM_model = self.LSTM_model.to(self.device)
        except RuntimeError as e:
            self.logger.error(f"Failed to move model to device: {e}")
            raise

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.LSTM_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.LSTM_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.LSTM_model.to(self.device)

        # Add gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask], weight=None)

        raise ValueError("unknown metric `%s`" % self.metric)

    @logger.catch
    def train_epoch(self, data_loader):
        self.LSTM_model.train()
        total_loss = 0
        
        for data, weight in data_loader:
            try:
                if data is None or len(data) == 0:
                    continue
                feature = data[:, :, 0:-1].float().to(self.device)
                label = data[:, -1, -1].float().to(self.device)
                
                # Clear gradients
                self.train_optimizer.zero_grad(set_to_none=True)
                
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        pred = self.LSTM_model(feature)
                        loss = self.loss_fn(pred, label, weight.to(self.device))
                    
                    # Use gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.train_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.LSTM_model.parameters(), 3.0)
                    self.scaler.step(self.train_optimizer)
                    self.scaler.update()
                else:
                    pred = self.LSTM_model(feature)
                    loss = self.loss_fn(pred, label, weight.to(self.device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.LSTM_model.parameters(), 3.0)
                    self.train_optimizer.step()
                
                total_loss += loss.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.warning("OOM detected, attempting to recover...")
                    continue
                else:
                    raise e

        return total_loss / len(data_loader)

    def test_epoch(self, data_loader):
        self.LSTM_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            # Convert data and weights to float32 explicitly
            feature = data[:, :, 0:-1].float().to(self.device)
            label = data[:, -1, -1].float().to(self.device)
            weight = weight.float().to(self.device)  # Convert weight to float32

            with torch.no_grad():
                pred = self.LSTM_model(feature)
                loss = self.loss_fn(pred, label, weight)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    @logger.catch
    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        # Protect against data loading issues
        try:
            dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

        # Verify data is not empty
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            # Convert weights to float32
            wl_train = np.ones(len(dl_train), dtype=np.float32)
            wl_valid = np.ones(len(dl_valid), dtype=np.float32)
        elif isinstance(reweighter, Reweighter):
            # Convert reweighter outputs to float32
            wl_train = reweighter.reweight(dl_train).astype(np.float32)
            wl_valid = reweighter.reweight(dl_valid).astype(np.float32)

        else:
            raise ValueError("Unsupported reweighter type.")

        # Use safer DataLoader configuration
        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            drop_last=True,
            generator=torch.Generator(device='cpu')  # Ensure generator is on CPU
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            drop_last=True,
            generator=torch.Generator(device='cpu')  # Ensure generator is on CPU
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # Add cleanup handler
        def cleanup():
            for loader in [train_loader, valid_loader]:
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    loader._iterator._shutdown_workers()

        atexit.register(cleanup)

        # train
        self.logger.info("training...")
        self.fitted = True

        # Add try-except block around the training loop
        try:
            for step in range(self.n_epochs):
                self.logger.info("Epoch%d:", step)
                self.logger.info("training...")
                train_loss = self.train_epoch(train_loader)
                self.logger.info("evaluating...")
                
                with torch.no_grad():
                    train_loss, train_score = self.test_epoch(train_loader)
                    val_loss, val_score = self.test_epoch(valid_loader)
                
                self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
                evals_result["train"].append(train_score)
                evals_result["valid"].append(val_score)

                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.LSTM_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        self.logger.info("early stop")
                        break

            self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
            self.LSTM_model.load_state_dict(best_param)
            torch.save(best_param, save_path)

        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        finally:
            cleanup()
            atexit.unregister(cleanup)
            if self.use_gpu:
                torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        
        # Modify test DataLoader
        test_loader = DataLoader(
            dl_test, 
            batch_size=self.batch_size, 
            num_workers=self.n_jobs,
            persistent_workers=True,
            pin_memory=True,
        )

        # Add cleanup handler
        def cleanup():
            if hasattr(test_loader, '_iterator') and test_loader._iterator is not None:
                test_loader._iterator._shutdown_workers()

        atexit.register(cleanup)

        try:
            self.LSTM_model.eval()
            preds = []

            for data in test_loader:
                feature = data[:, :, 0:-1].to(self.device)
                with torch.no_grad():
                    pred = self.LSTM_model(feature.float()).detach().cpu().numpy()
                preds.append(pred)

            return pd.Series(np.concatenate(preds), index=dl_test.get_index())
        finally:
            cleanup()
            atexit.unregister(cleanup)


class LSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
