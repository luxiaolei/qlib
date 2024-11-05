import unittest

import numpy as np

from qlib.tests import TestAutoData


class TestNN(TestAutoData):
    def test_both_dataset(self):
        try:
            from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
            from qlib.data.dataset import DatasetH, TSDatasetH
            from qlib.data.dataset.handler import DataHandlerLP
        except ImportError:
            print("Import error.")
            return

        data_handler_config = {
            "start_time": "2008-01-01",
            "end_time": "2020-08-01",
            "instruments": "csi300",
            "data_loader": {
                "class": "QlibDataLoader",  # Assuming QlibDataLoader is a string reference to the class
                "kwargs": {
                    "config": {
                        "feature": [["$high", "$close", "$low"], ["H", "C", "L"]],
                        "label": [["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]],
                    },
                    "freq": "day",
                },
            },
            # TODO: processors
            "learn_processors": [
                {
                    "class": "DropnaLabel",
                },
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
        }
        segments = {
            "train": ["2008-01-01", "2014-12-31"],
            "valid": ["2015-01-01", "2016-12-31"],
            "test": ["2017-01-01", "2020-08-01"],
        }
        data_handler = DataHandlerLP(**data_handler_config)

        # time-series dataset
        tsds = TSDatasetH(handler=data_handler, segments=segments)

        # tabular dataset
        tbds = DatasetH(handler=data_handler, segments=segments)


        model_l = [
            GeneralPTNN(
                n_epochs=2,
                pt_model_uri="qlib.contrib.model.pytorch_gru_ts.GRUModel",
                pt_model_kwargs={
                    "d_feat": 3,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "dropout": 0.0,
                },
                n_jobs=0,
            ),
            GeneralPTNN(
                n_epochs=2,
                pt_model_uri="qlib.contrib.model.pytorch_nn.Net",  # it is a MLP
                pt_model_kwargs={
                    "input_dim": 3,
                },
                n_jobs=0,
            ),
        ]
        
        model = model_l[0]
        
        dataset = tsds
        ists = isinstance(dataset, TSDatasetH)  # is this time series dataset

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        
        wl_train = np.ones(len(dl_train))
        wl_valid = np.ones(len(dl_valid))

        # Preprocess for data.  To align to Dataset Interface for DataLoader
        if ists:
            dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
            dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        else:
            # If it is a tabular, we convert the dataframe to numpy to be indexable by DataLoader
            dl_train = dl_train.values
            dl_valid = dl_valid.values
        
        # If I comment out the following code, the model.fit fails with msg:
        
        ''''
train_loader debug:
/Users/xlmbp/Library/Caches/pypoetry/virtualenvs/pyqlib-US6TLr8Z-py3.9/lib/python3.9/site-packages/joblib/externals/loky/backend/resource_tracker.py:314: UserWarning: resource_tracker: There appear to be 2 leaked folder objects to clean up at shutdown
  warnings.warn(
/Users/xlmbp/Library/Caches/pypoetry/virtualenvs/pyqlib-US6TLr8Z-py3.9/lib/python3.9/site-packages/joblib/externals/loky/backend/resource_tracker.py:330: UserWarning: resource_tracker: /var/folders/t_/_y2zd7sj4jn40698ccyvkc1c0000gn/T/joblib_memmapping_folder_85449_b3f2b1f19c014b368be5231b59854004_479e73645d3845a9bc878226649c78e7: FileNotFoundError(2, 'No such file or directory')
  warnings.warn(f"resource_tracker: {name}: {e!r}")
/Users/xlmbp/Library/Caches/pypoetry/virtualenvs/pyqlib-US6TLr8Z-py3.9/lib/python3.9/site-packages/joblib/externals/loky/backend/resource_tracker.py:330: UserWarning: resource_tracker: /var/folders/t_/_y2zd7sj4jn40698ccyvkc1c0000gn/T/joblib_memmapping_folder_85449_27ee680f33754deea015a06745947ff9_5cd1524d96864a72975f401ffe8e6203: FileNotFoundError(2, 'No such file or directory')
  warnings.warn(f"resource_tracker: {name}: {e!r}")
pyqlib-py3.9xlmbp@xl qlib % 
        '''
        '''
        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=2000,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        '''

        model.fit(tsds)

import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    unittest.main()
    