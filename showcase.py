import qlib
import pandas as pd
import numpy as np
from qlib.config import C
from qlib.data import D
from qlib.data.dataset import Dataset, Alpha158
from qlib.model.gbdt import LGBModel
from qlib.strategy.strategy import TopkStrategy
from qlib.backtest import backtest, create_report
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import ALPHA360
from qlib.contrib.strategy.rule_strategy import RuleStrategy
from qlib.contrib.model.xgboost import XGBModel
from qlib.workflow import R

# 1. Initialize Qlib with different data providers (example)
# You can switch between different providers by changing the provider_uri
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn") # China market
# qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us") # US market

# 2. Different markets and benchmarks
if C["market"] == "cn":
    benchmark = "SH000300"
    market = "CSI300"
elif C["market"] == "us":
    benchmark = "^GSPC" # S&P 500
    market = "NASDAQ"
else:
    raise ValueError("Market not supported")

# 3. Custom Dataset Handler (inheriting from Alpha158/Alpha360)
class CustomAlpha158(Alpha158):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_feature(self, instrument=None, start_time=None, end_time=None):
        df = super().get_feature(instrument, start_time, end_time)
        # Add custom factors here
        df["custom_factor_1"] = df["open"] / df["close"]
        df["custom_factor_2"] = df["high"] - df["low"]
        return df

class CustomAlpha360(ALPHA360):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_feature(self, instrument=None, start_time=None, end_time=None):
        df = super().get_feature(instrument, start_time, end_time)
        # Add custom factors here
        df["custom_factor_1"] = df["open"] / df["close"]
        df["custom_factor_2"] = df["high"] - df["low"]
        return df


# 4. Symbol Preselection (Example: select top 100 by market cap)
instruments = D.instruments(market=market)
df_inst = D.features(instruments, ["market_cap"], start_time=pd.Timestamp("2020-01-01"), end_time=pd.Timestamp("2020-01-01"))
top_symbols = df_inst.nlargest(100, "market_cap").index.tolist()

# 5. Processors for Learn and Predict (defined in config later)

# 6. Define Label (using config)

# 7. Full Dictionary Style Config
config = {
    "task": {
        "model": {
            "class": "LGBModel", # Available models: LGBModel, XGBModel
            "module_path": "qlib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "verbose": -1,
            },
        },
        "dataset": {
            "class": "CustomAlpha158" if C["market"] == "cn" else "CustomAlpha360", # Use custom dataset
            "module_path": "__main__",
            "kwargs": {
                "features": ["$close", "$volume", "$open", "$high", "$low"], # Example features
                "label": ["Ref($close, -1)/$close - 1"], # Example label: next day return
            },
        },
        "record": ['qlib.workflow.expm.record.Recorder', 'qlib.workflow.expm.record.MetricsRecorder'],
    },
    "market": market,
    "benchmark": benchmark,
    "data_handler_config": {
        "start_time": "2020-01-01",
        "end_time": "2020-12-31",
        "fit_start_time": "2020-01-01",
        "fit_end_time": "2020-09-30",
        "instruments": top_symbols,
    },
    "strategy": {
        "class": "TopkStrategy", # Available strategies: TopkStrategy, RuleStrategy
        "module_path": "qlib.strategy.strategy",
        "kwargs": {"topk": 50},
    },
    "backtest": {
        "start_time": "2020-10-01",
        "end_time": "2020-12-31",
        "account": 1000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_orders": True,
        },
    },
}

# 8. Available Models and Strategies (indicated in comments above)

# 9. Run Experiment and Show Outputs
with R.start(experiment_name="qlib_example"):
    R.log_params(**config)
    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])
    strategy = init_instance_by_config(config["strategy"])

    # train model
    model.fit(dataset)

    # generate prediction
    pred = model.predict(dataset)
    R.save_objects(predictions=pred)

    # backtest and analysis
    portfolio_metircs = backtest(pred, config)
    R.save_objects(portfolio_metircs=portfolio_metircs)
    create_report(portfolio_metircs, config)

    # The outputs are saved in the experiment directory (default: ~/.qlib/experiment)
    # including model checkpoints, predictions, backtest results, and report.
    print(f"Experiment finished. Results are saved in {R.get_exp_dir()}")