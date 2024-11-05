#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
import warnings
from pprint import pprint
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import trange

from qlib.utils.exceptions import LoadObjectError

from ..backtest import backtest as normal_backtest
from ..contrib.eva.alpha import calc_ic, calc_long_short_prec, calc_long_short_return
from ..contrib.evaluate import indicator_analysis, risk_analysis
from ..data.dataset import DatasetH
from ..data.dataset.handler import DataHandlerLP
from ..log import get_module_logger
from ..utils import class_casting, fill_placeholder, flatten_dict, get_date_by_shift
from ..utils.data import deepcopy_basic_type
from ..utils.exceptions import QlibException
from ..utils.time import Freq

logger = get_module_logger("workflow", logging.INFO)


class RecordTemp:
    """
    This is the Records Template class that enables user to generate experiment results such as IC and
    backtest in a certain format.
    """

    artifact_path = None
    depend_cls = None  # the dependant class of the record; the record will depend on the results generated by
    # `depend_cls`

    @classmethod
    def get_path(cls, path=None):
        names = []
        if cls.artifact_path is not None:
            names.append(cls.artifact_path)

        if path is not None:
            names.append(path)

        return "/".join(names)

    def save(self, **kwargs):
        """
        It behaves the same as self.recorder.save_objects.
        But it is an easier interface because users don't have to care about `get_path` and `artifact_path`
        """
        art_path = self.get_path()
        if art_path == "":
            art_path = None
        self.recorder.save_objects(artifact_path=art_path, **kwargs)

    def __init__(self, recorder):
        self._recorder = recorder

    @property
    def recorder(self):
        if self._recorder is None:
            raise ValueError("This RecordTemp did not set recorder yet.")
        return self._recorder

    def generate(self, **kwargs):
        """
        Generate certain records such as IC, backtest etc., and save them.

        Parameters
        ----------
        kwargs

        Return
        ------
        """
        raise NotImplementedError(f"Please implement the `generate` method.")

    def load(self, name: str, parents: bool = True):
        """
        It behaves the same as self.recorder.load_object.
        But it is an easier interface because users don't have to care about `get_path` and `artifact_path`

        Parameters
        ----------
        name : str
            the name for the file to be load.

        parents : bool
            Each recorder has different `artifact_path`.
            So parents recursively find the path in parents
            Sub classes has higher priority

        Return
        ------
        The stored records.
        """
        try:
            return self.recorder.load_object(self.get_path(name))
        except LoadObjectError as e:
            if parents:
                if self.depend_cls is not None:
                    with class_casting(self, self.depend_cls):
                        return self.load(name, parents=True)
            raise e

    def list(self):
        """
        List the supported artifacts.
        Users don't have to consider self.get_path

        Return
        ------
        A list of all the supported artifacts.
        """
        return []

    def check(self, include_self: bool = False, parents: bool = True):
        """
        Check if the records is properly generated and saved.
        It is useful in following examples

        - checking if the dependant files complete before generating new things.
        - checking if the final files is completed

        Parameters
        ----------
        include_self : bool
            is the file generated by self included
        parents : bool
            will we check parents

        Raise
        ------
        FileNotFoundError
            whether the records are stored properly.
        """
        if include_self:
            # Some mlflow backend will not list the directly recursively.
            # So we force to the directly
            artifacts = {}

            def _get_arts(dirn):
                if dirn not in artifacts:
                    artifacts[dirn] = self.recorder.list_artifacts(dirn)
                return artifacts[dirn]

            for item in self.list():
                ps = self.get_path(item).split("/")
                dirn = "/".join(ps[:-1])
                if self.get_path(item) not in _get_arts(dirn):
                    raise FileNotFoundError
        if parents:
            if self.depend_cls is not None:
                with class_casting(self, self.depend_cls):
                    self.check(include_self=True)


class SignalRecord(RecordTemp):
    """
    This is the Signal Record class that generates the signal prediction. This class inherits the ``RecordTemp`` class.
    """

    def __init__(self, model=None, dataset=None, recorder=None):
        super().__init__(recorder=recorder)
        self.model = model
        self.dataset = dataset

    @staticmethod
    def generate_label(dataset):
        with class_casting(dataset, DatasetH):
            params = dict(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            try:
                # Assume the backend handler is DataHandlerLP
                raw_label = dataset.prepare(**params)
            except TypeError:
                # The argument number is not right
                del params["data_key"]
                # The backend handler should be DataHandler
                raw_label = dataset.prepare(**params)
            except AttributeError as e:
                # The data handler is initialized with `drop_raw=True`...
                # So raw_label is not available
                logger.warning(f"Exception: {e}")
                raw_label = None
        return raw_label

    def generate(self, **kwargs):
        # generate prediction
        pred = self.model.predict(self.dataset)
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        self.save(**{"pred.pkl": pred})

        logger.info(
            f"Signal record 'pred.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
        )
        # print out results
        pprint(f"The following are prediction results of the {type(self.model).__name__} model.")
        pprint(pred.head(5))

        if isinstance(self.dataset, DatasetH):
            raw_label = self.generate_label(self.dataset)
            self.save(**{"label.pkl": raw_label})

    def list(self):
        return ["pred.pkl", "label.pkl"]


class ACRecordTemp(RecordTemp):
    """Automatically checking record template"""

    def __init__(self, recorder, skip_existing=False):
        self.skip_existing = skip_existing
        super().__init__(recorder=recorder)

    def generate(self, *args, **kwargs):
        """automatically checking the files and then run the concrete generating task"""
        if self.skip_existing:
            try:
                self.check(include_self=True, parents=False)
            except FileNotFoundError:
                pass  # continue to generating metrics
            else:
                logger.info("The results has previously generated, Generation skipped.")
                return

        try:
            self.check()
        except FileNotFoundError:
            logger.warning("The dependent data does not exists. Generation skipped.")
            return
        artifact_dict = self._generate(*args, **kwargs)
        if isinstance(artifact_dict, dict):
            self.save(**artifact_dict)
        return artifact_dict

    def _generate(self, *args, **kwargs) -> Dict[str, object]:
        """
        Run the concrete generating task, return the dictionary of the generated results.
        The caller method will save the results to the recorder.
        """
        raise NotImplementedError(f"Please implement the `_generate` method")


class HFSignalRecord(SignalRecord):
    """
    This is the Signal Analysis Record class that generates the analysis results such as IC and IR. This class inherits the ``RecordTemp`` class.
    """

    artifact_path = "hg_sig_analysis"
    depend_cls = SignalRecord

    def __init__(self, recorder, **kwargs):
        super().__init__(recorder=recorder)

    def generate(self):
        pred = self.load("pred.pkl")
        raw_label = self.load("label.pkl")
        long_pre, short_pre = calc_long_short_prec(pred.iloc[:, 0], raw_label.iloc[:, 0], is_alpha=True)
        ic, ric = calc_ic(pred.iloc[:, 0], raw_label.iloc[:, 0])
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
            "Long precision": long_pre.mean(),
            "Short precision": short_pre.mean(),
        }
        objects = {"ic.pkl": ic, "ric.pkl": ric}
        objects.update({"long_pre.pkl": long_pre, "short_pre.pkl": short_pre})
        long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], raw_label.iloc[:, 0])
        metrics.update(
            {
                "Long-Short Average Return": long_short_r.mean(),
                "Long-Short Average Sharpe": long_short_r.mean() / long_short_r.std(),
            }
        )
        objects.update(
            {
                "long_short_r.pkl": long_short_r,
                "long_avg_r.pkl": long_avg_r,
            }
        )
        self.recorder.log_metrics(**metrics)
        self.save(**objects)
        pprint(metrics)

    def list(self):
        return ["ic.pkl", "ric.pkl", "long_pre.pkl", "short_pre.pkl", "long_short_r.pkl", "long_avg_r.pkl"]


class SigAnaRecord(ACRecordTemp):
    """
    This is the Signal Analysis Record class that generates the analysis results such as IC and IR.
    This class inherits the ``RecordTemp`` class.
    """

    artifact_path = "sig_analysis"
    depend_cls = SignalRecord

    def __init__(self, recorder, ana_long_short=False, ann_scaler=252, label_col=0, skip_existing=False):
        super().__init__(recorder=recorder, skip_existing=skip_existing)
        self.ana_long_short = ana_long_short
        self.ann_scaler = ann_scaler
        self.label_col = label_col

    def _generate(self, label: Optional[pd.DataFrame] = None, **kwargs):
        pred = self.load("pred.pkl")
        if label is None:
            label = self.load("label.pkl")
        if label is None or not isinstance(label, pd.DataFrame) or label.empty:
            logger.warning(f"Empty label.")
            return
            
        # Calculate IC
        ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, self.label_col])
        
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            "Rank IC": ric.mean(),
            "Rank ICIR": ric.mean() / ric.std(),
        }
        
        objects = {"ic.pkl": ic, "ric.pkl": ric}
        
        if self.ana_long_short:
            long_short_r, long_avg_r = calc_long_short_return(pred.iloc[:, 0], label.iloc[:, self.label_col])
            
            # Create returns plot
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(x=long_short_r.index, y=long_short_r.cumsum(), 
                                           mode='lines', name='Long-Short Cumulative Return'))
            fig_returns.add_trace(go.Scatter(x=long_avg_r.index, y=long_avg_r.cumsum(), 
                                           mode='lines', name='Long-Avg Cumulative Return'))
            fig_returns.update_layout(title='Cumulative Returns', 
                                    xaxis_title='Date',
                                    yaxis_title='Cumulative Return')
                                    
            # Log the plotly figure
            self.recorder.log_plotly_figure(fig_returns, "returns_time_series.html")
            
            metrics.update({
                "Long-Short Ann Return": long_short_r.mean() * self.ann_scaler,
                "Long-Short Ann Sharpe": long_short_r.mean() / long_short_r.std() * self.ann_scaler**0.5,
                "Long-Avg Ann Return": long_avg_r.mean() * self.ann_scaler,
                "Long-Avg Ann Sharpe": long_avg_r.mean() / long_avg_r.std() * self.ann_scaler**0.5,
            })
            objects.update({
                "long_short_r.pkl": long_short_r,
                "long_avg_r.pkl": long_avg_r,
            })
            
        # Log metrics using recorder
        self.recorder.log_metrics(**metrics)
        pprint(metrics)
        return objects

    def list(self):
        paths = ["ic.pkl", "ric.pkl"]
        if self.ana_long_short:
            paths.extend(["long_short_r.pkl", "long_avg_r.pkl"])
        return paths


class PortAnaRecord(ACRecordTemp):
    """
    This is the Portfolio Analysis Record class that generates the analysis results such as those of backtest. This class inherits the ``RecordTemp`` class.

    The following files will be stored in recorder

    - report_normal.pkl & positions_normal.pkl:

        - The return report and detailed positions of the backtest, returned by `qlib/contrib/evaluate.py:backtest`
    - port_analysis.pkl : The risk analysis of your portfolio, returned by `qlib/contrib/evaluate.py:risk_analysis`
    """

    artifact_path = "portfolio_analysis"
    depend_cls = SignalRecord

    def __init__(
        self,
        recorder,
        config=None,
        risk_analysis_freq: Union[List, str] = None,
        indicator_analysis_freq: Union[List, str] = None,
        indicator_analysis_method=None,
        skip_existing=False,
        **kwargs,
    ):
        """
        config["strategy"] : dict
            define the strategy class as well as the kwargs.
        config["executor"] : dict
            define the executor class as well as the kwargs.
        config["backtest"] : dict
            define the backtest kwargs.
        risk_analysis_freq : str|List[str]
            risk analysis freq of report
        indicator_analysis_freq : str|List[str]
            indicator analysis freq of report
        indicator_analysis_method : str, optional, default by None
            the candidate values include 'mean', 'amount_weighted', 'value_weighted'
        """
        super().__init__(recorder=recorder, skip_existing=skip_existing, **kwargs)

        if config is None:
            config = {  # Default config for daily trading
                "strategy": {
                    "class": "TopkDropoutStrategy",
                    "module_path": "qlib.contrib.strategy",
                    "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
                },
                "backtest": {
                    "start_time": None,
                    "end_time": None,
                    "account": 100000000,
                    "benchmark": "SH000300",
                    "exchange_kwargs": {
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                },
            }
        # We only deepcopy_basic_type because
        # - We don't want to affect the config outside.
        # - We don't want to deepcopy complex object to avoid overhead
        config = deepcopy_basic_type(config)

        self.strategy_config = config["strategy"]
        _default_executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        }
        self.executor_config = config.get("executor", _default_executor_config)
        self.backtest_config = config["backtest"]

        self.all_freq = self._get_report_freq(self.executor_config)
        if risk_analysis_freq is None:
            risk_analysis_freq = [self.all_freq[0]]
        if indicator_analysis_freq is None:
            indicator_analysis_freq = [self.all_freq[0]]

        if isinstance(risk_analysis_freq, str):
            risk_analysis_freq = [risk_analysis_freq]
        if isinstance(indicator_analysis_freq, str):
            indicator_analysis_freq = [indicator_analysis_freq]

        self.risk_analysis_freq = [
            "{0}{1}".format(*Freq.parse(_analysis_freq)) for _analysis_freq in risk_analysis_freq
        ]
        self.indicator_analysis_freq = [
            "{0}{1}".format(*Freq.parse(_analysis_freq)) for _analysis_freq in indicator_analysis_freq
        ]
        self.indicator_analysis_method = indicator_analysis_method

    def _get_report_freq(self, executor_config):
        ret_freq = []
        if executor_config["kwargs"].get("generate_portfolio_metrics", False):
            _count, _freq = Freq.parse(executor_config["kwargs"]["time_per_step"])
            ret_freq.append(f"{_count}{_freq}")
        if "inner_executor" in executor_config["kwargs"]:
            ret_freq.extend(self._get_report_freq(executor_config["kwargs"]["inner_executor"]))
        return ret_freq

    def _generate(self, **kwargs):
        pred = self.load("pred.pkl")

        # replace the "<PRED>" with prediction saved before
        placeholder_value = {"<PRED>": pred}
        for k in "executor_config", "strategy_config":
            setattr(self, k, fill_placeholder(getattr(self, k), placeholder_value))

        # if the backtesting time range is not set, it will automatically extract time range from the prediction file
        dt_values = pred.index.get_level_values("datetime")
        if self.backtest_config["start_time"] is None:
            self.backtest_config["start_time"] = dt_values.min()
        if self.backtest_config["end_time"] is None:
            self.backtest_config["end_time"] = get_date_by_shift(dt_values.max(), 1)

        artifact_objects = {}
        # custom strategy and get backtest
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=self.executor_config, strategy=self.strategy_config, **self.backtest_config
        )
        for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
            artifact_objects.update({f"report_normal_{_freq}.pkl": report_normal})
            artifact_objects.update({f"positions_normal_{_freq}.pkl": positions_normal})

        for _freq, indicators_normal in indicator_dict.items():
            artifact_objects.update({f"indicators_normal_{_freq}.pkl": indicators_normal[0]})
            artifact_objects.update({f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]})

        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq not in portfolio_metric_dict:
                warnings.warn(
                    f"the freq {_analysis_freq} report is not found, please set the corresponding env with `generate_portfolio_metrics=True`"
                )
            else:
                report_normal, _ = portfolio_metric_dict.get(_analysis_freq)
                analysis = dict()
                analysis["excess_return_without_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"], freq=_analysis_freq
                )
                analysis["excess_return_with_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_analysis_freq
                )

                # Create performance plot
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(x=report_normal.index, y=(1 + report_normal["return"]).cumprod(), 
                                            mode='lines', name='Strategy'))
                fig_perf.add_trace(go.Scatter(x=report_normal.index, y=(1 + report_normal["bench"]).cumprod(), 
                                            mode='lines', name='Benchmark'))
                fig_perf.update_layout(title=f'Cumulative Performance ({_analysis_freq})', 
                                     xaxis_title='Date',
                                     yaxis_title='Cumulative Return')
                                     
                # Log the plotly figure
                self.recorder.log_plotly_figure(fig_perf, f"performance_{_analysis_freq}.html")

                analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                # log metrics
                analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
                self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
                # save results
                artifact_objects.update({f"port_analysis_{_analysis_freq}.pkl": analysis_df})
                logger.info(
                    f"Portfolio analysis record 'port_analysis_{_analysis_freq}.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                )
                # print out results
                pprint(f"The following are analysis results of benchmark return({_analysis_freq}).")
                pprint(risk_analysis(report_normal["bench"], freq=_analysis_freq))
                pprint(f"The following are analysis results of the excess return without cost({_analysis_freq}).")
                pprint(analysis["excess_return_without_cost"])
                pprint(f"The following are analysis results of the excess return with cost({_analysis_freq}).")
                pprint(analysis["excess_return_with_cost"])

        for _analysis_freq in self.indicator_analysis_freq:
            if _analysis_freq not in indicator_dict:
                warnings.warn(f"the freq {_analysis_freq} indicator is not found")
            else:
                indicators_normal = indicator_dict.get(_analysis_freq)[0]
                if self.indicator_analysis_method is None:
                    analysis_df = indicator_analysis(indicators_normal)
                else:
                    analysis_df = indicator_analysis(indicators_normal, method=self.indicator_analysis_method)
                # log metrics
                analysis_dict = analysis_df["value"].to_dict()
                self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
                # save results
                artifact_objects.update({f"indicator_analysis_{_analysis_freq}.pkl": analysis_df})
                logger.info(
                    f"Indicator analysis record 'indicator_analysis_{_analysis_freq}.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                )
                pprint(f"The following are analysis results of indicators({_analysis_freq}).")
                pprint(analysis_df)
        return artifact_objects

    def list(self):
        list_path = []
        for _freq in self.all_freq:
            list_path.extend(
                [
                    f"report_normal_{_freq}.pkl",
                    f"positions_normal_{_freq}.pkl",
                ]
            )
        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq in self.all_freq:
                list_path.append(f"port_analysis_{_analysis_freq}.pkl")
            else:
                warnings.warn(f"risk_analysis freq {_analysis_freq} is not found")

        for _analysis_freq in self.indicator_analysis_freq:
            if _analysis_freq in self.all_freq:
                list_path.append(f"indicator_analysis_{_analysis_freq}.pkl")
            else:
                warnings.warn(f"indicator_analysis freq {_analysis_freq} is not found")
        return list_path


class MultiPassPortAnaRecord(PortAnaRecord):
    """
    This is the Multiple Pass Portfolio Analysis Record class that run backtest multiple times and generates the analysis results such as those of backtest. This class inherits the ``PortAnaRecord`` class.

    If shuffle_init_score enabled, the prediction score of the first backtest date will be shuffled, so that initial position will be random.
    The shuffle_init_score will only works when the signal is used as <PRED> placeholder. The placeholder will be replaced by pred.pkl saved in recorder.

    Parameters
    ----------
    recorder : Recorder
        The recorder used to save the backtest results.
    pass_num : int
        The number of backtest passes.
    shuffle_init_score : bool
        Whether to shuffle the prediction score of the first backtest date.
    """

    depend_cls = SignalRecord

    def __init__(self, recorder, pass_num=10, shuffle_init_score=True, **kwargs):
        """
        Parameters
        ----------
        recorder : Recorder
            The recorder used to save the backtest results.
        pass_num : int
            The number of backtest passes.
        shuffle_init_score : bool
            Whether to shuffle the prediction score of the first backtest date.
        """
        self.pass_num = pass_num
        self.shuffle_init_score = shuffle_init_score

        super().__init__(recorder, **kwargs)

        # Save original strategy so that pred df can be replaced in next generate
        self.original_strategy = deepcopy_basic_type(self.strategy_config)
        if not isinstance(self.original_strategy, dict):
            raise QlibException("MultiPassPortAnaRecord require the passed in strategy to be a dict")
        if "signal" not in self.original_strategy.get("kwargs", {}):
            raise QlibException("MultiPassPortAnaRecord require the passed in strategy to have signal as a parameter")

    def random_init(self):
        pred_df = self.load("pred.pkl")

        all_pred_dates = pred_df.index.get_level_values("datetime")
        bt_start_date = pd.to_datetime(self.backtest_config.get("start_time"))
        if bt_start_date is None:
            first_bt_pred_date = all_pred_dates.min()
        else:
            first_bt_pred_date = all_pred_dates[all_pred_dates >= bt_start_date].min()

        # Shuffle the first backtest date's pred score
        first_date_score = pred_df.loc[first_bt_pred_date]["score"]
        np.random.shuffle(first_date_score.values)

        # Use shuffled signal as the strategy signal
        self.strategy_config = deepcopy_basic_type(self.original_strategy)
        self.strategy_config["kwargs"]["signal"] = pred_df

    def _generate(self, **kwargs):
        risk_analysis_df_map = {}

        # Collect each frequency's analysis df as df list
        for i in trange(self.pass_num):
            if self.shuffle_init_score:
                self.random_init()

            # Not check for cache file list
            single_run_artifacts = super()._generate(**kwargs)

            for _analysis_freq in self.risk_analysis_freq:
                risk_analysis_df_list = risk_analysis_df_map.get(_analysis_freq, [])
                risk_analysis_df_map[_analysis_freq] = risk_analysis_df_list

                analysis_df = single_run_artifacts[f"port_analysis_{_analysis_freq}.pkl"]
                analysis_df["run_id"] = i
                risk_analysis_df_list.append(analysis_df)

        result_artifacts = {}
        # Concat df list
        for _analysis_freq in self.risk_analysis_freq:
            combined_df = pd.concat(risk_analysis_df_map[_analysis_freq])

            # Calculate return and information ratio's mean, std and mean/std
            multi_pass_port_analysis_df = combined_df.groupby(level=[0, 1]).apply(
                lambda x: pd.Series(
                    {"mean": x["risk"].mean(), "std": x["risk"].std(), "mean_std": x["risk"].mean() / x["risk"].std()}
                )
            )

            # Only look at "annualized_return" and "information_ratio"
            multi_pass_port_analysis_df = multi_pass_port_analysis_df.loc[
                (slice(None), ["annualized_return", "information_ratio"]), :
            ]
            pprint(multi_pass_port_analysis_df)

            # Save new df
            result_artifacts.update({f"multi_pass_port_analysis_{_analysis_freq}.pkl": multi_pass_port_analysis_df})

            # Log metrics
            metrics = flatten_dict(
                {
                    "mean": multi_pass_port_analysis_df["mean"].unstack().T.to_dict(),
                    "std": multi_pass_port_analysis_df["std"].unstack().T.to_dict(),
                    "mean_std": multi_pass_port_analysis_df["mean_std"].unstack().T.to_dict(),
                }
            )
            self.recorder.log_metrics(**metrics)
        return result_artifacts

    def list(self):
        list_path = []
        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq in self.all_freq:
                list_path.append(f"multi_pass_port_analysis_{_analysis_freq}.pkl")
            else:
                warnings.warn(f"risk_analysis freq {_analysis_freq} is not found")
        return list_path
