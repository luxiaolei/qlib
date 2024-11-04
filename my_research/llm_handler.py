from typing import List, Text, Union
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.data.filter import NameDFilter
from qlib.contrib.data.handler import check_transform_proc


class LLMDataHandler(DataHandlerLP):
    """Data handler for LLM-based forecasting."""
    
    def __init__(
        self,
        instruments: Union[List, Text] = "sp500",
        start_time: str = "2019-01-01",
        end_time: str = "2020-01-01",
        freq: str = "day",
        infer_processors: List[Union[dict, Processor]] = [],
        learn_processors: List[Union[dict, Processor]] = [],
        fit_start_time: str = "2019-01-01",
        fit_end_time: str = "2019-05-31",
        stock_id: str = "NVDA",
        **kwargs
    ):
        """Initialize LLM data handler."""
        # Create name filter for single stock
        name_filter = NameDFilter(name_rule_re=f"^{stock_id}$")
        filter_pipe = [name_filter]
        
        # Initialize data loader with config
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
            },
        }
        
        # Process time parameters for processors
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **kwargs
        )

    def get_feature_config(self):
        """Get feature configuration with basic price and volume data."""
        fields = [
            "$close",
            "$volume",
        ]
        
        names = [
            "CLOSE",
            "VOLUME",
        ]
        
        return (fields, names)

    def get_label_config(self):
        """Get label configuration."""
        return (["Sign(Ref($close, -2) / Ref($close, -1) - 1)"], ["LABEL"])