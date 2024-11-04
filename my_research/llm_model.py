from typing import Text, Union
import pandas as pd
import numpy as np
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from langserve import RemoteRunnable
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
import redis


class LLMForecaster(Model):
    """LLM-based forecasting model for single stock prediction."""
    
    def __init__(
        self,
        base_url: str = "http://185.150.189.132:8011",
        auth_token: str = "EAFramwork@2024",
        prompt_template: str = "",
        batch_size: int = 10,
        redis_url: str = "redis://localhost:6379"
    ):
        """Initialize LLM forecaster."""
        super().__init__()
        self._params = {
            "base_url": base_url,
            "auth_token": auth_token,
            "prompt_template": prompt_template,
            "batch_size": batch_size
        }
        
        # Initialize Redis client
        redis_client = redis.Redis.from_url(url=redis_url)
        
        # Initialize LLM with Redis cache
        set_llm_cache(RedisCache(redis_=redis_client))
        
        self.proxyllm = RemoteRunnable(
            f"{base_url}/proxyllm/pro/mid_temp", 
            headers={"X-Token": auth_token}
        )
        
    def _determine_global_precision(self, data: np.ndarray, tolerance: float = 0.01) -> int:
        """Determine global precision for an array of data."""
        original_mean = np.mean(data)
        original_std = np.std(data)
        
        for precision in range(10):
            rounded_data = np.round(data, precision)
            mean_diff = abs(np.mean(rounded_data) - original_mean)
            std_diff = abs(np.std(rounded_data) - original_std)
            
            if mean_diff <= tolerance * abs(original_mean) and std_diff <= tolerance * abs(original_std):
                return precision
        
        return 10  # Default to high precision if no suitable precision is found

    def _trim_numeric_data(self, data: np.ndarray) -> np.ndarray:
        """Trim numeric data to a determined global precision."""
        precision = self._determine_global_precision(data)
        return np.round(data, precision)

    def _format_features(self, features_array: np.ndarray, feature_names: list) -> str:
        """Format features into a string for the prompt."""
        # Trim data to global precision
        trimmed_features_array = self._trim_numeric_data(features_array)
        
        # Include processing information
        processing_info = (
            "We are building a financial market price simulator. "
            "Based on processed simulated market dynamics data, "
            "analyze its inherent dynamics and make a prediction.\n"
            "Data is normalized using Z-score normalization.\n"
            "Please output 1 if the price will go up, -1 if the price will go down. Do NOT output anything else."
        )
        
        # Format all timesteps' features
        formatted_features = []
        for timestep in trimmed_features_array:
            formatted_timestep = ", ".join(
                [f"{value:.4g}" for value in timestep]  # Use .4g to remove trailing zeros
            )
            formatted_features.append(formatted_timestep)
        
        # Combine processing info and formatted features
        return processing_info + "Features: " + ", ".join(feature_names) + "\n" + "\n".join(formatted_features)

    def _parse_llm_response(self, response) -> int:
        """Parse LLM response into a numerical prediction."""
        try:
            prediction = float(response.content.strip())
            return 1 if prediction > 0 else -1
        except ValueError:
            return 0

    def fit(self, dataset: DatasetH):
        """No training needed for LLM model."""
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        """Generate predictions for the dataset."""
        if not hasattr(self, "proxyllm"):
            raise ValueError("model is not initialized yet!")
            
        x_test = dataset.prepare(
            segment, 
            col_set="feature",
            data_key=DataHandlerLP.DK_I
        )
        
        if x_test.empty:
            raise ValueError("Empty input data for prediction")
            
        predictions = []
        feature_names = dataset.handler.get_feature_config()[1]  # Get feature names
        
        # Prepare prompts for batch processing
        prompts = []
        for i in range(len(x_test)):
            sample = x_test[i]
            market_data = self._format_features(sample, feature_names)
            prompt = self._params["prompt_template"].format(market_data=market_data)
            prompts.append(prompt)
        
        # Batch process the prompts with caching
        batch_size = self._params["batch_size"]
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            try:
                responses = self.proxyllm.batch(batch_prompts)
                batch_predictions = [self._parse_llm_response(response) for response in responses]
            except Exception as e:
                print(f"Error in batch prediction: {e}")
                batch_predictions = [0] * len(batch_prompts)
                    
            predictions.extend(batch_predictions)
        
        # Get the index from the dataset
        index = x_test.get_index()
        return pd.Series(predictions, index=index)