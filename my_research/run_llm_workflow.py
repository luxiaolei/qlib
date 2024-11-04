import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.utils import init_instance_by_config, flatten_dict
import yaml

def main():
    """Run the LLM-based forecasting workflow."""
    
    # Load config
    with open("my_research/workflow_config_llm.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize Qlib
    qlib.init(**config["qlib_init"])
    
    # Create and run workflow
    with R.start(experiment_name="LLM_Forecasting"):
        # Log parameters
        R.log_params(**flatten_dict(config["task"]))
        
        # Initialize model and dataset
        model = init_instance_by_config(config["task"]["model"])
        dataset = init_instance_by_config(config["task"]["dataset"])
        
        # Run workflow
        #model.fit(dataset)
        
        # Save model
        #R.save_objects(trained_model=model)
        
        # Get recorder
        recorder = R.get_recorder()
        
        # Generate predictions and record signals
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        pred_score = sr.load("pred.pkl")
        
        # Calculate metrics
        sar = SigAnaRecord(recorder)
        sar.generate()
        
        # Run portfolio analysis
        par = PortAnaRecord(recorder, config["port_analysis_config"])
        par.generate()
        
        # Save experiment URI
        uri_path = R.get_uri()
        print(f"Experiment URI: {uri_path}")

if __name__ == "__main__":
    main() 