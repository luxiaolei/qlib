#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import functools
import glob
import inspect
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from operator import xor
from pathlib import Path
from pprint import pprint

import fire
import yaml

import qlib
from qlib.tests.data import GetData
from qlib.workflow import R


# decorator to check the arguments
def only_allow_defined_args(function_to_decorate):
    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        """Internal wrapper function."""
        argspec = inspect.getfullargspec(function_to_decorate)
        valid_names = set(argspec.args + argspec.kwonlyargs)
        if "self" in valid_names:
            valid_names.remove("self")
        for arg_name in kwargs:
            if arg_name not in valid_names:
                raise ValueError("Unknown argument seen '%s', expected: [%s]" % (arg_name, ", ".join(valid_names)))
        return function_to_decorate(*args, **kwargs)

    return _return_wrapped


# function to handle ctrl z and ctrl c
def handler(signum, frame):
    os.system("kill -9 %d" % os.getpid())


signal.signal(signal.SIGINT, handler)


# function to calculate the mean and std of a list in the results dictionary
def cal_mean_std(results) -> dict:
    mean_std = dict()
    for fn in results:
        mean_std[fn] = dict()
        for metric in results[fn]:
            mean = statistics.mean(results[fn][metric]) if len(results[fn][metric]) > 1 else results[fn][metric][0]
            std = statistics.stdev(results[fn][metric]) if len(results[fn][metric]) > 1 else 0
            mean_std[fn][metric] = [mean, std]
    return mean_std


# function to create the environment ofr an anaconda environment
def create_env():
    """Create a new Poetry virtual environment for running models.
    
    Returns:
        tuple: (temp_dir, env_path, python_path, None)
    """
    # Create temp directory for organizing files
    temp_dir = tempfile.mkdtemp()
    env_path = Path(temp_dir).absolute()
    sys.stderr.write(f"Creating working directory: {env_path}...\n")
    
    # Use the current Python interpreter
    python_path = sys.executable
    
    return temp_dir, env_path, python_path, None


# function to execute the cmd
def execute(cmd, wait_when_err=False, raise_err=True):
    print("Running CMD:", cmd)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            sys.stdout.write(line.split("\b")[0])
            if "\b" in line:
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b" * 10 + "\b".join(line.split("\b")[1:-1]))

    if p.returncode != 0:
        if wait_when_err:
            input("Press Enter to Continue")
        if raise_err:
            raise RuntimeError(f"Error when executing command: {cmd}")
        return p.stderr
    else:
        return None


# function to get all the folders benchmark folder
def get_all_folders(models, exclude) -> dict:
    folders = dict()
    if isinstance(models, str):
        model_list = models.split(",")
        models = [m.lower().strip("[ ]") for m in model_list]
    elif isinstance(models, list):
        models = [m.lower() for m in models]
    elif models is None:
        models = [f.name.lower() for f in os.scandir("benchmarks")]
    else:
        raise ValueError("Input models type is not supported. Please provide str or list without space.")
    for f in os.scandir("benchmarks"):
        add = xor(bool(f.name.lower() in models), bool(exclude))
        if add:
            path = Path("benchmarks") / f.name
            folders[f.name] = str(path.resolve())
    return folders


# function to get all the files under the model folder
def get_all_files(folder_path, dataset=None, universe="", all_yaml=False) -> (list, str):
    """Get YAML and requirement files from the model folder.
    
    Args:
        folder_path (str): Path to the model folder
        dataset (str, optional): Dataset name to filter YAML files. If None, all datasets are included.
        universe (str): Universe name to filter YAML files
        all_yaml (bool): If True, return all YAML files regardless of dataset/universe
    
    Returns:
        tuple: (list of YAML files, requirements file path)
    """
    if all_yaml or dataset is None:
        yaml_path = str(Path(f"{folder_path}") / f"*.yaml")
    else:
        if universe != "":
            universe = f"_{universe}"
        yaml_path = str(Path(f"{folder_path}") / f"*{dataset}{universe}.yaml")
    
    req_path = str(Path(f"{folder_path}") / f"*.txt")
    yaml_files = glob.glob(yaml_path)
    req_file = glob.glob(req_path)
    
    if len(yaml_files) == 0:
        return None, None
    else:
        return yaml_files, req_file[0] if req_file else None


# function to retrieve all the results
def get_all_results(folders) -> dict:
    results = dict()
    for fn in folders:
        try:
            exp = R.get_exp(experiment_name=fn, create=False)
        except ValueError:
            # No experiment results
            continue
        recorders = exp.list_recorders()
        result = dict()
        result["annualized_return_with_cost"] = list()
        result["information_ratio_with_cost"] = list()
        result["max_drawdown_with_cost"] = list()
        result["ic"] = list()
        result["icir"] = list()
        result["rank_ic"] = list()
        result["rank_icir"] = list()
        for recorder_id in recorders:
            if recorders[recorder_id].status == "FINISHED":
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
                metrics = recorder.list_metrics()
                if "1day.excess_return_with_cost.annualized_return" not in metrics:
                    print(f"{recorder_id} is skipped due to incomplete result")
                    continue
                result["annualized_return_with_cost"].append(metrics["1day.excess_return_with_cost.annualized_return"])
                result["information_ratio_with_cost"].append(metrics["1day.excess_return_with_cost.information_ratio"])
                result["max_drawdown_with_cost"].append(metrics["1day.excess_return_with_cost.max_drawdown"])
                result["ic"].append(metrics["IC"])
                result["icir"].append(metrics["ICIR"])
                result["rank_ic"].append(metrics["Rank IC"])
                result["rank_icir"].append(metrics["Rank ICIR"])
        results[fn] = result
    return results


# function to generate and save markdown table
def gen_and_save_md_table(metrics, dataset):
    table = "| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for fn in metrics:
        ic = metrics[fn]["ic"]
        icir = metrics[fn]["icir"]
        ric = metrics[fn]["rank_ic"]
        ricir = metrics[fn]["rank_icir"]
        ar = metrics[fn]["annualized_return_with_cost"]
        ir = metrics[fn]["information_ratio_with_cost"]
        md = metrics[fn]["max_drawdown_with_cost"]
        table += f"| {fn} | {dataset} | {ic[0]:5.4f}±{ic[1]:2.2f} | {icir[0]:5.4f}±{icir[1]:2.2f}| {ric[0]:5.4f}±{ric[1]:2.2f} | {ricir[0]:5.4f}±{ricir[1]:2.2f} | {ar[0]:5.4f}±{ar[1]:2.2f} | {ir[0]:5.4f}±{ir[1]:2.2f}| {md[0]:5.4f}±{md[1]:2.2f} |\n"
    pprint(table)
    with open("table.md", "w") as f:
        f.write(table)
    return table


# read yaml, remove seed kwargs of model, and then save file in the temp_dir
def gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir):
    with open(yaml_path, "r") as fp:
        config = yaml.safe_load(fp)
    try:
        del config["task"]["model"]["kwargs"]["seed"]
    except KeyError:
        # If the key does not exists, use original yaml
        # NOTE: it is very important if the model most run in original path(when sys.rel_path is used)
        return yaml_path
    else:
        # otherwise, generating a new yaml without random seed
        file_name = yaml_path.split("/")[-1]
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, "w") as fp:
            yaml.dump(config, fp)
        return temp_path


class ModelRunner:
    def _init_qlib(self, exp_folder_name):
        # init qlib
        GetData().qlib_data(exists_skip=True)
        qlib.init(
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": "file:" + str(Path(os.getcwd()).resolve() / exp_folder_name),
                    "default_exp_name": "Experiment",
                },
            }
        )

    # function to run the all the models
    @only_allow_defined_args
    def run(
        self,
        times=1,
        models=None,
        dataset=None,
        universe="",
        exclude=False,
        qlib_uri: str = "git+https://github.com/luxiaolei/qlib#egg=pyqlib",
        exp_folder_name: str = "run_all_model_records",
        wait_before_rm_env: bool = False,
        wait_when_err: bool = False,
        all_yaml: bool = False,
    ):
        """
        Please be aware that this function can only work under Linux. MacOS and Windows will be supported in the future.
        Any PR to enhance this method is highly welcomed. Besides, this script doesn't support parallel running the same model
        for multiple times, and this will be fixed in the future development.

        Parameters:
        -----------
        times : int
            determines how many times the model should be running.
        models : str or list
            determines the specific model or list of models to run or exclude.
        exclude : boolean
            determines whether the model being used is excluded or included.
        dataset : str, optional
            determines the dataset to be used for each model. If None, all datasets will be used.
        universe  : str
            the stock universe of the dataset.
            default "" indicates that
        qlib_uri : str
            the uri to install qlib with pip
            it could be URI on the remote or local path (NOTE: the local path must be an absolute path)
        exp_folder_name: str
            the name of the experiment folder
        wait_before_rm_env : bool
            wait before remove environment.
        wait_when_err : bool
            wait when errors raised when executing commands

        Usage:
        -------
        Here are some use cases of the function in the bash:

        The run_all_models  will decide which config to run based no `models` `dataset`  `universe`
        Example 1):

            models="lightgbm", dataset="Alpha158", universe="" will result in running the following config
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

            models="lightgbm", dataset="Alpha158", universe="csi500" will result in running the following config
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500.yaml

        .. code-block:: bash

            # Case 1 - run all models multiple times
            python run_all_model.py run 3

            # Case 2 - run specific models multiple times
            python run_all_model.py run 3 mlp

            # Case 3 - run specific models multiple times with specific dataset
            python run_all_model.py run 3 mlp Alpha158

            # Case 4 - run other models except those are given as arguments for multiple times
            python run_all_model.py run 3 [mlp,tft,lstm] --exclude=True

            # Case 5 - run specific models for one time
            python run_all_model.py run --models=[mlp,lightgbm]

            # Case 6 - run other models except those are given as arguments for one time
            python run_all_model.py run --models=[mlp,tft,sfm] --exclude=True

            # Case 7 - run lightgbm model on csi500.
            python run_all_model.py run 3 lightgbm Alpha158 csi500

            # Case 8 - run all models with all datasets
            python run_all_model.py run

            # Case 9 - run specific model with all datasets
            python run_all_model.py run --models=lightgbm

        """
        try:
            self._init_qlib(exp_folder_name)
        except Exception as e:
            sys.stderr.write(f"Failed to initialize qlib: {str(e)}\n")
            return

        # get all folders
        folders = get_all_folders(models, exclude)
        # init error messages:
        errors = dict()
        # run all the model for iterations
        for fn in folders:
            try:
                # get all files
                sys.stderr.write("Retrieving files...\n")
                yaml_paths, req_path = get_all_files(folders[fn], dataset, universe=universe, all_yaml=all_yaml)
                if yaml_paths is None:
                    sys.stderr.write(f"There are no YAML files in {folders[fn]}\n")
                    continue
                
                # Create temp directory for organizing files
                temp_dir, env_path, python_path, _ = create_env()
                
                # Track results by dataset
                dataset_results = {}
                
                # Process each YAML file
                for yaml_path in (yaml_paths if isinstance(yaml_paths, list) else [yaml_paths]):
                    current_dataset = self._get_dataset_name(yaml_path)
                    sys.stderr.write(f"Processing dataset: {current_dataset}\n")
                    
                    # read yaml, remove seed kwargs of model, and then save file in the temp_dir
                    processed_yaml_path = gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir)
                    
                    # Run the model using current environment
                    yaml_name = Path(yaml_path).name
                    sys.stderr.write(f"Running the model: {fn} with config: {yaml_name}...\n")
                    
                    # Run the model using current environment
                    sys.stderr.write(f"Running the model: {fn}...\n")
                    for i in range(times):
                        try:
                            sys.stderr.write(f"Running iteration {i+1}...\n")
                            cmd = f"qrun {processed_yaml_path}"
                            errs = execute(
                                cmd,
                                wait_when_err=wait_when_err,
                            )
                            if errs is not None:
                                _errs = errors.get(fn, {})
                                _errs.update({i: errs})
                                errors[fn] = _errs
                            sys.stderr.write("\n")
                        except Exception as e:
                            sys.stderr.write(f"Failed iteration {i+1} for model {fn}: {str(e)}\n")
                            continue
                
                # Clean up temp directory
                shutil.rmtree(temp_dir)

                # Generate results for each dataset
                for current_dataset, results in dataset_results.items():
                    if len(results) > 0:
                        results = cal_mean_std(results)
                        gen_and_save_md_table(results, current_dataset)
                        # Move results with dataset-specific naming
                        result_path = f"table_{current_dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.md"
                        shutil.move("table.md", result_path)

            except Exception as e:
                sys.stderr.write(f"Failed to run model {fn}: {str(e)}\n")
                continue

        # print errors
        sys.stderr.write(f"Here are some of the errors of the models...\n")
        pprint(errors)
        self._collect_results(exp_folder_name, dataset)

    def _collect_results(self, exp_folder_name, dataset):
        folders = get_all_folders(exp_folder_name, dataset)
        # getting all results
        sys.stderr.write(f"Retrieving results...\n")
        results = get_all_results(folders)
        if len(results) > 0:
            # calculating the mean and std
            sys.stderr.write(f"Calculating the mean and std of results...\n")
            results = cal_mean_std(results)
            # generating md table
            sys.stderr.write(f"Generating markdown table...\n")
            gen_and_save_md_table(results, dataset)
            sys.stderr.write("\n")
        sys.stderr.write("\n")
        # move results folder
        shutil.move(exp_folder_name, exp_folder_name + f"_all_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        shutil.move("table.md", f"table_all_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.md")

    def _get_dataset_name(self, yaml_path):
        """Extract dataset name from YAML file path."""
        filename = Path(yaml_path).stem
        # Assuming format: workflow_config_modelname_datasetname[_universe]
        parts = filename.split('_')
        if len(parts) >= 4:
            return parts[3]  # Return dataset name part
        return "unknown"


if __name__ == "__main__":
    fire.Fire(ModelRunner)  # run all the model
