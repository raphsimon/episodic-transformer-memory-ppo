import sys
import logging
import torch
import argparse
import traceback
from trainer import PPOTrainer
from yaml_parser import YamlParser

import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from sqlalchemy.pool import NullPool

"""
This script performs a hyperparameter search for the PPO-TrXL algorithm.
What's expected as input is first and foremost a complete config, just
so that all the values can be set to perform some of the decisions.
Then, we sample the hyperparameters and create and instance of the
Learner class and train the agent. The reported score is the return
after the last performed evaluation during the training run. This 
value is then stored alongside other trial details in the database,
and also used to prune other trials.
"""

def optimize_hyperparameters(study_name, optimize_trial, database_url, 
                             n_trials=100, max_total_trials=None):
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    print(f"Provided database: {database_url}")

    sqlite_timeout = 300
    engine_kwargs = None
    if "sqlite" in database_url:
        engine_kwargs={
            'connect_args': {'timeout': sqlite_timeout},
        }
    elif "postgresql" in database_url:
        engine_kwargs = {
            "poolclass": NullPool,
            "connect_args": {
                "connect_timeout": 60,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10
            }
        }
    print(f'Using {engine_kwargs} for engine_kwargs')

    storage = optuna.storages.RDBStorage(
        database_url,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=2),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize'
    ) # No sampler is specified, so a default sampler (TPE) is used.
    
    if max_total_trials is not None:
        # Note: we count already running trials here otherwise we get
        #  (max_total_trials + number of workers) trials in total.
        counted_states = [
            TrialState.COMPLETE,
            TrialState.RUNNING,
            TrialState.PRUNED,
        ]
        completed_trials = len(study.get_trials(states=counted_states))
        if completed_trials < max_total_trials:
            study.optimize(
                optimize_trial,
                callbacks=[
                    MaxTrialsCallback(
                        max_total_trials,
                        states=counted_states,
                    )   
                ],
                gc_after_trial=True
            )
    else:
        study.optimize(optimize_trial, n_trials=n_trials, gc_after_trial=True)

    # Print results
    print(f"Best value (accuracy): {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"{param}: {value}")

    return study

def suggest_ppo_trxl_params(trial: optuna.Trial):
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return: dictionary with all the sampled hyperparameters.
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.975, 0.99])
    lamda = trial.suggest_categorical("gae-lambda", [0.9, 0.95, 0.99])
    epochs = trial.suggest_int("epochs", 2, 4)
    n_mini_batch = trial.suggest_categorical("n_mini_batch", [4, 8])
    value_loss_coefficient = trial.suggest_categorical("value_loss_coefficient", [0.2, 0.3, 0.5])
    hidden_layer_size = trial.suggest_categorical("hidden_layer_size", [64, 128, 256])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.25, 0.3, 0.5, 1.0])
    transformer_num_blocks = trial.suggest_categorical("transformer_num_blocks", [2, 3, 4])
    transformer_embed_dim = trial.suggest_categorical("transformer_embed_dim", [128, 256, 384])
    transformer_num_heads = trial.suggest_categorical("transformer_num_heads", [1, 4, 8])
    transformer_memory_length = trial.suggest_categorical("transformer_memory_length", [16, 32, 64])
    transformer_positional_encoding = trial.suggest_categorical("transformer_positional_encoding", ["", "relative", "learned"])
    transformer_layer_norm = trial.suggest_categorical("transformer_layer_norm", ["", "pre", "post"])
    transformer_gtrxl = trial.suggest_categorical("transformer_gtrxl", [True, False])
    transformer_gtrxl_bias = trial.suggest_categorical("transformer_gtrxl_bias", [0.0, 2.0])
    learning_rate_initial = trial.suggest_categorical("learning_rate_initial", [2.0e-4, 2.75e-4, 3.0e-4, 3.5e-4])
    clip_range_initial = trial.suggest_categorical("clip_range_initial", [0.1, 0.2, 0.3])

    hyperparams = {
        "gamma": gamma,
        "lamda": lamda,
        "epochs": epochs,
        "n_mini_batch": n_mini_batch,
        "value_loss_coefficient": value_loss_coefficient,
        "hidden_layer_size": hidden_layer_size,
        "max_grad_norm": max_grad_norm,
        "transformer": {
            "num_blocks": transformer_num_blocks,
            "embed_dim": transformer_embed_dim,
            "num_heads": transformer_num_heads,
            "memory_length": transformer_memory_length,
            "positional_encoding": transformer_positional_encoding,
            "layer_norm": transformer_layer_norm,
            "gtrxl": transformer_gtrxl,
            "gtrxl_bias": transformer_gtrxl_bias,
        },
        "learning_rate_schedule": {
            "initial": learning_rate_initial,
            "final": 3.0e-5,
            "power": 1.0,
            "max_decay_steps": 10,
        },
        "beta_schedule": {
            "initial": 0.001,
            "final": 0.0001,
            "power": 1.0,
            "max_decay_steps": 10,
        },
        "clip_range_schedule": {
            "initial": clip_range_initial,
            "final": clip_range_initial,
            "power": 1.0,
            "max_decay_steps": 10,
        },
    }
    
    return hyperparams

def optimize_trial(trial):
    # Determine the device to be used for training and set the default tensor type
    if not args.cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    
    try:
        # Here we need to sample hyperparams and run the training
        sampled_hyperparams = suggest_ppo_trxl_params(trial)
        # Merge sampled hyperparameters with the base config
        config.update(sampled_hyperparams)
        # Initialize the PPO trainer and commence training
        trainer = PPOTrainer(config, run_id=str(trial.number), device=device)
        score = trainer.run_training(save_model=False, evaluate_model=True)
        # The score we use is the mean undiscounted return over 10 episodes
        print(f"Trial {trial.number} finished with score {score}")
        trainer.close()
        return score
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        print(traceback.format_exc())
        trainer.close()
        raise  # Re-raise to let Optuna handle the failure

if __name__ == '__main__':

    def argparser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True,
                            help="Path to the yaml config file, containing the basic hyperparameters.")
        parser.add_argument("--db-url", type=str, required=True,
                            help="Example: sqlite:///optuna.db")
        parser.add_argument("--trials", type=int, default=50, 
                            help="Number of trials to run.")
        parser.add_argument("--max_total_trials", type=int, default=None,
                            help="Maximum total number of trials to run. We count running, pruned, and completed trials.")
        parser.add_argument("--cpu", action="store_true",
                            help="Force training on CPU.")
        return parser.parse_args()
    
    args = argparser()
    config = YamlParser(args.config).get_config()
    env_name = config["environment"]["name"]
    study_name = f"PPO-TrXL_{env_name}"

    study = optimize_hyperparameters(study_name, optimize_trial, args.db_url, 
                                     args.trials, args.max_total_trials)

