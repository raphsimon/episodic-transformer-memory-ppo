import torch
from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser
import time

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
        --save-model               Save the model after training [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    save_model = options["--save-model"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser(options["--config"]).get_config()

    # Determine the device to be used for training and set the default tensor type
    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Start the timer and print the start time
    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training(save_model)
    trainer.close()

    # Calculate and print the total duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training duration: {duration:.2f} seconds")

if __name__ == "__main__":
    main()