import logging

logging.basicConfig(level=logging.INFO)

from experiment_runner.def_nn_experiment import NN_tune_trainable
import ray

import json
import torch
from pathlib import Path

PATH_ROOT = Path(".")

def main():
    cpus = 12
    gpus = 1

    cpu_per_trial = 12
    gpu_fraction = ((gpus * 100) // (cpus / cpu_per_trial)) / 100
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_fraction}

    experiment_name = "VIT_FINETUNING"

    config = json.load(open("example_configs/vision/config_vit_finetuning.json", "r"))


    net_dir = PATH_ROOT.joinpath("vit_finetuning")
    try:
        net_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    print(f"Saving models to {net_dir.absolute()}")

    config["cuda"] = True if gpus > 0 and torch.cuda.is_available() else False

    context = ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
    )
    assert ray.is_initialized() == True
    print(f"started ray. running dashboard under {context.dashboard_url}")
    if config.get("dataset::dump", None) is not None:
        print(f"loading dataset from {config['dataset::dump']}")
        dataset = torch.load(config["dataset::dump"])
        dataset_ref = ray.put(dataset)

    config["dataset::ref"] = dataset_ref


    experiment = ray.tune.Experiment(
        name = experiment_name,
        run = NN_tune_trainable,
        stop = {
            "training_iteration": config["training::epochs_train"]
        },
        checkpoint_config= ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config.get("training::output_epoch", 1),
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=net_dir.absolute(),
        resources_per_trial=resources_per_trial,
    )

    ray.tune.run_experiments(
        experiments = experiment,
        resume = False,
        reuse_actors = False,
        verbose = 3,
    )

    ray.shutdown()

if __name__ == "__main__":
    main()



