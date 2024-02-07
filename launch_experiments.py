"""
Commands for launching experiments.
"""

import glob
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import GPUtil
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.utils.scripts import run_command
from typing_extensions import Annotated, Literal

from experiments import arguments_list_of_lists as experiments_list
from pds.utils.experiment_utils import get_experiment_name_and_argument_combinations


def launch_experiments(jobs, dry_run: bool = False, gpu_ids: Optional[List[int]] = None):
    """Launch the experiments.
    Args:
        jobs: list of commands to run
        dry_run: if True, don't actually run the commands
        gpu_ids: list of gpu ids that we can use. If none, we can use any
    """

    num_jobs = len(jobs)
    while jobs:
        # get GPUs that capacity to run these jobs
        gpu_devices_available = GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1)
        print("-" * 80)
        print("Available GPUs: ", gpu_devices_available)
        if gpu_ids:
            print("Restricting to subset of GPUs: ", gpu_ids)
            gpu_devices_available = [gpu for gpu in gpu_devices_available if gpu in gpu_ids]
            print("Using GPUs: ", gpu_devices_available)
        print("-" * 80)

        if len(gpu_devices_available) == 0:
            print("No GPUs available, waiting 10 seconds...")
            time.sleep(10)
            continue

        # thread list
        threads = []
        while gpu_devices_available and jobs:
            gpu = gpu_devices_available.pop(0)
            command = f"CUDA_VISIBLE_DEVICES={gpu} " + jobs.pop(0)

            def task():
                print(f"Command:\n{command}\n")
                if not dry_run:
                    _ = run_command(command, verbose=False)
                # print("Finished command:\n", command)

            threads.append(threading.Thread(target=task))
            threads[-1].start()

            # NOTE(ethan): here we need a delay, otherwise the wandb/tensorboard naming is messed up... not sure why
            if not dry_run:
                time.sleep(5)

        # wait for all threads to finish
        for t in threads:
            t.join()

        # print("Finished all threads")
    print(f"Finished all {num_jobs} jobs")


@dataclass
class ExperimentConfig(PrintableConfig):
    """Experiment config code."""

    dry_run: bool = False
    output_folder: Optional[Path] = None
    gpu_ids: Optional[List[int]] = None

    def main(self, dry_run: bool = False) -> None:
        """Run the code."""
        raise NotImplementedError


@dataclass
class Train(ExperimentConfig):
    """Train nerfbusters models."""

    experiment_name: Literal["pds_gen"] = "pds_gen"
    """Which experiment to run"""

    def main(self, dry_run: bool = False):
        jobs = []
        experiment_names = []
        argument_combinations = []

        if self.experiment_name == "pds_gen":
            # For our baseline experiments in the paper
            experiment_names, argument_combinations = get_experiment_name_and_argument_combinations(
                experiments_list
            )
        else:
            raise ValueError(self.experiment_name)


        for experiment_name, argument_string in zip(experiment_names, argument_combinations):
            
            base_cmd = f"python3 pds/pds_gen.py --config.experiment_name {experiment_name}"
            jobs.append(f" {base_cmd} {argument_string}")

        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


Commands = Annotated[Train, tyro.conf.subcommand(name="train")]
# Union[
#     ,
#     # Annotated[Render, tyro.conf.subcommand(name="render")],
#     # Annotated[Metrics, tyro.conf.subcommand(name="metrics")],
# ]


def main(
    benchmark: ExperimentConfig,
):
    """Script to run the benchmark experiments for the Nerfstudio paper.
    - nerfacto-on-mipnerf360: The MipNeRF-360 experiments on the MipNeRF-360 Dataset.
    - nerfacto-ablations: The Nerfacto ablations on the Nerfstudio Dataset.
    Args:
        benchmark: The benchmark to run.
    """
    benchmark.main(dry_run=benchmark.dry_run)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Train))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa