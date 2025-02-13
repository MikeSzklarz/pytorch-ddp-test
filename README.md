## Files

* [src/datautils.py](src/datautils.py): Contains the `MyTrainDataset` class used for creating the training dataset.
  * `MyTrainDataset`: A custom dataset class that generates random data for training.

* [src/multinode.py](src/multinode.py): Implements DDP on multiple nodes using Torchrun (and optionally Slurm).
  * `ddp_setup()`: Sets up the DDP environment.
  * `Trainer`: A class that handles the training process, including loading snapshots, running batches and epochs, and saving snapshots.
  * `load_train_objs()`: Loads the training dataset, model, and optimizer.
  * `prepare_dataloader()`: Prepares the DataLoader with a distributed sampler.
  * `main()`: The main function that sets up DDP, loads training objects, prepares the DataLoader, and starts the training process.
  
* [sbatch_run.sh](sbatch_run.sh): A Slurm script to launch the training job.
  * Sets up the Slurm job parameters.
  * Activates the conda environment.
  * Configures the NCCL backend for communication.
  * Uses `torchrun` to start the training script on multiple nodes.

* [environment.yml](environment.yml): Conda environment configuration file.
  * Specifies the Python version and dependencies required for the project.

* [requirements.txt](requirements.txt): Lists the Python packages required for the project.
  * Specifies the required version of PyTorch.

## Usage

### Setting Up the Environment

First, make sure you have Conda installed. Then, create a new environment with all the dependencies by running the following command:

```sh
conda env create -f environment.yml
```

To activate the environment. We do not need this but :

```sh
conda activate ddp-test
```

### Running the Training Script

To run the training script on a Slurm cluster, use the `sbatch_run.sh` script. This script will set up the Slurm job and start the training process using `torchrun`. Make sure you are in the directory containing the `sbatch_run.sh` script and run the following command:

```sh
sbatch ./sbatch_run.sh
```

This will submit the job to the Slurm cluster, and the training process will start. You can monitor the progress of the job using the `squeue` command. The script will run only on a single node that it will choose automatically if the resources are available. To run more than 1 node you can change the `-N` when running the `sbatch_run.sh` script.

```
sbatch -N 2 ./sbatch_run.sh
```

## Notes

* Ensure that the `logs` directory exists before running the `sbatch_run.sh` script, as it will save the output log files there.
* The `sbatch_run.sh` script assumes that each node has one GPU. Adjust the script if your setup is different.
* The `NCCL_SOCKET_IFNAME` environment variable is set to ensure that the NCCL backend uses the correct network interface for communication.

For more details on DDP and distributed training with PyTorch, refer to the [PyTorch DDP tutorial series](https://pytorch.org/tutorials/beginner/ddp_series_intro.html).