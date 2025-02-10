# distributed-pytorch

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

## Files
* [src/datautils.py](src/datautils.py): Contains the `MyTrainDataset` class used for creating the training dataset.

* [src/multinode.py](src/multinode.py): DDP on multiple nodes using Torchrun (and optionally Slurm)
    * [sbatch_run.sh](sbatch_run.sh): slurm script to launch the training job

## Usage

First make sure to have conda installed and run the following command to create a new environment with all the dependencies:

`conda env create -f environment.yml`

Then activate the environment:

`conda activate ddp_tutorial`


To run the training script on a Slurm cluster, use the `sbatch_run.sh` script


```sh
sbatch sbatch_run.sh