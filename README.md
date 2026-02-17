# Code and Data for "Sampling on Discrete Spaces with Temporal Point Processes"

This repository contains all of the code and generated data used for the simulation study section of the paper "Sampling on Discrete Spaces with Temporal Point Processes". For reproducibility purposes, the random seeds set in the files are the same seeds used in the simulation study.

>:warning: Be aware that reproducing all 3,150 ESS values is likely to take weeks if done sequentially, primarily due to the extremely long runs of the stochastic processes that we commit to (10,000,000 jumps in state per run).

>:keyboard: If you are only interested in seeing an implementation of the point process sampler, see the `pointProcess` function in `simulation/simulation.py`.

## Dependencies
- Python (version 3.6 and above)
- NumPy

To regenerate the plots in the paper, you will also need:
- Matplotlib
- a LaTeX distribution

## Repository Structure
The code is split into three folders:
- :open_file_folder: `parameter_gen` contains code to generate the weights and biases used in the Sherrington-Kirkpatrick (fully-connected Ising) and stochastic neural network models.
    - Execute `python ising_parameter_gen.py` and `python neural_parameter_gen.py` to generate the parameter files `ising_parameters.npz` and `neural_parameters.npz` respectively.
    - The generated parameter files used in the simulation study are located in the `simulation` folder.
- :open_file_folder: `simulation` contains code to run all of the simulations as described in the paper.
    - Execute `python simulation.py -h` to see usage instructions.
    - The argument `run` determines the random seed for the run. If you wish to reproduce a result used in the paper, `run` should be set to an integer from `0` to `9`.
    - Variations in floating-point arithmetic implementations between machines may cause extremely small differences in reproduced ESS values. Reproducing CPU time is not expected.
    - The data generated in the simulation study is located in the `plot/data` folder.
- :open_file_folder: `plot` contains code to reproduce the plots in the paper.
    - Execute `python plot.py` to reproduce the `plots.eps` graphic displayed in the paper, using data from the 315 CSV files in `plot/data`.
    - The filenames in `plot/data` take the form `sampler_model_scale.csv`, where `scale` is a float corresponding to the horizontal axis of the plots ($\Lambda$, $\beta$, or $\alpha$).
    - Each CSV file contains ten rows and two columns. Each row corresponds to one of the ten runs made for each sampler and distribution pair. The first column reports ESS. The second column reports CPU time in seconds.

## Usage Example

To regenerate one of the rows in `plot/data/point_poisson_0.1.csv` execute
```text
python simulation.py point poisson 0.1 <run>
```
where `<run>` is an integer from `0` to `9`. The ordering of the rows in each CSV file does not necessarily correspond to the value of `<run>`.