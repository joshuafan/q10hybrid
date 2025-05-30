# Ecosystem respiration experiments (ScIReN paper)

This section explains how to reproduce Tables 1-2 (Ecosystem Respiration experiments) in the ScIReN paper.

## Usage

If running locally on Mac first run:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The following commands will reproduce Table 1.
```
python experiments/1_linear.py --model pure_nn --num_layers 2 --stage final;
python experiments/1_linear.py --model nn --rb_constraint softplus --num_layers 2 --stage final;
python experiments/1_linear.py --model nn --rb_constraint relu --num_layers 2 --stage final;
python experiments/1_linear.py --model kan --rb_constraint softplus --num_layers 1 --stage final;
python experiments/1_linear.py --model kan --rb_constraint relu --num_layers 1 --stage final;
```

A folder will be created under `logs`, with the name given by the `--log_dir` argument (plus more metadata). Inside the folder, you can view the final results in `results_summary_file.csv`, or view visualizations for each run inside the `version_XX` folders.

NOTE: These results may not exactly match the submitted paper, as the results in the paper were run without setting `deterministic=True` in the PyTorch Lightning Trainer (`TRAINER_ARGS`), and hence are slightly nondeterministic (even though seeds were prescribed). However it should be very close. We will fix this in the revised version.


The following commands will reproduce Table 2.
```
python experiments/2_abs.py --model pure_nn --num_layers 2 --stage final;
python experiments/2_abs.py --model nn --rb_constraint softplus --num_layers 2 --stage final;
python experiments/2_abs.py --model nn --rb_constraint relu --num_layers 2 --stage final;
python experiments/2_abs.py --model kan --rb_constraint relu --num_layers 1 --stage final;
python experiments/2_abs.py --model kan --rb_constraint relu --num_layers 2 --hidden_dim 8 --stage final
```

See the documentation in those files (`experiments/1_linear.py`, `experiments/2_abs.py`) for more details. 

To do hyperparameter tuning, change `--stage final` to `--stage tuning`, and create a hyperparameter grid in the `__main__` function.

To run ablations, `experiments/1a_linear_ablation.py` and `experiments/2a_abs_ablation.py` provide examples.

## Code summary

- `experiments/` folder contains main methods that start each experiment (linear Rb, abs Rb, and ablations).
   * `search_space` can be used to specify a grid of hyperparameters.
- `model/hybrid.py` contains LightningModules for the core models.
   * For pure-NN, set `model='purenn'`. For Blackbox-Hybrid, set `model='nn'`. For ScIReN, set `model='kan'`.
   * You can set `rb_constraint` to `softplus` (nonlinear) or `relu` (linear).
   * The module defines `train_step`, `validation_step`, and creates visualizations every 10 epochs.
- `project/fluxdata.py` is where the synthetic labels are constructed, and we remove high-temperature parameters from the training set. `rb_synth=9` means linear, and `rb_synth=8` means abs (nonlinear).


## Optuna notes

This codebase supports using Optuna for hyperparameter tuning and visualization. To do this, run 
```
pip install optuna-dashboard
optuna-dashboard sqlite:///./logs/1_abs_pure_nn_layers=2_constraint=softplus/optuna.db --port 8081
```
(You should replace the path with the path to the `optuna.db` that was created inside the log directory.)

If this is being run on a remote server: run this command on your local machine
```
ssh -N <username>@<server> -L 8081:localhost:8081
```
(Or if it's being run on a compute node that's separate from the head node:)
```
ssh -N -J <username>@<server> <username>@<node> -L 8081:localhost:8081
```

Finally, in all cases, navigate to http://127.0.0.1:8081/ in a local web browser.




# Original README

The code base was derived from this repo: https://github.com/bask0/q10hybrid

Here is the original documentation from the repo:

Author: B. Kraft [bkraf@bgc-jena.mpg.de]

<div align="center">


# Hybrid modeling of ecosystem respiration temperature sensitivity




</div><br><br>

## Description

Q10 hybrid modeling experiment for a book chapter.

## How to run

First, install dependencies.

```bash
# clone project
git clone https://github.com/bask0/q10hybrid

# Optional: create and activate new conda environment.
conda create --yes --name q10hybrid python=3.6
conda activate q10hybrid

# install project
cd q10hybrid
pip install -e .
pip install -r requirements.txt
```

## Q10 hybrid modeling experiment

Base respiration is simulated using observed short-wave irradiation and the delta thereof. Ecosyste respiration is calculated using the [Q10 approach](https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)).

<img src="https://render.githubusercontent.com/render/math?math=Rb_\mathrm{syn} = f(W_\mathrm{in, pot}, \Delta SW_\mathrm{in, pot})"><br>

<img src="https://render.githubusercontent.com/render/math?math=RECO_\mathrm{syn} = Rb_\mathrm{syn} \cdot 1.5^{0.1 \cdot (TA - 15.0)}">

## Experiment

Estimate Q10 in two different setups:

* Rb=NN(SW_in, dSW_in)
* Rb=NN(SW_in, dSW_in, T)

We investigate wheter we can estimate Q10 in both cases robustly and how model hyperparameters (here: dropout={0.0, 0.2, 0.4, 0.6}) impact the results.

![data](/analysis/plots/data.png)


Run experiments:

```bash
# Create a new study (delete old runs).
python experiments/experiment_01.py --create_study
```

```bash
# Start first process on GPU 0.
CUDA_VISIBLE_DEVICES="0" python experiments/experiment_01.py
```

To work on independent runs in parallel, just call the study again from another terminal!

```bash
# Start a second process on GPU 1.
CUDA_VISIBLE_DEVICES="1" python experiments/experiment_01.py
```

Alternatively, you can use `run_experiment.py` to create a new study and spawn multiple processes, for example with 12 jobs distributed
on 4 GPUs (0,1,2,3). 
```bash
# Start a second process on GPU 1.
CUDA_VISIBLE_DEVICES="0,1,2,3" python run_experiment.py --num_jobs 12
```

Use `analysis/analysis.ipynb` for evaluation.

## Note

> From the `optuna` doc: `GridSampler` automatically stops the optimization if all combinations in the passed `search_space` have already been evaluated, internally invoking the `stop()` method.

The grid search runs too many combinations, they are cleane in `analysis/analysis.ipynb`.

### Results 

Q10 estimation **without** (top) and **with** (bottom) air temperature as predictor:

![training progress](/analysis/plots/q10_training.png)

Q10 estimation and loss for different HPs.

![Q10 interactions](/analysis/plots/q10_interactions.png)

## Citation

```tex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
