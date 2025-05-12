
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

### Usage

If running locally on Mac:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

@joshuafan:
Create study. Run this everytime the command-line arguments changed.
```
python experiments/experiment_01.py --create_study
python experiments/experiment_hardconstraint.py --create_study

```

Commands
```
python experiments/experiment_01.py --model nn --rb_constraint softplus
python experiments/experiment_linear.py --model nn --rb_constraint softplus --num_layers 1
python experiments/experiment_hardconstraint.py --model kan --rb_constraint relu --num_layers 1 --learning_rate 1e-2 --weight_decay 0 --single_seed
```

KAN with softplus (so it has to learn inverse softplus)
python experiments/experiment_hardconstraint.py --model kan --rb_constraint relu --num_layers 1 --learning_rate 1e-2 --weight_decay 0 --single_seed


Hyperparameter tuning
```
python experiments/20250428_hyperparam_hardconstraint.py --model kan --rb_constraint relu --num_layers 1 
```


## Optuna notes

Install Optuna dashboard
```
pip install optuna-dashboard

optuna-dashboard sqlite:///./logs/20250509_abs_pure_nn_layers=2_constraint=softplus/optuna.db --port 8081
Best 5, Params = [seed: 0, lambda_kan_l1: 1e-10, lambda_kan_entropy: 1e-10, lambda_kan_coefdiff2: 1e-10, learning_rate: 0.1, weight_decay: 0.0]

optuna-dashboard sqlite:///./logs/20250509_abs_nn_layers=2_constraint=softplus/optuna.db --port 8081
Best 1, Params = [seed: 0, lambda_kan_l1: 1e-10, lambda_kan_entropy: 1e-10, lambda_kan_coefdiff2: 1e-10, learning_rate: 0.001, weight_decay: 0.001]

optuna-dashboard sqlite:///./logs/20250509_abs_nn_layers=2_constraint=relu/optuna.db --port 8081
Best 2, Params = [seed: 0, lambda_kan_l1: 1e-10, lambda_kan_entropy: 1e-10, lambda_kan_coefdiff2: 1e-10, learning_rate: 0.001, weight_decay: 0.0001]

optuna-dashboard sqlite:///./logs/20250509_abs_kan_layers=1_constraint=relu/optuna.db --port 8081
Best 11, Params = [seed: 0, lambda_kan_l1: 0.01, lambda_kan_entropy: 0.01, lambda_kan_coefdiff2: 0.01, learning_rate: 0.01, weight_decay: 0.0001]

optuna-dashboard sqlite:///./logs/20250509_abs_kan_layers=2_constraint=relu/optuna.db --port 8081
Best 36, Params = [seed: 0, lambda_kan_l1: 0.01, lambda_kan_entropy: 0.1, lambda_kan_coefdiff2: 1.0, learning_rate: 0.01, weight_decay: 0.0001]



optuna-dashboard sqlite:///./logs/20250509_linear_pure_nn_layers=2_constraint=softplus/optuna.db --port 8081
Best 5, Params = [seed: 0, lambda_kan_l1: 1e-10, lambda_kan_entropy: 1e-10, lambda_kan_coefdiff2: 1e-10, learning_rate: 0.1, weight_decay: 0.0]

optuna-dashboard sqlite:///./logs/20250509_linear_nn_layers=2_constraint=softplus/optuna.db --port 8081
Best 5, Params = [seed: 0, lambda_kan_l1: 1e-10, lambda_kan_entropy: 1e-10, lambda_kan_coefdiff2: 1e-10, learning_rate: 0.1, weight_decay: 0.0]

optuna-dashboard sqlite:///./logs/20250509_linear_nn_layers=2_constraint=relu/optuna.db --port 8081
Best 5, Params = [seed: 0, lambda_kan_l1: 1e-10, lambda_kan_entropy: 1e-10, lambda_kan_coefdiff2: 1e-10, learning_rate: 0.1, weight_decay: 0.0]

optuna-dashboard sqlite:///./logs/20250509_linear_kan_layers=1_constraint=softplus/optuna.db --port 8081
Best 5, Params = [seed: 0, lambda_kan_l1: 0.01, lambda_kan_entropy: 0.01, lambda_kan_coefdiff2: 0.1, learning_rate: 0.1, weight_decay: 0.0001]

optuna-dashboard sqlite:///./logs/20250509_linear_kan_layers=1_constraint=relu/optuna.db --port 8081
Best 8, Params = [seed: 0, lambda_kan_l1: 0.01, lambda_kan_entropy: 0.01, lambda_kan_coefdiff2: 1.0, learning_rate: 0.01, weight_decay: 0.0001]









# If this is being run on remote server on a compute node c0011 (different from head node):
ssh -N -J jyf6@aida.cac.cornell.edu jyf6@c0011 -L 8081:localhost:8081
# If this is being run on a remote server (head node)
ssh -N jyf6@aida.cac.cornell.edu -L 8081:localhost:8081
# If this is run locally, ignore the above.
# In all cases, go to local browser.
http://127.0.0.1:8081/


```



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
