"""
Synthetic linear
python experiments/20250430_hyperparam_synth0_nn.py --model nn --rb_constraint softplus --num_layers 2
python experiments/20250430_hyperparam_synth0_nn.py --model kan --rb_constraint softplus --num_layers 1

"""


import pytorch_lightning as pl
import optuna
import xarray as xr

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import os
import shutil
from argparse import ArgumentParser
from datetime import datetime

from project.fluxdata import FluxData
from models.hybrid import Q10Model

# Hardcoded `Trainer` args. Note that these cannot be passed via cli.
TRAINER_ARGS = dict(
    max_epochs=100,
    # log_every_n_steps=1,
    # weights_summary=None,  # Doesn't exist in current version of pytorch lightning https://stackoverflow.com/a/74706463
    accelerator="auto",
    devices="auto",
    strategy="auto",
)


class Objective(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, trial: optuna.trial.Trial) -> float:
        # FIXED hyperparameters
        rb_synth = 0
        reco_noise_std = 0.0
        subset_frac = 1.0
        q10_init = 0.5
        seed = 0
        use_ta = True
        # dropout = 0.0
        kan_base_fun = 'silu_identity'
        kan_affine_trainable = True
        kan_grid = 30
        kan_grid_margin = 1.0
        kan_update_grid = 1
        kan_noise = 0.3
        lambda_jacobian_l05 = 0.0
        weight_decay = 0.0
        lambda_param_violation = 0.0
        lambda_kan_l1 = 0.0
        lambda_jacobian_l1 = 0.0

        # Loss weights / model complexity
        # lambda_param_violation = trial.suggest_float('lambda_param_violation', 0.0, 1.0)
        # lambda_kan_l1 = trial.suggest_float('lambda_kan_l1', 0.0, 1.0)
        lambda_kan_entropy = trial.suggest_float('lambda_kan_entropy', 1e-3, 1e-2, log=True)
        lambda_kan_coefdiff = trial.suggest_float('lambda_kan_coefdiff', 1e-3, 1e-2, log=True)
        lambda_kan_coefdiff2 = trial.suggest_float('lambda_kan_coefdiff2', 1e-3, 1e-2, log=True)  #, log=True)
        # lambda_jacobian_l1 = trial.suggest_float('lambda_jacobian_l1', 0.0, 1.0)

        # Optimization
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)

        weight_decay = 0.0  # trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        dropout = 0.0  # trial.suggest_float('dropout', 0.0, 0.5)
        # kan_base_fun = trial.suggest_categorical('kan_base_fun', ['silu_identity', 'silu', 'identity', 'zero'])
        # kan_affine_trainable = trial.suggest_categorical('kan_affine_trainable', [True, False])
        # kan_grid = trial.suggest_int('kan_grid', 3, 50)
        # kan_grid_margin = trial.suggest_float('kan_grid_margin', 0.0, 2.0)
        # kan_update_grid = trial.suggest_categorical('kan_update_grid', [0, 1])
        # kan_noise = trial.suggest_float('kan_noise', 0.1, 0.5, log=True)
        # lambda_jacobian_l05 = trial.suggest_float('lambda_jacobian_l05', 0.0, 100.0)
        # lambda_robustness = trial.suggest_float('lambda_robustness', 0.0, 100.0)

        if use_ta:
            features = ['sw_pot', 'dsw_pot', 'ta']
        else:
            features = ['sw_pot', 'dsw_pot']

        pl.seed_everything(seed)

        # Further variables used in the hybrid model.
        physical = ['ta']

        # Target (multiple targets not possible currently).
        targets = ['reco']

        # Find variables that are only needed in physical model but not in NN.
        physical_exclusive = [v for v in physical if v not in features]

        # ------------
        # data
        # ------------
        ds = xr.open_dataset(self.args.data_path)

        fluxdata = FluxData(
            ds,
            features=features + physical_exclusive,
            targets=targets,
            context_size=1,
            train_time=slice('2003-01-01', '2006-12-31'),
            valid_time=slice('2007-01-01', '2007-12-31'),
            test_time=slice('2008-01-01', '2008-12-31'),
            batch_size=self.args.batch_size,
            data_loader_kwargs={'num_workers': 2, 'persistent_workers': True},  # persistent_workers=True necessary to avoid long pause between epochs https://github.com/Lightning-AI/pytorch-lightning/issues/10389 
            subset_frac=subset_frac,
            rb_synth=rb_synth,
            reco_noise_std=reco_noise_std,
            plot_dir=self.args.log_dir)

        train_loader = fluxdata.train_dataloader()
        val_loader = fluxdata.val_dataloader()
        test_loader = fluxdata.test_dataloader()

        # Create empty xr.Dataset, will be used by the model to save predictions every epoch.
        max_epochs = TRAINER_ARGS['max_epochs']
        ds_pred = fluxdata.target_xr('valid', varnames=['reco', 'rb'], num_epochs=max_epochs)
        ds_test = fluxdata.target_xr('test', varnames=['reco', 'rb'], num_epochs=max_epochs)

        # ------------
        # model
        # ------------
        model = Q10Model(
            features=features,
            targets=targets,
            norm=fluxdata._norm,
            ds=ds_pred,
            ds_test=ds_test,
            q10_init=q10_init,
            hidden_dim=self.args.hidden_dim,
            num_layers=self.args.num_layers,
            learning_rate=learning_rate,
            dropout=dropout,
            weight_decay=weight_decay,
            lambda_param_violation=lambda_param_violation,
            lambda_kan_l1=lambda_kan_l1,
            lambda_kan_entropy=lambda_kan_entropy,
            lambda_kan_coefdiff=lambda_kan_coefdiff,
            lambda_kan_coefdiff2=lambda_kan_coefdiff2,
            lambda_jacobian_l1=lambda_jacobian_l1,
            lambda_jacobian_l05=lambda_jacobian_l05,
            kan_grid=kan_grid,
            kan_update_grid=kan_update_grid,
            kan_grid_margin=kan_grid_margin,
            kan_noise=kan_noise,
            kan_base_fun=kan_base_fun,
            kan_affine_trainable=kan_affine_trainable,
            num_steps=len(train_loader) * max_epochs,
            model=self.args.model,
            rb_constraint=self.args.rb_constraint)

        # ------------
        # training
        # ------------
        # print("Args", self.args)
        # exit(1)
        # trainer = pl.Trainer.from_argparse_args(
        #     self.args,
        
        trainer = pl.Trainer(
            default_root_dir=self.args.log_dir,
            **TRAINER_ARGS,
            callbacks=[
                EarlyStopping(
                    monitor='valid_loss',
                    patience=10,
                    min_delta=0.00001),  # TODO!
                ModelCheckpoint(
                    filename='{epoch}-{val_loss:.2f}',
                    save_top_k=1,
                    verbose=False,
                    monitor='valid_loss',
                    mode='min')
                    #prefix=model.__class__.__name__)
            ])
        trainer.fit(model, train_loader, val_loader)

        # Save the best valid loss as this will go away after testing
        best_valid_loss = trainer.callback_metrics['valid_loss'].item()

        # ------------
        # testing
        # ------------
        trainer.test(dataloaders=test_loader, ckpt_path="best")

        # ------------
        # save results
        # ------------
        # Store predictions.
        ds = fluxdata.add_scalar_record(model.ds, varname='q10', x=model.q10_history)
        trial.set_user_attr('q10', ds.q10[-1].item())

        # Add some attributes that are required for analysis.
        ds.attrs = {
            'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'author': 'bkraft@bgc-jena.mpg.de',
            'q10_init': q10_init,
            'dropout': dropout,
            'use_ta': int(use_ta),
            'loss': best_valid_loss
        }

        ds = ds.isel(epoch=slice(0, trainer.current_epoch + 1))

        # Save data.
        save_dir = os.path.join(model.logger.log_dir, 'predictions.nc')
        print(f'Saving predictions to: {save_dir}')
        ds.to_netcdf(save_dir)

        return best_valid_loss

    @staticmethod
    def add_project_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--batch_size', default=240, type=int)
        parser.add_argument(
            '--data_path', default='./data/Synthetic4BookChap.nc', type=str)
        parser.add_argument(
            '--log_dir', default='./logs/20250430_synth0_kan_softplus/', type=str)
        return parser


def main(parser: ArgumentParser = None, **kwargs):
    """Use kwargs to overload argparse args."""

    # ------------
    # args
    # ------------
    if parser is None:
        parser = ArgumentParser()

    parser = Objective.add_project_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)  # TODO
    parser = Q10Model.add_model_specific_args(parser)
    parser.add_argument('--create_study', action='store_true', help='create new study (deletes old) and exits')
    # parser.add_argument('--single_seed', action='store_true', help='use only one seed instead of (1, ..., 10).')
    args = parser.parse_args()

    globargs = TRAINER_ARGS.copy()
    globargs.update(kwargs)

    for k, v in globargs.items():
        setattr(args, k, v)

    # # ------------
    # # study setup
    # # ------------
    # search_space = {
    #     'subset_frac': [1.0],
    #     'q10_init': [0.5],  # [0.5, 1.5, 2.5],
    #     'seed': [0] if args.single_seed else [i for i in range(10)],
    #     'dropout': [0.0],  # , 0.2, 0.4, 0.6],
    #     'use_ta': [True],  # [True, False]
    #     'lambda_param_violation': [1.0],  # [1.0],
    #     'lambda_kan_l1': [0.0],  # = trial.suggest_float('lambda_kan_l1', 0.0, 2.0)
    #     'lambda_kan_entropy': [0.1],  # = trial.suggest_float('lambda_kan_entropy', 0.0, 4.0)
    #     'lambda_kan_coefdiff': [0.0],  # = trial.suggest_float('lambda_kan_coefdiff', 0.0, 2.0)
    #     'lambda_kan_coefdiff2': [0.0],  # = trial.suggest_float('lambda_kan_coefdiff', 0.0, 2.0)
    #     'kan_grid': [3],  #trial.suggest_int('kan_grid', 3, 50)
    #     'kan_update_grid': [1],
    #     'kan_grid_margin': [1.0],  # = trial.suggest_float('kan_grid_margin', 0.0, 2.0)
    #     'kan_noise': [0.3],  # = trial.suggest_float('kan_noise', 0.1, 0.5)
    #     'kan_base_fun': ['silu_identity'],  # = trial.suggest_categorical('kan_base_fun', ['silu_identity', 'silu', 'identity'])
    #     'kan_affine_trainable': [True],  # = trial.suggest_categorical('kan_affine_trainable', [True, False])
    #     'lambda_jacobian_l1': [0.1],  #  [0.001, 0.01, 0.1, 1.0, 10.0]
    #     'lambda_jacobian_l05': [0],
    #     # 'lambda_robustness': [0],
    # }

    sql_file = os.path.abspath(os.path.join(args.log_dir, "optuna.db"))
    sql_path = f'sqlite:///{sql_file}'
    print("study name:", os.path.basename(args.log_dir))

    if args.create_study | (not os.path.isfile(sql_file)):
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir, exist_ok=True)
        study = optuna.create_study(
            study_name=os.path.basename(args.log_dir),
            storage=sql_path,
            sampler=optuna.samplers.GPSampler(seed=42),
            direction='minimize',
            load_if_exists=False)

        if args.create_study:
            return None

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # ------------
    # run study
    # ------------
    n_trials = 30
    study = optuna.load_study(
        study_name=os.path.basename(args.log_dir),
        storage=sql_path,
        sampler=optuna.samplers.GPSampler(seed=42))  # (search_space))
    study.optimize(Objective(args), n_trials=n_trials)


if __name__ == '__main__':
    main()
