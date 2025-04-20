"""Feedforward multilayer."""

import xarray as xr
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as torchf
import pytorch_lightning as pl

from utils.data_utils import Normalize

from models.feedforward import FeedForward, SENN_Simple, robustness_loss







class Q10Model(pl.LightningModule):
    def __init__(
            self,
            features: List[str],
            targets: List[str],
            norm: Normalize,
            ds: xr.Dataset,
            q10_init: int = 1.5,
            hidden_dim: int = 128,
            num_layers: int = 2,
            learning_rate: float = 0.01,
            weight_decay: float = 0.1,
            lambda_jacobian_l1: float = 0.0,
            lambda_jacobian_l05: float = 0.0,
            lambda_robustness: float = 0.0,
            lambda_out_of_range: float = 0.0,
            rb_constraint: str = 'softplus',
            dropout: float = 0.,
            activation: str = 'relu',  # 'tanh',
            num_steps: int = 0,
            model: str = 'nn') -> None:
        """Hybrid Q10 model.

        Note that restoring is not working currently as the model training is only taking
        some minutes.
        """

        super().__init__()
        self.validation_step_outputs = []  # Needed to change validation_epoch_end to on_validation_epoch_end: https://github.com/Lightning-AI/pytorch-lightning/pull/16520 
        self.save_hyperparameters(
            'features',
            'targets',
            'q10_init',
            'hidden_dim',
            'num_layers',
            'dropout',
            'activation',
            'learning_rate',
            'weight_decay',
            'lambda_jacobian_l1',
            'lambda_jacobian_l05',
            'lambda_robustness',
            'lambda_out_of_range',
            'rb_constraint'
        )

        self.features = features
        self.targets = targets

        self.q10_init = q10_init

        self.input_norm = norm.get_normalization_layer(variables=self.features, invert=False, stack=True)

        self.model = model
        if model == 'nn':
            self.nn = FeedForward(
                num_inputs=len(self.features),
                num_outputs=len(self.targets),
                num_hidden=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                dropout_last=False,
                activation=activation,
            )
        elif model == 'senn':
            self.nn = SENN_Simple(
                num_inputs=len(self.features),
                num_outputs=len(self.targets),
                num_hidden=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                dropout_last=False,
                activation=activation,
            )
        else:
            raise ValueError("Invalid model")
        print("MODEL", self.nn)

        self.target_norm = norm.get_normalization_layer(variables=self.targets, invert=False, stack=True)
        self.target_denorm = norm.get_normalization_layer(variables=self.targets, invert=True, stack=True)

        self.criterion = torch.nn.MSELoss()

        self.q10 = torch.nn.Parameter(torch.ones(1) * self.q10_init)
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.ta_ref = 15.0

        self.num_steps = num_steps

        # Used for strring results.
        self.ds = ds

        # Error if more than 100000 steps--ok here, but careful if you copy code for other projects!.
        self.q10_history = np.zeros(100000, dtype=np.float32) * np.nan


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Note that `x` is a dict of features and targets, input_norm extracts *only* features and stacks
        # them along last dimension.
        self.nn_input = self.input_norm(x)

        # Debugging: see if the label really is what it should be
        # import math
        # print(">>>>>>>> True label", x["reco"][0:5])
        # Rb_syn_tilde = 0.01 * x['sw_pot'] - 0.005 * x['dsw_pot']
        # Rb_syn = 0.75 * (Rb_syn_tilde - Rb_syn_tilde.min() + 0.1 * math.pi)
        # Rb_syn = 0.0075 * x['sw_pot'] - 0.00375 * x['dsw_pot'] + 1.023
        # Reco_syn = Rb_syn * (1.5 **(0.1 * (x['ta'] - 15)))
        # print("Reco_syn", Reco_syn[0:5])

        # Forward pass through NN.
        self.nn_input.requires_grad = True
        z = self.nn(self.nn_input)

        # No denormalization done currently.
        if self.hparams.rb_constraint == 'softplus':
            rb = torchf.softplus(z)
        elif self.hparams.rb_constraint == 'softplus_beta':
            rb = torchf.softplus(z, beta=self.beta)
        elif self.hparams.rb_constraint == 'relu':
            rb = torchf.relu(z)
        elif self.hparams.rb_constraint == 'none':
            rb = z
        else:
            raise ValueError("Invalid rb_constraint")
        self.rb = rb

        # Physical part.
        reco = rb * self.q10 ** (0.1 * (x['ta'] - self.ta_ref))

        return reco, rb, z

    def criterion_normed(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate criterion on normalized predictions and target."""
        return self.criterion(
            self.target_norm(y_hat),
            self.target_norm(y)
        )


    # ==================== Different approaches to compute gradient of Rb w.r.t inputs ==========================
    def predict_params_summed(self, func, input):
        """
        Helper function that takes in preprocessed NN input of shape [batch, n_input],
        passes it through nn and softplus, sums over batch dimension, then returns [n_output]
        """
        output = torchf.softplus(func(input))
        return output.sum(0)

    def get_jacobian_jacrev(self):
        """
        Returns Jacobian using jacrev, of shape [batch, n_output, n_input].
        For each batch item, it is dParam/dInput.

        Borrowed from PyTorch func tutorial / BINN code

        input should have shape [batch, n_input]
        """
        # Using jacrev, summing the outputs across batch as each example's output
        # only depends on that example's input
        self.nn.zero_grad()
        batch_jacobian1 = torch.func.jacrev(self.predict_params_summed, argnums=1)(self.nn, self.nn_input)  # [n_outputs, batch, n_inputs]
        batch_jacobian1 = batch_jacobian1.squeeze(0)  # [batch, n_inputs]
        return batch_jacobian1

    def get_jacobian_grad_precomputed(self):
        self.nn.zero_grad()
        dOutput_dInput = torch.autograd.grad(self.rb, self.nn_input, grad_outputs=torch.ones_like(self.rb), create_graph=True)
        return dOutput_dInput[0]

    def get_jacobian_grad(self):
        """
        Returns Jacobian using autograd. Assumes there is only one output.
        """

        self.nn_input.requires_grad = True
        self.nn.zero_grad()
        outputs = torchf.softplus(self.nn(self.nn_input))
        dOutput_dInput = torch.autograd.grad(outputs, self.nn_input, grad_outputs=torch.ones_like(outputs))
        return dOutput_dInput[0]

    def get_jacobian_finite_difference(self):
        """
        Returns Jacobian using finite difference approximation.
        """      
        eps = 0.01
        
        # TODO For-loop over each input feature is not efficient, try vectorizing
        # TODO Assuming there is only one output
        jacobian = torch.zeros_like(self.nn_input)
        orig_output = self.rb
        for feature_idx in range(self.nn_input.shape[1]):
            perturbed_input = self.nn_input.detach().clone()
            perturbed_input[:, feature_idx] += eps 
            perturbed_output = torchf.softplus(self.nn(perturbed_input))
            jacobian[:, feature_idx] = ((perturbed_output - orig_output) / eps).squeeze()
        return jacobian





    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Split batch (a dict) into actual data and the time-index returned by the dataset.
        if batch_idx == 1 and self.model == 'senn':
            print("COEFS", self.nn.coefs[0:5], self.nn.bias)

        batch, _ = batch

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, _, z = self(batch)

        # gradient of rb wrt inputs
        jacobian = self.get_jacobian_grad_precomputed()  # [batch, n_inputs]
        # jacobian2 = self.get_jacobian_grad()
        # jacobian3 = self.get_jacobian_jacrev()
        # jacobian = self.get_jacobian_finite_difference()
        # print("Lambda", self.hparams.lambda_jacobian_l1, "Jacobian", jacobian[0:5])
        # print("Jacobian", jacobian[0:10, :])
        # print("Weights", self.nn.model[0].weight)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Jacobian sparsity losses
        jacobian_l1_loss = jacobian.abs().mean()
        jacobian_l05_loss = l12_smooth(jacobian)
        jacobian_sparsity = (jacobian.abs() < 1e-5).float().mean()

        # Out-of-range loss
        out_of_range_loss = torchf.relu(-z).sum()

        # Save Q10 values, we want to know how they evolve with training,
        self.q10_history[self.global_step] = self.q10.item()

        # Logging.
        self.log('train_loss', loss, prog_bar=True)  #on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('out_of_range_loss', out_of_range_loss, prog_bar=True)
        # self.log('jacobian_l1', jacobian_l1_loss, prog_bar=True)  # on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('jacobian_l05', jacobian_l05_loss, prog_bar=True)
        # self.log('jacobian_sparsity', jacobian_sparsity, prog_bar=True)
        self.log('q10', self.q10, prog_bar=True)  # on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('beta', self.beta, prog_bar=True)
        total_loss = loss + self.hparams.lambda_out_of_range * out_of_range_loss
            #+ self.hparams.lambda_jacobian_l1 * jacobian_l1_loss + \
            #self.hparams.lambda_jacobian_l05 * jacobian_l05_loss

        # SENN robustness loss
        if self.model == "senn":
            r_loss = robustness_loss(self.nn_input, self.nn.predictions, self.nn.coefs)
            self.log('robustness_loss', r_loss, prog_bar=True)
            total_loss += self.hparams.lambda_robustness * r_loss
        return total_loss


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Predictions are stored in validation step. This is not best practice, but we are more interested
        # in predictions over training than on the final test predictions here.

        # Split batch (a dict) into actual data and the time-index returned by the dataset.
        batch, idx = batch

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, rb_hat, _ = self(batch)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Calculate loss on normalized data.
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # This dict is available in `on_validation_epoch_end` after saving to self.validation_step_outputs.
        outputs = {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}
        self.validation_step_outputs.append(outputs)
        return outputs


    def on_validation_epoch_end(self) -> None:
        # Iterate results from each validation step.
        for item in self.validation_step_outputs:
            reco_hat = item['reco_hat'][:, 0].cpu()
            rb_hat = item['rb_hat'][:, 0].cpu()
            idx = item['idx'].cpu()

            # Assign predictions to the right time steps.
            self.ds['reco_pred'].values[self.current_epoch, idx] = reco_hat
            self.ds['rb_pred'].values[self.current_epoch, idx] = rb_hat
        self.validation_step_outputs.clear()  # frree memory


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Evaluation on test set.
        batch, _ = batch
        reco_hat, _ = self(batch)
        loss = self.criterion_normed(reco_hat, batch['reco'])
        self.log('test_loss', loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = torch.optim.AdamW(
            [
                {
                    'params': self.nn.parameters(),
                    'weight_decay': self.hparams.weight_decay,
                    'learning_rate': self.hparams.learning_rate
                },
                {
                    'params': [self.q10, self.beta],
                    'weight_decay': 0.0,
                    'learning_rate': self.hparams.learning_rate * 10
                }
            ]
        )

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=16)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--c', type=float, default=0.1)
        parser.add_argument('--learning_rate', type=float, default=0.05)
        parser.add_argument('--weight_decay', type=float, default=0.1)
        parser.add_argument('--model', type=str, choices=['nn', 'senn'], default='nn')
        parser.add_argument('--rb_constraint', type=str, choices=['softplus', 'relu', 'none'], default='softplus',
                            help="""How to constrain Rb to be positive. softplus uses softplus function. relu uses
                                a relu function, and none imposes no constraint. With relu or none, we advise using
                                'lambda_out_of_range' to avoid vanishing gradients and softly encourage the constraint""")
        return parser




class Q10ModelSimple(pl.LightningModule):
    def __init__(
            self,
            features: List[str],
            targets: List[str],
            norm: Normalize,
            ds: xr.Dataset,
            q10_init: int = 1.5,
            hidden_dim: int = 128,
            num_layers: int = 2,
            learning_rate: float = 0.01,
            weight_decay: float = 0.1,
            lambda_jacobian_l1: float = 0.0,
            lambda_jacobian_l05: float = 0.0,
            lambda_robustness: float = 0.0,
            lambda_out_of_range: float = 0.0,
            dropout: float = 0.,
            activation: bool = 'relu',  # 'tanh',
            num_steps: int = 0,
            model = 'nn') -> None:
        """Hybrid Q10 model.

        Note that restoring is not working currently as the model training is only taking
        some minutes.
        """

        super().__init__()
        self.validation_step_outputs = []  # Needed to change validation_epoch_end to on_validation_epoch_end: https://github.com/Lightning-AI/pytorch-lightning/pull/16520 
        self.save_hyperparameters(
            'features',
            'targets',
            'q10_init',
            'hidden_dim',
            'num_layers',
            'dropout',
            'activation',
            'learning_rate',
            'weight_decay',
            'lambda_jacobian_l1',
            'lambda_jacobian_l05',
            'lambda_robustness',
            'lambda_out_of_range',
        )

        self.features = features
        self.targets = targets

        self.q10_init = q10_init

        self.input_norm = norm.get_normalization_layer(variables=self.features, invert=False, stack=True)

        self.model = model
        self.nn = nn.Linear(len(self.features), 1)
 
        self.target_norm = norm.get_normalization_layer(variables=self.targets, invert=False, stack=True)
        self.target_denorm = norm.get_normalization_layer(variables=self.targets, invert=True, stack=True)

        self.criterion = torch.nn.MSELoss()

        self.q10 = torch.nn.Parameter(torch.ones(1) * self.q10_init)
        self.ta_ref = 15.0

        self.num_steps = num_steps

        # Used for strring results.
        self.ds = ds

        # Error if more than 100000 steps--ok here, but careful if you copy code for other projects!.
        self.q10_history = np.zeros(100000, dtype=np.float32) * np.nan

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        # Note that `x` is a dict of features and targets, input_norm extracts *only* features and stacks
        # them along last dimension.
        self.nn_input = self.input_norm(x)

        # Forward pass through NN.
        self.nn_input.requires_grad = True
        self.rb = rb = self.nn(self.nn_input)

        # print("INPUT", self.nn_input[0:5])
        # print("SW POT", x['sw_pot'][0:5], "DSW_POT", x['dsw_pot'][0:5])
        # analytic_rb = 0.0075 * x['sw_pot'] - 0.00375 * x['dsw_pot'] + 1.023  # Unnormalized data
        # analytic_rb = 1.02 * self.nn_input[:, 0:1] - 1.81 * self.nn_input[:, 1:2] + 3.28
        # analytic_reco = analytic_rb * (1.5 ** (0.1 * (x['ta'] - self.ta_ref)))

        # Physical part.
        reco = rb * self.q10 ** (0.1 * (x['ta'] - self.ta_ref))


        # print(f"Rb: analytic {analytic_rb[0:5]}, predicted {self.rb[0:5]}")
        # print(f"Reco: predicted {reco[0:5]}, true {x['reco'][0:5]}, analytic {analytic_reco[0:5]}")
        # print("WEIGHTS", self.nn.weight.flatten(), self.nn.bias)

        return reco, rb

    def criterion_normed(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate criterion on normalized predictions and target."""
        return self.criterion(
            self.target_norm(y_hat),
            self.target_norm(y)
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Split batch (a dict) into actual data and the time-index returned by the dataset.
        batch, _ = batch

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, _ = self(batch)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Save Q10 values, we want to know how they evolve with training,
        self.q10_history[self.global_step] = self.q10.item()

        # Logging.
        self.log('train_loss', loss, prog_bar=True)  #on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('q10', self.q10, prog_bar=True)  # on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Predictions are stored in validation step. This is not best practice, but we are more interested
        # in predictions over training than on the final test predictions here.

        # Split batch (a dict) into actual data and the time-index returned by the dataset.
        batch, idx = batch

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, rb_hat = self(batch)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Calculate loss on normalized data.
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # This dict is available in `on_validation_epoch_end` after saving to self.validation_step_outputs.
        outputs = {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}
        self.validation_step_outputs.append(outputs)
        return outputs


    def on_validation_epoch_end(self) -> None:
        print("WEIGHTS", self.nn.weight.flatten(), self.nn.bias)

        # Iterate results from each validation step.
        for item in self.validation_step_outputs:
            reco_hat = item['reco_hat'][:, 0].cpu()
            rb_hat = item['rb_hat'][:, 0].cpu()
            idx = item['idx'].cpu()

            # Assign predictions to the right time steps.
            self.ds['reco_pred'].values[self.current_epoch, idx] = reco_hat
            self.ds['rb_pred'].values[self.current_epoch, idx] = rb_hat
        self.validation_step_outputs.clear()  # frree memory


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Evaluation on test set.
        batch, _ = batch
        reco_hat, _ = self(batch)
        loss = self.criterion_normed(reco_hat, batch['reco'])
        self.log('test_loss', loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = torch.optim.AdamW(
            [
                {
                    'params': self.nn.parameters(),
                    'weight_decay': self.hparams.weight_decay,
                    'learning_rate': self.hparams.learning_rate
                },
                {
                    'params': [self.q10],
                    'weight_decay': 0.0,
                    'learning_rate': self.hparams.learning_rate * 10
                }
            ]
        )

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=16)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--c', type=float, default=0.1)
        parser.add_argument('--learning_rate', type=float, default=0.05)
        parser.add_argument('--weight_decay', type=float, default=0.1)
        parser.add_argument('--model', type=str, choices=['nn', 'senn'], default='nn')
        return parser

"""Methods for regularization to produce sparse networks.

L2 regularization mostly penalizes the weight magnitudes without introducing sparsity.
L1 regularization promotes sparsity.
L1/2 promotes sparsity even more than L1. However, it can be difficult to train due to non-convexity and exploding
gradients close to 0. Thus, we introduce a smoothed L1/2 regularization to remove the exploding gradients.

Source: https://github.com/samuelkim314/DeepSymRegTorch/blob/main/utils/regularization.py"""

import torch
import torch.nn as nn


class L12Smooth(nn.Module):
    def __init__(self):
        super(L12Smooth, self).__init__()

    def forward(self, input_tensor, a=0.05):
        """input: predictions"""
        return l12_smooth(input_tensor, a)


def l12_smooth(input_tensor, a=0.05):
    """Smoothed L1/2 norm"""
    if type(input_tensor) == list:
        return sum([l12_smooth(tensor) for tensor in input_tensor])

    smooth_abs = torch.where(torch.abs(input_tensor) < a,
                             torch.pow(input_tensor, 4) / (-8 * a ** 3) + torch.square(input_tensor) * 3 / 4 / a + 3 * a / 8,
                             torch.abs(input_tensor))

    return torch.sum(torch.sqrt(smooth_abs))