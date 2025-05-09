"""Feedforward multilayer."""

import xarray as xr
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as torchf
import pytorch_lightning as pl

from utils.data_utils import Normalize

import math
import os
import sys
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from models.feedforward import FeedForward, SENN_Simple, robustness_loss

sys.path.append('../')
sys.path.append('../../')
import misc_utils
import visualization_utils
import kan 
from alibi.explainers import ALE, PartialDependenceVariance, plot_ale, plot_pd_variance




class Q10Model(pl.LightningModule):
    def __init__(
            self,
            features: List[str],
            targets: List[str],
            norm: Normalize,
            ds_train: xr.Dataset,
            ds_val: xr.Dataset,
            ds_test: xr.Dataset,
            q10_init: int = 1.5,
            hidden_dim: int = 128,
            num_layers: int = 2,
            learning_rate: float = 0.01,
            weight_decay: float = 0.1,
            lambda_jacobian_l1: float = 0.0,
            lambda_jacobian_l05: float = 0.0,
            # lambda_robustness: float = 0.0,
            lambda_param_violation: float = 0.0,
            lambda_kan_l1: float = 1.0,
            lambda_kan_entropy: float = 2.0,
            lambda_kan_node_entropy: float = 0.0,
            lambda_kan_coefdiff: float = 1.0,
            lambda_kan_coefdiff2: float = 1.0,
            kan_grid: int = 3,
            kan_grid_margin: int = 1.0,
            kan_update_grid: int = 1,
            kan_noise: int = 0.3,
            kan_base_fun: str = 'zero',
            kan_affine_trainable: bool = True, 
            rb_constraint: str = 'softplus',
            dropout: float = 0.,
            activation: str = 'relu',  # 'tanh',
            num_steps: int = 0,
            model: str = 'nn',
            true_relationships: np.array = None) -> None:
        """Hybrid Q10 model.

        Note that restoring is not working currently as the model training is only taking
        some minutes.
        """

        super().__init__()

        self.train_step_outputs = []
        self.validation_step_outputs = []  # Needed to change validation_epoch_end to on_validation_epoch_end: https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        self.test_step_outputs = []
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
            # 'lambda_robustness',
            'lambda_param_violation',
            'lambda_kan_l1',
            'lambda_kan_entropy',
            'lambda_kan_node_entropy',
            'lambda_kan_coefdiff',
            'lambda_kan_coefdiff2',
            'kan_grid',
            'kan_update_grid',
            'kan_grid_margin',
            'kan_noise',
            'kan_base_fun',
            'kan_affine_trainable',
            'rb_constraint'
        )

        if lambda_param_violation > 0:
            assert rb_constraint == "relu", "If out-of-range loss is set, rb_constraint must be relu"

        self.features = features
        self.targets = targets

        self.q10_init = q10_init

        self.input_norm = norm.get_normalization_layer(variables=self.features, invert=False, stack=True)

        self.model = model
        if model in ['nn', 'pure_nn']:
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
        elif model in ['kan', 'pure_kan']:
            layer_sizes = [len(self.features)] + [hidden_dim] * (num_layers - 1) + [len(self.targets)]
            
            # Version with mult nodes
            # layer_sizes = [len(self.features)] + [[hidden_dim // 2, hidden_dim // 2]] * (num_layers - 1) + [len(self.targets)]
            self.nn = kan.KAN(width=layer_sizes, grid=kan_grid, k=3, seed=42, device=self.device,  # residual=residual,
                               input_size=layer_sizes[0], noise_scale=kan_noise,
                               base_fun=kan_base_fun, affine_trainable=kan_affine_trainable, grid_eps=1.0, 
                               grid_margin=kan_grid_margin, drop_rate=dropout, drop_mode=('none' if dropout == 0.0 else 'postact'))
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

        # Used for storing results.
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.true_relationships = true_relationships

        # Error if more than 100000 steps--ok here, but careful if you copy code for other projects!.
        self.q10_history = np.zeros(100000, dtype=np.float32) * np.nan
        self.kan_update_grid = kan_update_grid


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Note that `x` is a dict of features and targets, input_norm extracts *only* features and stacks
        # them along last dimension.
        self.nn_input = self.input_norm(x)

        # print("After input norm. Means", self.nn_input.mean(dim=0), "Stds", self.nn_input.std(dim=0))  #, "Min", self.nn_input.min(dim=0), "Max", self.nn_input.max(dim=0))

        # # # Debugging: see if the label really is what it should be
        # import math
        # print(">>>>>>>> True label", x["reco"][0:5])
        # Rb_syn_tilde = 0.01 * x['sw_pot'] - 0.005 * x['dsw_pot']
        # Rb_syn = 0.75 * (Rb_syn_tilde - Rb_syn_tilde.min() + 0.1 * math.pi)
        # print("min Rb syn", Rb_syn_tilde.min())
        # Rb_syn = 0.0075 * x['sw_pot'] - 0.00375 * x['dsw_pot'] + 1.023
        # Reco_syn = Rb_syn * (1.5 **(0.1 * (x['ta'] - 15)))
        # print("Reco_syn", Reco_syn[0:5])

        # Forward pass through NN.
        self.nn_input.requires_grad = True
        z = self.nn(self.nn_input)

        # Pure NN or Pure KAN - only return predicted Reco (set predicted Rb = Reco)
        if self.model in ['pure_nn', 'pure_kan']:
            self.rb = self.target_denorm(z).unsqueeze(1)  # z had shape [batch, 1]. target_denorm removed the last dim, so add it back.
            return self.rb, self.rb, None

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
        if torch.isnan(reco).any():
            print("Reco contained nan", reco)
            print("Rb", rb)
            print("Q10", self.q10)
            print("ta_ref", self.ta_ref)

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
        if self.hparams.rb_constraint == "softplus":
            output = torchf.softplus(func(input))
        elif self.hparams.rb_constraint == "relu":
            output = torchf.relu(func(input))
        elif self.hparams.rb_constraint == "none":
            output = func(input)
        else:
            raise ValueError("invalid rb_constraint")
        return output.sum(0)

    def get_jacobian_jacrev(self, input):
        """
        Returns Jacobian using jacrev, of shape [batch, n_output, n_input].
        For each batch item, it is dParam/dInput.

        Borrowed from PyTorch func tutorial / BINN code

        input should have shape [batch, n_input]
        """
        if input is None:
            input = self.nn_input

        # Using jacrev, summing the outputs across batch as each example's output
        # only depends on that example's input
        self.nn.zero_grad()
        batch_jacobian1 = torch.func.jacrev(self.predict_params_summed, argnums=1)(self.nn, input)  # [n_outputs, batch, n_inputs]
        batch_jacobian1 = batch_jacobian1.permute((1, 0, 2))  # [batch, n_output, n_input]  # NOTE Previously .squeeze(0)  # [batch, n_inputs]
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
        # # Split batch (a dict) into actual data and the time-index returned by the dataset.
        # if batch_idx == 1 and self.model == 'senn':
        #     print("COEFS", self.nn.coefs[0:5], self.nn.bias)

        batch, idx = batch

        # print("TRAIN STEP: Unnormalized", [f"{var} mean {x.mean()} std {x.std()}" for var, x in batch.items()])
        # print("TRAIN STEP: Normalized means", self.input_norm(batch).mean(dim=0), "stds", self.input_norm(batch).std(dim=0))

        # Update grid for KAN if desired
        if self.model == 'kan' and self.kan_update_grid == 1 and batch_idx == 1 and self.current_epoch < 20:
            self.nn.update_grid(self.input_norm(batch))
            print('Updated grid to', self.nn.act_fun[1].grid)

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, rb_hat, z = self(batch)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Save Q10 values, we want to know how they evolve with training,
        self.q10_history[self.global_step] = self.q10.item()

        # Logging.
        self.log('train_loss', loss, prog_bar=True)  #on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('q10', self.q10, prog_bar=True)  # on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # self.log('beta', self.beta, prog_bar=True)
        total_loss = loss

        # Out-of-range loss
        if z is not None:
            param_violation_loss = torchf.relu(-z).sum()
            if self.hparams.rb_constraint == "relu":
                self.log('param_violation_loss', param_violation_loss, prog_bar=True)
                total_loss += (self.hparams.lambda_param_violation * param_violation_loss)
        else:
            assert self.model in ["pure_nn", "pure_kan"], "If unconstrained Rb is None, model must be pure_nn or pure_kan"
            assert self.hparams.lambda_param_violation == 0, "If unconstrained Rb is None, lambda_param_violation must be 0"

        # Calculate Jacobian (derivative of Rb wrt inputs)
        jacobian = self.get_jacobian_grad_precomputed()  # [batch, n_inputs]
        # jacobian2 = self.get_jacobian_grad()
        # jacobian3 = self.get_jacobian_jacrev()
        # jacobian = self.get_jacobian_finite_difference()
        # print("Lambda", self.hparams.lambda_jacobian_l1, "Jacobian", jacobian[0:5])
        # print("Jacobian", jacobian[0:10, :])
        # print("Weights", self.nn.model[0].weight)

        # Losses related to Jacobian
        jacobian_l1_loss = jacobian.abs().mean()
        jacobian_l05_loss = l12_smooth(jacobian)
        jacobian_sparsity = (jacobian.abs() < 1e-5).float().mean()
        self.log('jacobian_sparsity', jacobian_sparsity, prog_bar=True)
        if self.hparams.lambda_jacobian_l1 > 0:
            self.log('jacobian_l1', jacobian_l1_loss, prog_bar=True)  # on_step=False, on_epoch=True, prog_bar=False, logger=True)
            total_loss += (self.hparams.lambda_jacobian_l1 * jacobian_l1_loss)
        if self.hparams.lambda_jacobian_l05 > 0:
            self.log('jacobian_l05', jacobian_l05_loss, prog_bar=True)
            total_loss += (self.hparams.lambda_jacobian_05 * jacobian_l05_loss)

        # KAN-related losses
        if self.model in ["kan", "pure_kan"]:
            # NOTE: the lamb values passed are completely unused, as we directly obtain the individual loss components and weight them later.
            # For default weights see https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py#L1411
            kan_l1_loss, kan_entropy_loss, kan_coef_loss, kan_coefdiff_loss, kan_coefdiff2_loss = self.nn.reg(reg_metric='edge_backward', lamb_l1=1., lamb_entropy=1., lamb_coef=1., lamb_coefdiff=1., return_indiv=True)
            _, kan_node_entropy_loss, _, _, _ = self.nn.reg(reg_metric='node_influence_on_output', lamb_l1=0, lamb_entropy=1., lamb_coef=0., lamb_coefdiff=0., return_indiv=True)
            if self.hparams.lambda_kan_l1 > 0:
                self.log('kan_l1_loss', kan_l1_loss, prog_bar=True)
                # new_lambda = self.hparams.lambda_kan_l1  # 0 if self.current_epoch < 20 else self.hparams.lambda_kan_l1  # max(0, self.hparams.lambda_kan_l1 * (self.current_epoch - 20))
                # total_loss += (new_lambda * kan_l1_loss)
                total_loss += (self.hparams.lambda_kan_l1 * kan_l1_loss)
            if self.hparams.lambda_kan_entropy > 0:
                self.log('kan_entropy_loss', kan_entropy_loss, prog_bar=True)
                # new_lambda = self.hparams.lambda_kan_entropy  # 0 if self.current_epoch < 20 else self.hparams.lambda_kan_entropy   #max(0, self.hparams.lambda_kan_entropy * (self.current_epoch - 20))
                # total_loss += (new_lambda * kan_entropy_loss)
                total_loss += (self.hparams.lambda_kan_entropy * kan_entropy_loss)
            if self.hparams.lambda_kan_node_entropy > 0:
                self.log('lambda_kan_node_entropy', kan_node_entropy_loss, prog_bar=True)
                # new_lambda = self.hparams.lambda_kan_node_entropy
                # total_loss += (new_lambda * kan_node_entropy_loss)
                total_loss += (self.hparams.lambda_kan_node_entropy * kan_node_entropy_loss)
            if self.hparams.lambda_kan_coefdiff > 0:
                self.log('kan_coefdiff_loss', kan_coefdiff_loss, prog_bar=True)
                # new_lambda = self.hparams.lambda_kan_coefdiff  # 0 if self.current_epoch < 20 else self.hparams.lambda_kan_coefdiff   # max(0, self.hparams.lambda_kan_coefdiff * (self.current_epoch - 20))
                # total_loss += (new_lambda * kan_coefdiff_loss)
                total_loss += (self.hparams.lambda_kan_coefdiff * kan_coefdiff_loss)            
            if self.hparams.lambda_kan_coefdiff2 > 0:
                self.log('kan_coefdiff2_loss', kan_coefdiff2_loss, prog_bar=True)
                # new_lambda = self.hparams.lambda_kan_coefdiff2  # 0 if self.current_epoch < 20 else self.hparams.lambda_kan_coefdiff2   # max(0, self.hparams.lambda_kan_coefdiff2 * (self.current_epoch - 20))
                # total_loss += (new_lambda * kan_coefdiff2_loss)
                total_loss += (self.hparams.lambda_kan_coefdiff2 * kan_coefdiff2_loss)

        # # SENN robustness loss
        # if self.model == "senn":
        #     r_loss = robustness_loss(self.nn_input, self.nn.predictions, self.nn.coefs)
        #     self.log('robustness_loss', r_loss, prog_bar=True)
        #     total_loss += self.hparams.lambda_robustness * r_loss

        # This dict is available in `on_train_epoch_end` after saving to self.train_step_outputs.
        outputs = {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}
        self.train_step_outputs.append(outputs)
        return total_loss



    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Predictions are stored in validation step. This is not best practice, but we are more interested
        # in predictions over training than on the final test predictions here.

        # Split batch (a dict) into actual data and the time-index returned by the dataset.
        batch, idx = batch

        # print("VALIDATION STEP: Unnormalized", [f"{var} mean {x.mean()} std {x.std()}" for var, x in batch.items()])
        # print("VALIDATION STEP: Normalized means", self.input_norm(batch).mean(dim=0), "stds", self.input_norm(batch).std(dim=0))

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, rb_hat, _ = self(batch)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Calculate loss on normalized data.
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # This dict is available in `on_validation_epoch_end` after saving to self.validation_step_outputs.
        outputs = {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}
        self.validation_step_outputs.append(outputs)
        return outputs


    # Special predictor function to use NN
    @torch.no_grad()
    def predictor(self, X: np.ndarray) -> np.ndarray:
        assert self.nn.training == False, "Model must be in eval mode"
        X = torch.as_tensor(X, device=self.device)
        return self.nn.forward(X).cpu().numpy()


    def on_train_epoch_end(self) -> None:
        """
        Plot true vs predicted values for training set. This helps assess the degree of overfitting.
        """
        # Iterate results from each train step.
        for item in self.train_step_outputs:
            reco_hat = item['reco_hat'][:, 0].detach().cpu()
            rb_hat = item['rb_hat'][:, 0].detach().cpu()
            idx = item['idx'].detach().cpu()

            # Assign predictions to the right time steps.
            self.ds_train['reco_pred'].values[self.current_epoch, idx] = reco_hat
            self.ds_train['rb_pred'].values[self.current_epoch, idx] = rb_hat

        if self.current_epoch % 10 == 0:
            # True vs predicted scatters
            print("Plotting to ", self.logger.log_dir)
            y_hats = [self.ds_train['reco_pred'].values[self.current_epoch, :], self.ds_train['rb_pred'].values[self.current_epoch, :]]
            ys = [self.ds_train['reco'].values, self.ds_train['rb'].values]
            titles = ['R_eco (labeled)', 'R_b (latent)']
            visualization_utils.plot_true_vs_predicted_multiple(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_true_vs_predicted_train.png"),
                                                                y_hats=y_hats, ys=ys, titles=titles)
        self.train_step_outputs.clear()  # free memory
        print("\n")


    def on_validation_epoch_end(self) -> None:
        print("VALID EPOCH END", self.current_epoch)
        # print("GRID", self.nn.act_fun[1].grid)
        # print("COEF", self.nn.act_fun[1].coef)
        # print("Trainer", self.trainer.max_epochs, self.trainer.should_stop)
        # print("2nd layer grid", self.nn.act_fun[1].grid)
        # print("Inputs to 2nd layer", self.nn.acts[1][0:10])
        print("ACTFUN MASK", self.nn.act_fun[1].mask)
        print("SYMBOLIC MASK", self.nn.symbolic_fun[1].mask)
        print("Scale base", self.nn.act_fun[1].scale_base, self.nn.act_fun[1].scale_sp)
        print("Coef", self.nn.act_fun[1].coef)
        print("Grid", self.nn.act_fun[1].grid)


        # print("Node bias", self.nn.node_bias[1], self.nn.node_scale[1])

        # Iterate results from each validation step.
        for item in self.validation_step_outputs:
            reco_hat = item['reco_hat'][:, 0].cpu()
            rb_hat = item['rb_hat'][:, 0].cpu()
            idx = item['idx'].cpu()

            # Assign predictions to the right time steps.
            self.ds_val['reco_pred'].values[self.current_epoch, idx] = reco_hat
            self.ds_val['rb_pred'].values[self.current_epoch, idx] = rb_hat

        print("ON VAL END - Rb", self.ds_val['rb_pred'].mean(), self.ds_val['rb_pred'].std())

        if self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs or self.trainer.should_stop:
            if self.current_epoch == self.trainer.max_epochs or self.trainer.should_stop:
                print("Early stopped", self.current_epoch)
                epoch_str = "FINAL"
            else:
                epoch_str = f"epoch{self.current_epoch}"

            # True vs predicted scatters
            y_hats = [self.ds_val['reco_pred'].values[self.current_epoch, :], self.ds_val['rb_pred'].values[self.current_epoch, :]]
            ys = [self.ds_val['reco'].values, self.ds_val['rb'].values]
            titles = ['R_eco (labeled)', 'R_b (latent)']
            visualization_utils.plot_true_vs_predicted_multiple(os.path.join(self.logger.log_dir, f"{epoch_str}_true_vs_predicted_val.png"),
                                                                y_hats=y_hats, ys=ys, titles=titles)
            
            # # normed version of reco
            # y_hats = [self.target_norm(torch.tensor(self.ds_val['reco_pred'].values[self.current_epoch, :]).unsqueeze(1))]
            # ys = [self.target_norm(torch.tensor(self.ds_val['reco'].values).unsqueeze(1))]
            # titles = ['R_eco (NORMALIZED labeled)']
            # visualization_utils.plot_true_vs_predicted_multiple(os.path.join(self.logger.log_dir, f"{epoch_str}_true_vs_predicted_val_NORMALIZED.png"),
            #                                                     y_hats=y_hats, ys=ys, titles=titles)

            # KAN plot. Note this should go before other plots, since it relies on cached values from the last batch passed through the model
            # (the below plots pass other non-representative data through the model)
            if self.model in ["kan", "pure_kan"]:  #  and (self.current_epoch % 10 == 0 or epoch_str == "FINAL"):
                out_vars = ["Rb"] if self.model == "kan" else ["Reco"]
                self.nn.to(self.device)
                self.nn.attribute()
                self.nn.node_attribute()

                # # Plot the pruned model
                # pruned_model = self.nn.prune()  # node_th=0.03, edge_th=0.03)
                # # pruned_model.auto_swap()  # swap neurons to make it simpler? 
                # pruned_model.plot(folder=os.path.join(self.logger.log_dir, "splines_pruned"), in_vars=self.features, out_vars=out_vars)  #, scale=5, varscale=0.15)
                # plt.savefig(os.path.join(self.logger.log_dir, f"{epoch_str}_kan_plot_pruned.png"))
                # plt.close()

                # Plot the unpruned model
                self.nn.plot(folder=os.path.join(self.logger.log_dir, "splines"), in_vars=self.features, out_vars=out_vars)  #, scale=5, varscale=0.15)
                plt.savefig(os.path.join(self.logger.log_dir, f"{epoch_str}_kan_plot.png"))
                plt.close()

            # print("True impt", self.true_relationships)
            # print("Pred jac impt", jacobian_importances)
            # print("PRed pdv impt", pdv_importances)

            # # If functional relationships are known, compare KAN's predicted relationships with ground-truth relationships
            # # Save picture of functional relationships. Source: https://stackoverflow.com/questions/69986007/matplotlib-imshow-with-1-color-for-each-discrete-value
            # fig, axeslist = plt.subplots(1, len(predicted_importances_all)+1, figsize=(6*(len(predicted_importances_all)+1), 6))
            # cmap = plt.get_cmap('viridis')
            # for pred_idx, (pred_method, pred_rel) in enumerate(predicted_importances_all.items()):
            #     relationship_kl = misc_utils.kl_divergence(self.true_relationships, pred_rel)
            #     relationship_l2 = math.sqrt(((self.true_relationships - pred_rel) ** 2).sum())
            #     print("REl metrics", relationship_kl, relationship_l2)
            #     im = axeslist[pred_idx].imshow(pred_rel, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
            #     axeslist[pred_idx].set_xticks(np.arange(len(self.targets)))
            #     axeslist[pred_idx].set_yticks(np.arange(len(self.features)))
            #     axeslist[pred_idx].set_xticklabels(self.targets, rotation='vertical')
            #     axeslist[pred_idx].set_yticklabels(self.features)
            #     axeslist[pred_idx].set_title(f"Predicted by {pred_method}\n(KL: {relationship_kl:.3f}, Euclidean dist: {relationship_l2:.3f})")
            # im = axeslist[-1].imshow(self.true_relationships, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
            # axeslist[-1].set_xticks(np.arange(len(self.targets)))
            # axeslist[-1].set_yticks(np.arange(len(self.features)))
            # axeslist[-1].set_xticklabels(self.targets, rotation='vertical')
            # axeslist[-1].set_yticklabels(self.features)
            # axeslist[-1].set_title("Ground-truth")
            # fig.colorbar(im, orientation="vertical", ax=axeslist[-1])
            # plt.tight_layout()
            # plt.savefig(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_functional_relationships.png"))
            # plt.close()

            # if self.model == "nn":
            #     # Plot PDP variance
            #     plot_pd_variance(exp=exp_importance)
            #     plt.savefig(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_pd_variance.png"))
            #     plt.close()

            #     # Accumulated Local Effects plot (similar to partial dependence plot
            #     # but better with correlated features)
            #     ale = ALE(self.predictor, feature_names=self.features, target_names=self.targets)
            #     exp = ale.explain(cached_nn_input.detach().cpu().numpy())
            #     plot_ale(exp)
            #     plt.savefig(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_ale.png"))
            #     plt.close()

        # Metrics to save
        valid_reco_r2, valid_reco_mse, valid_reco_mae, valid_reco_corr = misc_utils.compute_metrics(self.ds_val['reco'].values, self.ds_val['reco_pred'].values[self.current_epoch, :])
        valid_rb_r2, valid_rb_mse, valid_rb_mae, valid_rb_corr = misc_utils.compute_metrics(self.ds_val['rb'].values, self.ds_val['rb_pred'].values[self.current_epoch, :])
        self.log('valid_reco_r2', valid_reco_r2)
        self.log('valid_reco_mse', valid_reco_mse)
        self.log('valid_reco_mae', valid_reco_mae)
        self.log('valid_reco_corr', valid_reco_corr)
        self.log('valid_rb_r2', valid_rb_r2)
        self.log('valid_rb_mse', valid_rb_mse)
        self.log('valid_rb_mae', valid_rb_mae)
        self.log('valid_rb_corr', valid_rb_corr)
        self.validation_step_outputs.clear()  # free memory
        print("\n")


    # def on_validation_end(self) -> None:
    #     print("on_validation_end", self.current_epoch)

    #     # Iterate results from each validation step.
    #     for item in self.validation_step_outputs:
    #         reco_hat = item['reco_hat'][:, 0].cpu()
    #         rb_hat = item['rb_hat'][:, 0].cpu()
    #         idx = item['idx'].cpu()

    #         # Assign predictions to the right time steps.
    #         self.ds_val['reco_pred'].values[self.current_epoch, idx] = reco_hat
    #         self.ds_val['rb_pred'].values[self.current_epoch, idx] = rb_hat

    #     print("ON VALIDATION END - Rb", self.ds_val['rb_pred'].mean(), self.ds_val['rb_pred'].std())
    #     epoch_str = "FINAL"

    #     # True vs predicted scatters
    #     print("Plotting to ", self.logger.log_dir)
    #     y_hats = [self.ds_val['reco_pred'].values[self.current_epoch, :], self.ds_val['rb_pred'].values[self.current_epoch, :]]
    #     ys = [self.ds_val['reco'].values, self.ds_val['rb'].values]
    #     titles = ['R_eco (labeled)', 'R_b (latent)']
    #     visualization_utils.plot_true_vs_predicted_multiple(os.path.join(self.logger.log_dir, f"{epoch_str}_true_vs_predicted_val.png"),
    #                                                         y_hats=y_hats, ys=ys, titles=titles)
        
    #     # normed version of reco
    #     y_hats = [self.target_norm(torch.tensor(self.ds_val['reco_pred'].values[self.current_epoch, :]).unsqueeze(1))]
    #     ys = [self.target_norm(torch.tensor(self.ds_val['reco'].values).unsqueeze(1))]
    #     titles = ['R_eco (NORMALIZED labeled)']
    #     visualization_utils.plot_true_vs_predicted_multiple(os.path.join(self.logger.log_dir, f"{epoch_str}_true_vs_predicted_val_NORMALIZED.png"),
    #                                                         y_hats=y_hats, ys=ys, titles=titles)

    #     # KAN plot. Note this should go before other plots, since it relies on cached values from the last batch passed through the model
    #     # (the below plots pass other non-representative data through the model)
    #     if self.model in ["kan", "pure_kan"]:
    #         out_vars = ["Rb"] if self.model == "kan" else ["Reco"]
    #         self.nn.to(self.device)
    #         self.nn.attribute()
    #         self.nn.node_attribute()

    #         # Plot the pruned model
    #         pruned_model = self.nn.prune()  # node_th=0.03, edge_th=0.03)
    #         # pruned_model.auto_swap()  # swap neurons to make it simpler? 
    #         pruned_model.plot(folder=os.path.join(self.logger.log_dir, "splines_pruned"), in_vars=self.features, out_vars=out_vars)  #, scale=5, varscale=0.15)
    #         plt.savefig(os.path.join(self.logger.log_dir, f"FINAL_kan_plot_pruned.png"))
    #         plt.close()

    #         # Plot the unpruned model
    #         self.nn.plot(folder=os.path.join(self.logger.log_dir, "splines"), in_vars=self.features, out_vars=out_vars)  #, scale=5, varscale=0.15)
    #         plt.savefig(os.path.join(self.logger.log_dir, f"FINAL_kan_plot.png"))
    #         plt.close()

    #     self.validation_step_outputs.clear()  # free memory
    #     print("\n")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Evaluation on test set.
        batch, idx = batch
        reco_hat, rb_hat, _ = self(batch)
        loss = self.criterion_normed(reco_hat, batch['reco'])
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        outputs = {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}
        self.test_step_outputs.append(outputs)
        return outputs


    # # Need to enable grad to compute Jacobian feature importance
    # # https://github.com/Lightning-AI/pytorch-lightning/issues/201
    @torch.inference_mode(False)
    @torch.enable_grad()
    def on_test_epoch_end(self) -> None:
        # Iterate results from each epoch step.
        if self.model == "kan":
            # print("GRID FINAL TEST", self.nn.act_fun[1].grid)
            # print("COEF", self.nn.act_fun[1].coef)
            # print("TEST Node bias", self.nn.node_bias[1], self.nn.node_scale[1])
            print("TEST TEST) 2nd layer grid", self.nn.act_fun[1].grid)
            print("Inputs to 2nd layer", self.nn.acts[1][0:10])

        with torch.no_grad():
            for item in self.test_step_outputs:
                reco_hat = item['reco_hat'][:, 0].cpu()
                rb_hat = item['rb_hat'][:, 0].cpu()
                idx = item['idx'].cpu()

                # Assign predictions to the right time steps.
                # For some reason, self.current_epoch is equal to max_epochs (1 higher than it
                # should be). Reduce by 1 to fit into ds_test.
                self.ds_test['reco_pred'].values[self.current_epoch, idx] = reco_hat
                self.ds_test['rb_pred'].values[self.current_epoch, idx] = rb_hat

        print("ON TEST END - Rb", self.ds_test['rb_pred'].mean(), self.ds_test['rb_pred'].std())

        # Store nn_input in case the following methods change it
        cached_nn_input = self.nn_input.clone()

        # Compute feature importances
        # Feature importance by Jacobian
        jacobian = self.get_jacobian_jacrev(cached_nn_input)  # [batch, n_params, n_inputs]
        avg_jacobian_magnitude = jacobian.abs().mean(dim=0).detach().cpu().numpy().T  # transpose to [n_inputs, n_params]
        jacobian_importances = avg_jacobian_magnitude / avg_jacobian_magnitude.sum(axis=0, keepdims=True)

        # Feature importance by Partial Dependence Variance
        with torch.no_grad():            
            pd_variance = PartialDependenceVariance(predictor=self.predictor,
                                    feature_names=self.features,
                                    target_names=self.targets)
            exp_importance = pd_variance.explain(cached_nn_input.detach().cpu().numpy(), method='importance')
            importance_scores = exp_importance.data['feature_importance'].T  # transpose to [n_inputs, n_params]
            pdv_importances = importance_scores / importance_scores.sum(axis=0, keepdims=True)
            for feat_idx, feat in enumerate(self.features):
                self.log(f'{feat}_importance', pdv_importances.flatten()[feat_idx], prog_bar=False, logger=True)
            predicted_importances_all = {"Jacobian": jacobian_importances,
                                        "Partial Dependence Variance": pdv_importances}
            if self.model == "kan" and self.hparams.num_layers == 1:
                # If using one-layer KAN, read off functional relationships with mask
                kan_importances = self.nn.edge_scores[0].permute(1, 0).detach().cpu().numpy()
                print("KAN impts", kan_importances)
                predicted_importances_all["KAN"] = kan_importances
            DEFAULT_REL = "KAN" if "KAN" in predicted_importances_all else "Partial Dependence Variance"
            default_kl = misc_utils.kl_divergence(self.true_relationships, predicted_importances_all[DEFAULT_REL])
            default_l2 = math.sqrt(((self.true_relationships - predicted_importances_all[DEFAULT_REL]) ** 2).sum())

            # If functional relationships are known, compare KAN's predicted relationships with ground-truth relationships
            # Save picture of functional relationships. Source: https://stackoverflow.com/questions/69986007/matplotlib-imshow-with-1-color-for-each-discrete-value
            fig, axeslist = plt.subplots(1, len(predicted_importances_all)+1, figsize=(6*(len(predicted_importances_all)+1), 6))
            cmap = plt.get_cmap('Greens')
            for pred_idx, (pred_method, pred_rel) in enumerate(predicted_importances_all.items()):
                relationship_kl = misc_utils.kl_divergence(self.true_relationships, pred_rel)
                relationship_l2 = math.sqrt(((self.true_relationships - pred_rel) ** 2).sum())
                im = axeslist[pred_idx].imshow(pred_rel, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
                axeslist[pred_idx].set_xticks(np.arange(len(self.targets)))
                axeslist[pred_idx].set_yticks(np.arange(len(self.features)))
                axeslist[pred_idx].set_xticklabels(self.targets, rotation='vertical')
                axeslist[pred_idx].set_yticklabels(self.features)
                axeslist[pred_idx].set_title(f"Predicted by {pred_method}\n(KL: {relationship_kl:.3f}, Euclidean dist: {relationship_l2:.3f})")
            im = axeslist[-1].imshow(self.true_relationships, cmap=cmap, vmin=0, vmax=1)  #, vmin=-0.5, vmax=5.5, cmap=cmap, interpolation="none")
            axeslist[-1].set_xticks(np.arange(len(self.targets)))
            axeslist[-1].set_yticks(np.arange(len(self.features)))
            axeslist[-1].set_xticklabels(self.targets, rotation='vertical')
            axeslist[-1].set_yticklabels(self.features)
            axeslist[-1].set_title("Ground-truth")
            fig.colorbar(im, orientation="vertical", ax=axeslist[-1])
            plt.tight_layout()
            plt.savefig(os.path.join(self.logger.log_dir, f"FINAL_functional_relationships.png"))
            plt.close()

            if self.model == "nn":
                # Plot PDP variance
                plot_pd_variance(exp=exp_importance)
                plt.savefig(os.path.join(self.logger.log_dir, f"FINAL_pd_variance.png"))
                plt.close()

                # Accumulated Local Effects plot (similar to partial dependence plot
                # but better with correlated features)
                ale = ALE(self.predictor, feature_names=self.features, target_names=self.targets)
                exp = ale.explain(cached_nn_input.detach().cpu().numpy())
                plot_ale(exp)
                plt.savefig(os.path.join(self.logger.log_dir, f"FINAL_ale.png"))
                plt.close()

            # True vs predicted scatters
            print("Plotting to ", self.logger.log_dir)
            y_hats = [self.ds_test['reco_pred'].values[self.current_epoch, :], self.ds_test['rb_pred'].values[self.current_epoch, :]]
            ys = [self.ds_test['reco'].values, self.ds_test['rb'].values]
            titles = ['R_eco (labeled)', 'R_b (latent)']
            visualization_utils.plot_true_vs_predicted_multiple(os.path.join(self.logger.log_dir, f"FINAL_true_vs_predicted_test.png"),
                                                                y_hats=y_hats, ys=ys, titles=titles)

            # # normed version of reco
            # y_hats = [self.target_norm(torch.tensor(self.ds_test['reco_pred'].values[self.current_epoch, :]).unsqueeze(1))]
            # ys = [self.target_norm(torch.tensor(self.ds_test['reco'].values).unsqueeze(1))]
            # titles = ['R_eco (NORMALIZED labeled)']
            # visualization_utils.plot_true_vs_predicted_multiple(os.path.join(self.logger.log_dir, f"FINAL_true_vs_predicted_test_NORMALIZED.png"),
            #                                                     y_hats=y_hats, ys=ys, titles=titles)

            # KAN plots
            # (the below plots pass other non-representative data through the model)
            if self.model in ["kan", "pure_kan"]:
                out_vars = ["Rb"] if self.model == "kan" else ["Reco"]
                self.nn.to(self.device)
                self.nn.cache_data = cached_nn_input
                self.nn.attribute()
                self.nn.node_attribute()

                # Plot the pruned model
                pruned_model = self.nn.prune()  # node_th=0.03, edge_th=0.03)
                # pruned_model.auto_swap()  # swap neurons to make it simpler? 
                pruned_model.plot(folder=os.path.join(self.logger.log_dir, "splines_pruned"), in_vars=self.features, out_vars=out_vars)  #, scale=5, varscale=0.15)
                plt.savefig(os.path.join(self.logger.log_dir, f"FINAL_kan_plot_pruned.png"))
                plt.close()

                # Plot the unpruned model
                self.nn.plot(folder=os.path.join(self.logger.log_dir, "splines"), in_vars=self.features, out_vars=out_vars)  #, scale=5, varscale=0.15)
                plt.savefig(os.path.join(self.logger.log_dir, f"FINAL_kan_plot.png"))
                plt.close()

            # Metrics to save
            test_reco_r2, test_reco_mse, test_reco_mae, test_reco_corr = misc_utils.compute_metrics(self.ds_test['reco'].values, self.ds_test['reco_pred'].values[self.current_epoch, :])
            test_rb_r2, test_rb_mse, test_rb_mae, test_rb_corr = misc_utils.compute_metrics(self.ds_test['rb'].values, self.ds_test['rb_pred'].values[self.current_epoch, :])
            self.log('test_reco_r2', test_reco_r2)
            self.log('test_reco_mse', test_reco_mse)
            self.log('test_reco_mae', test_reco_mae)
            self.log('test_reco_corr', test_reco_corr)
            self.log('test_rb_r2', test_rb_r2)
            self.log('test_rb_mse', test_rb_mse)
            self.log('test_rb_mae', test_rb_mae)
            self.log('test_rb_corr', test_rb_corr)
            self.log('relationship_kl', default_kl)
            self.log('relationship_l2', default_l2)
            self.test_step_outputs.clear()  # free memory

            # Plot q10 history
            non_nan_idx = np.flatnonzero(~np.isnan(self.q10_history))
            plt.plot(non_nan_idx, self.q10_history[non_nan_idx])
            plt.xlabel("Step number")
            plt.ylabel("Predicted Q10")
            plt.title("Q10 predictions throughout training")
            plt.savefig(os.path.join(self.logger.log_dir, "predicted_q10s.png"))
            plt.close()


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
                    'learning_rate': self.hparams.learning_rate * 100  # 10
                }
            ]
        )

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=16)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--kan_grid', type=int, default=30)
        parser.add_argument('--c', type=float, default=0.1)
        # parser.add_argument('--learning_rate', type=float, default=0.05)
        # parser.add_argument('--weight_decay', type=float, default=0.1)
        parser.add_argument('--model', type=str, choices=['nn', 'senn', 'kan', 'pure_nn', 'pure_kan'], default='nn')
        parser.add_argument('--rb_constraint', type=str, choices=['softplus', 'relu', 'none'], default='softplus',
                            help="""How to constrain Rb to be positive. softplus uses softplus function. relu uses
                                a relu function, and none imposes no constraint. With relu or none, we advise using
                                'lambda_param_violation' to avoid vanishing gradients and softly encourage the constraint""")
        return parser




# class Q10ModelSimple(pl.LightningModule):
#     def __init__(
#             self,
#             features: List[str],
#             targets: List[str],
#             norm: Normalize,
#             ds: xr.Dataset,
#             q10_init: int = 1.5,
#             hidden_dim: int = 128,
#             num_layers: int = 2,
#             learning_rate: float = 0.01,
#             weight_decay: float = 0.1,
#             lambda_jacobian_l1: float = 0.0,
#             lambda_jacobian_l05: float = 0.0,
#             lambda_robustness: float = 0.0,
#             lambda_param_violation: float = 0.0,
#             dropout: float = 0.,
#             activation: bool = 'relu',  # 'tanh',
#             num_steps: int = 0,
#             model = 'nn') -> None:
#         """Hybrid Q10 model.

#         Note that restoring is not working currently as the model training is only taking
#         some minutes.
#         """

#         super().__init__()
#         self.validation_step_outputs = []  # Needed to change validation_epoch_end to on_validation_epoch_end: https://github.com/Lightning-AI/pytorch-lightning/pull/16520 
#         self.save_hyperparameters(
#             'features',
#             'targets',
#             'q10_init',
#             'hidden_dim',
#             'num_layers',
#             'dropout',
#             'activation',
#             'learning_rate',
#             'weight_decay',
#             'lambda_jacobian_l1',
#             'lambda_jacobian_l05',
#             'lambda_robustness',
#             'lambda_param_violation',
#         )

#         self.features = features
#         self.targets = targets

#         self.q10_init = q10_init

#         self.input_norm = norm.get_normalization_layer(variables=self.features, invert=False, stack=True)

#         self.model = model
#         self.nn = nn.Linear(len(self.features), 1)
 
#         self.target_norm = norm.get_normalization_layer(variables=self.targets, invert=False, stack=True)
#         self.target_denorm = norm.get_normalization_layer(variables=self.targets, invert=True, stack=True)

#         self.criterion = torch.nn.MSELoss()

#         self.q10 = torch.nn.Parameter(torch.ones(1) * self.q10_init)
#         self.ta_ref = 15.0

#         self.num_steps = num_steps

#         # Used for strring results.
#         self.ds = ds

#         # Error if more than 100000 steps--ok here, but careful if you copy code for other projects!.
#         self.q10_history = np.zeros(100000, dtype=np.float32) * np.nan

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
#         # Note that `x` is a dict of features and targets, input_norm extracts *only* features and stacks
#         # them along last dimension.
#         self.nn_input = self.input_norm(x)

#         # Forward pass through NN.
#         self.nn_input.requires_grad = True
#         self.rb = rb = self.nn(self.nn_input)

#         # print("INPUT", self.nn_input[0:5])
#         # print("SW POT", x['sw_pot'][0:5], "DSW_POT", x['dsw_pot'][0:5])
#         # analytic_rb = 0.0075 * x['sw_pot'] - 0.00375 * x['dsw_pot'] + 1.023  # Unnormalized data
#         # analytic_rb = 1.02 * self.nn_input[:, 0:1] - 1.81 * self.nn_input[:, 1:2] + 3.28
#         # analytic_reco = analytic_rb * (1.5 ** (0.1 * (x['ta'] - self.ta_ref)))

#         # Physical part.
#         reco = rb * self.q10 ** (0.1 * (x['ta'] - self.ta_ref))


#         # print(f"Rb: analytic {analytic_rb[0:5]}, predicted {self.rb[0:5]}")
#         # print(f"Reco: predicted {reco[0:5]}, true {x['reco'][0:5]}, analytic {analytic_reco[0:5]}")
#         # print("WEIGHTS", self.nn.weight.flatten(), self.nn.bias)

#         return reco, rb

#     def criterion_normed(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """Calculate criterion on normalized predictions and target."""
#         return self.criterion(
#             self.target_norm(y_hat),
#             self.target_norm(y)
#         )

#     def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
#         # Split batch (a dict) into actual data and the time-index returned by the dataset.
#         batch, _ = batch

#         # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
#         reco_hat, _ = self(batch)

#         # Calculate loss on normalized data.
#         loss = self.criterion_normed(reco_hat, batch['reco'])

#         # Save Q10 values, we want to know how they evolve with training,
#         self.q10_history[self.global_step] = self.q10.item()

#         # Logging.
#         self.log('train_loss', loss, prog_bar=True)  #on_step=False, on_epoch=True, prog_bar=False, logger=True)
#         self.log('q10', self.q10, prog_bar=True)  # on_step=True, on_epoch=False, prog_bar=False, logger=True)
#         return loss


#     def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
#         # Predictions are stored in validation step. This is not best practice, but we are more interested
#         # in predictions over training than on the final test predictions here.

#         # Split batch (a dict) into actual data and the time-index returned by the dataset.
#         batch, idx = batch

#         # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
#         reco_hat, rb_hat = self(batch)

#         # Calculate loss on normalized data.
#         loss = self.criterion_normed(reco_hat, batch['reco'])

#         # Calculate loss on normalized data.
#         self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

#         # This dict is available in `on_validation_epoch_end` after saving to self.validation_step_outputs.
#         outputs = {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}
#         self.validation_step_outputs.append(outputs)
#         return outputs


#     def on_validation_epoch_end(self) -> None:
#         print("WEIGHTS", self.nn.weight.flatten(), self.nn.bias)

#         # Iterate results from each validation step.
#         for item in self.validation_step_outputs:
#             reco_hat = item['reco_hat'][:, 0].cpu()
#             rb_hat = item['rb_hat'][:, 0].cpu()
#             idx = item['idx'].cpu()

#             # Assign predictions to the right time steps.
#             self.ds['reco_pred'].values[self.current_epoch, idx] = reco_hat
#             self.ds['rb_pred'].values[self.current_epoch, idx] = rb_hat
#         self.validation_step_outputs.clear()  # frree memory


#     def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
#         # Evaluation on test set.
#         batch, _ = batch
#         reco_hat, _ = self(batch)
#         loss = self.criterion_normed(reco_hat, batch['reco'])
#         self.log('test_loss', loss)

#     def configure_optimizers(self) -> torch.optim.Optimizer:

#         optimizer = torch.optim.AdamW(
#             [
#                 {
#                     'params': self.nn.parameters(),
#                     'weight_decay': self.hparams.weight_decay,
#                     'learning_rate': self.hparams.learning_rate
#                 },
#                 {
#                     'params': [self.q10],
#                     'weight_decay': 0.0,
#                     'learning_rate': self.hparams.learning_rate * 10
#                 }
#             ]
#         )

#         return optimizer

#     @staticmethod
#     def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument('--hidden_dim', type=int, default=16)
#         parser.add_argument('--num_layers', type=int, default=2)
#         parser.add_argument('--c', type=float, default=0.1)
#         parser.add_argument('--learning_rate', type=float, default=0.05)
#         parser.add_argument('--weight_decay', type=float, default=0.1)
#         parser.add_argument('--model', type=str, choices=['nn', 'senn'], default='nn')
#         return parser



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