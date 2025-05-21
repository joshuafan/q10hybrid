"""Fluxnet data loaders."""

from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from pytorch_lightning import LightningDataModule

import xarray as xr
import numpy as np

from typing import List, Tuple, Dict, Union, Iterable

from joblib.externals.loky.backend.context import get_context
import warnings
from utils.data_utils import Normalize
import kan

class FDataset(Dataset):
    """Fluxnet site data.

    Args:
        ds (xr.Dataset):
            Site cube data with dimension `time`.
        features (str):
            List of feature names.
        targets (str):
            List of target names.
        context_size (int):
            Context length (t-context_size+1 : t).
        norm (Normalize):
            Normalization module with all features and targets registered.
    """

    def __init__(
            self,
            ds: xr.Dataset,
            features: List[str],
            targets: List[str],
            context_size: int,
            norm: Normalize) -> None:

        self._ds = ds
        self._features = [features] if isinstance(features, str) else features
        self._targets = [targets] if isinstance(targets, str) else targets
        self._variables = self._features + self._targets
        self._context_size = context_size
        self._norm = norm

        self._ind2coord_lookup = np.argwhere(
            self._ds[self._targets].notnull().to_array().any('variable').values).flatten()

    def __len__(self) -> int:
        """The dataset length (number of samples), required.

        Returns:
            int: the length.
        """
        return len(self._ind2coord_lookup)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Returns a single item corresponding to the index `ind`.

        Args:
            idx (int):
                The index of the sample, in range [0, len(self)).

        Returns:
            Tuple:
                - Data (Dict[str, torch.Tensor]): features and targets, each with shape (1, context_size)
                - Time index (torch.Tensor): the time index corresponding to the data.
        """

        time = self._ind2coord_lookup[idx]

        ds_site_x = self._ds[self._variables].isel(time=slice(time - self._context_size + 1, time + 1))

        return {var: ds_site_x[var].values.astype('float32') for var in self._variables}, idx

    @property
    def num_features(self) -> int:
        return len(self._features)

    @property
    def num_targets(self) -> int:
        return len(self._targets)

    @property
    def num_variables(self) -> int:
        return self.num_features + self.num_targets


class FluxData(LightningDataModule):
    """Fluxnet site data.

    Args:
        ds (xr.Dataset):
            Site cube data with dimension `time`.
        features (str):
            List of feature names.
        targets (str):
            List of target names.
        context_size (int):
            Context length (t-context_size+1 : t).
        train_time (slice):
            Slice containg start and end time, e.g., slice('2001-01-01', '2001-12-31') of the
            training data.
        valid_time (slice):
            Slice containg start and end time, e.g., slice('2001-01-01', '2001-12-31') of the
            validation data.
        test_time (slice):
            Slice containg start and end time, e.g., slice('2001-01-01', '2001-12-31') of the
            test data.
        batch_size (int):
            The batch size.
        data_loader_kwargs (Dict, optional):
            Keyword arguments passed to Dataloader when calling one of the
            `[set]_dataloader` methods. Defaults is passing no further arguments.
    """
    def __init__(
            self,
            ds: xr.Dataset,
            features: List[str],
            targets: List[str],
            train_time: slice,
            valid_time: slice,
            test_time: slice,
            context_size: int,
            batch_size: int,
            data_loader_kwargs: Dict = {},
            subset_frac: float = 1.0,
            rb_synth: int = 0,
            remove_high: str = "none",
            remove_high_frac: str = 0.0,
            train_set: int = 0,
            reco_noise_std: float = 0.0,
            plot_dir: str = "./") -> None:

        super().__init__()

        # Versions of variables between (0, 1) for generating synthetic data
        ds["sw_pot_norm"] = (ds["sw_pot"] - ds["sw_pot"].min()) / (ds["sw_pot"].max() - ds["sw_pot"].min())
        ds["dsw_pot_norm"] = (ds["dsw_pot"] - ds["dsw_pot"].min()) / (ds["dsw_pot"].max() - ds["dsw_pot"].min())
        ds["ta_norm"] = (ds["ta"] - ds["ta"].min()) / (ds["ta"].max() - ds["ta"].min())

        # Synthetic Rb
        import matplotlib.pyplot as plt
        import os
        import math
        import torch

        # # Verify the formulas used 
        # # ds["rb_synth_tilde"] = 0.01 * ds["sw_pot"] - 0.005 * ds["dsw_pot"]
        # # ds["rb_synth_paper"] = 0.75 * (ds["rb_synth_tilde"] - ds["rb_synth_tilde"].min() + 0.1 * math.pi)
        # ds["rb_synth_paper"] = 0.0075 * ds["sw_pot"] - 0.00375 * ds["dsw_pot"] + 1.03506858
        # print("Rb shape", ds["rb_synth_paper"].values.shape)
        # eps = torch.nn.init.trunc_normal_(torch.empty(ds["rb_synth_paper"].values.shape, requires_grad=False), mean=0, std=0.2, a=-0.95, b=0.95)
        # print("Eps", eps)
        # ds[f"reco_synth_paper"] = ds["rb_synth_paper"] * (1.5 ** (0.1 * (ds['ta'] - 15)))  # * (1 + eps.numpy())
        # # print("Min value", ds["rb_synth_tilde"].min())
        # # print("Rb theirs", ds["rb"].values)
        # # print("Rb ours", ds["rb_synth_paper"].values)
        # # print("Error", ds["rb"].values - ds["rb_synth_paper"].values)
        # plt.scatter(ds["rb_synth_paper"].values, ds["rb"].values)
        # plt.xlabel("Our generated Rb")
        # plt.ylabel("Paper generated Rb")
        # plt.savefig(os.path.join(plot_dir, "rb_synth_paper.png"))
        # plt.close()
        # plt.scatter(ds["reco_synth_paper"].values, ds["reco"].values)
        # plt.xlabel("Our generated Reco")
        # plt.ylabel("Paper generated Reco")
        # plt.savefig(os.path.join(plot_dir, "reco_synth_paper.png"))
        # plt.close()


        if rb_synth != 0:
            if rb_synth == 1:
                ds["rb"] = (ds["dsw_pot_norm"] - 0.5) ** 2
            elif rb_synth == 2:
                ds["rb"] = (ds["sw_pot_norm"] - 0.5) ** 2 + (ds["dsw_pot_norm"] - 0.5) ** 2
            elif rb_synth == 3:
                ds["rb"] = np.minimum(0.3, np.maximum(0, ds["sw_pot_norm"] - 0.4)) - np.minimum(0.3, np.maximum(0, ds["dsw_pot_norm"] - 0.4))
            elif rb_synth == 4:
                temp = ds["sw_pot_norm"] - ds["dsw_pot_norm"] 
                ds["rb"] = np.log(temp - temp.min() + 0.1)
            elif rb_synth == 6:
                old_rb = 0.0075 * ds["sw_pot"] - 0.00375 * ds["dsw_pot"]   # + 1.03506858
                print("Min of old rb", old_rb.min())
                ds["rb"] = ((old_rb - old_rb.mean()) / old_rb.std()) **2
            elif rb_synth == 7:  # Inverse relu
                old_rb = 0.0075 * ds["sw_pot"] - 0.00375 * ds["dsw_pot"]   # + 1.03506858
                print("Min of old rb", old_rb.min())
                ds["rb"] = np.minimum((old_rb - old_rb.mean()) / old_rb.std(), 0)
            elif rb_synth == 8:
                # linear followed by abs
                old_rb = 0.0075 * ds["sw_pot"] - 0.00375 * ds["dsw_pot"]
                mean_val, std_val = old_rb.mean().item(), old_rb.std().item()
                ds["rb"] = np.abs((old_rb - mean_val) / std_val)

                # Construct true KAN for true feature importance
                true_kan = kan.KAN(width=[len(features), 1, len(targets)], device="cpu", base_fun="identity", seed=torch.initial_seed())

                # set mask to 0 to ignore spline (learnable) portion and use symbolic only
                true_kan.act_fun[0].mask = torch.zeros_like(true_kan.act_fun[0].mask)
                true_kan.act_fun[1].mask = torch.zeros_like(true_kan.act_fun[1].mask)
                true_kan.save_acts = True
                def f1(x):
                    return 0.0075*x
                def f2(x):
                    return -0.00375*x
                def f3(x):
                    return ((x - mean_val) / std_val).abs()
                true_kan.fix_symbolic(0, 0, 0, fun_name=f1, random=False, fit_params_bool=False, verbose=False)
                true_kan.fix_symbolic(0, 1, 0, fun_name=f2, random=False, fit_params_bool=False, verbose=False)
                true_kan.fix_symbolic(1, 0, 0, fun_name=f3, random=False, fit_params_bool=False, verbose=False)

                with torch.no_grad():
                    input_features = torch.tensor(np.stack([ds["sw_pot"].values, ds["dsw_pot"].values, ds["ta"].values], axis=1), dtype=torch.float32)
                    prescribed_para = true_kan(input_features)

                # Plot true functional relationships
                true_kan.attribute()
                true_kan.node_attribute()
                true_kan.plot(folder=os.path.join(plot_dir, "splines"), in_vars=features, out_vars=targets)  #  scale=5, varscale=0.13)
                plt.savefig(os.path.join(plot_dir, f"TRUE_kan_plot.png"))
                plt.close()

                # For now, use PartialDependenceVariance to measure importance
                # of each functional relationship. There may be a better way
                # but not trivial for multi-layer KAN.
                from alibi.explainers import ALE, PartialDependenceVariance, plot_ale, plot_pd_variance
                @torch.no_grad()
                def predictor(X: np.ndarray) -> np.ndarray:
                    X = torch.as_tensor(X, device="cpu")
                    return true_kan(X).cpu().numpy()

                with warnings.catch_warnings():  # suppress warnings inside alibi code
                    warnings.simplefilter("ignore")
                    pd_variance = PartialDependenceVariance(predictor=predictor,
                                                            feature_names=features,
                                                            target_names=targets)
                    exp_importance = pd_variance.explain(input_features.detach().cpu().numpy(), method='importance')
                    importance_scores = exp_importance.data['feature_importance'].T  # transpose to [n_inputs, n_params]
                    self.true_relationships = importance_scores / importance_scores.sum(axis=0, keepdims=True)

            elif rb_synth == 9:
                ds["rb"] = 0.0075 * ds["sw_pot"] - 0.00375 * ds["dsw_pot"] + 1.03506858

                # construct true KAN for true feature importance
                true_kan = kan.KAN(width=[len(features), len(targets)], device="cpu", base_fun="identity", seed=torch.initial_seed())

                # set mask to 0 to ignore spline (learnable) portion and use symbolic only
                true_kan.act_fun[0].mask = torch.zeros_like(true_kan.act_fun[0].mask)
                true_kan.save_acts = True
                def f1(x):
                    return 0.0075*x
                def f2(x):
                    return -0.00375*x
                true_kan.fix_symbolic(0, 0, 0, fun_name=f1, random=False, fit_params_bool=False, verbose=False)
                true_kan.fix_symbolic(0, 1, 0, fun_name=f2, random=False, fit_params_bool=False, verbose=False)

                with torch.no_grad():
                    input_features = torch.tensor(np.stack([ds["sw_pot"].values, ds["dsw_pot"].values, ds["ta"].values], axis=1), dtype=torch.float32)
                    prescribed_para = true_kan(input_features)

                # Plot true functional relationships
                true_kan.attribute()
                true_kan.node_attribute()
                true_kan.plot(folder=os.path.join(plot_dir, "splines"), in_vars=features, out_vars=targets)  #  scale=5, varscale=0.13)
                plt.savefig(os.path.join(plot_dir, f"TRUE_kan_plot.png"))
                plt.close()

                # For now, use PartialDependenceVariance to measure importance
                # of each functional relationship. There may be a better way
                # but not trivial for multi-layer KAN.
                from alibi.explainers import ALE, PartialDependenceVariance, plot_ale, plot_pd_variance
                @torch.no_grad()
                def predictor(X: np.ndarray) -> np.ndarray:
                    X = torch.as_tensor(X, device="cpu")
                    return true_kan(X).cpu().numpy()

                self.true_relationships = true_kan.edge_scores[0].detach().cpu().numpy().T  # transpose to [n_inputs, n_params]
                self.true_relationships = self.true_relationships / self.true_relationships.sum(axis=0, keepdims=True)
                print("True relationships", self.true_relationships)
            elif rb_synth == 10:
                # linear followed by quadratic
                old_rb = 0.0075 * ds["sw_pot"] - 0.00375 * ds["dsw_pot"]
                mean_val, std_val = old_rb.mean().item(), old_rb.std().item()
                ds["rb"] = ((old_rb - mean_val) / std_val) ** 2

                # Construct true KAN for true feature importance
                true_kan = kan.KAN(width=[len(features), 1, len(targets)], device="cpu", base_fun="identity", seed=torch.initial_seed())

                # set mask to 0 to ignore spline (learnable) portion and use symbolic only
                true_kan.act_fun[0].mask = torch.zeros_like(true_kan.act_fun[0].mask)
                true_kan.act_fun[1].mask = torch.zeros_like(true_kan.act_fun[1].mask)
                true_kan.save_acts = True
                def f1(x):
                    return 0.0075*x
                def f2(x):
                    return -0.00375*x
                def f3(x):
                    return ((x - mean_val) / std_val) ** 2
                true_kan.fix_symbolic(0, 0, 0, fun_name=f1, random=False, fit_params_bool=False, verbose=False)
                true_kan.fix_symbolic(0, 1, 0, fun_name=f2, random=False, fit_params_bool=False, verbose=False)
                true_kan.fix_symbolic(1, 0, 0, fun_name=f3, random=False, fit_params_bool=False, verbose=False)

                with torch.no_grad():
                    input_features = torch.tensor(np.stack([ds["sw_pot"].values, ds["dsw_pot"].values, ds["ta"].values], axis=1), dtype=torch.float32)
                    prescribed_para = true_kan(input_features)

                # Plot true functional relationships
                true_kan.attribute()
                true_kan.node_attribute()
                true_kan.plot(folder=os.path.join(plot_dir, "splines"), in_vars=features, out_vars=targets)  #  scale=5, varscale=0.13)
                plt.savefig(os.path.join(plot_dir, f"TRUE_kan_plot.png"))
                plt.close()

                # For now, use PartialDependenceVariance to measure importance
                # of each functional relationship. There may be a better way
                # but not trivial for multi-layer KAN.
                from alibi.explainers import ALE, PartialDependenceVariance, plot_ale, plot_pd_variance
                @torch.no_grad()
                def predictor(X: np.ndarray) -> np.ndarray:
                    X = torch.as_tensor(X, device="cpu")
                    return true_kan(X).cpu().numpy()

                with warnings.catch_warnings():  # suppress warnings inside alibi code
                    warnings.simplefilter("ignore")
                    pd_variance = PartialDependenceVariance(predictor=predictor,
                                                            feature_names=features,
                                                            target_names=targets)
                    exp_importance = pd_variance.explain(input_features.detach().cpu().numpy(), method='importance')
                    importance_scores = exp_importance.data['feature_importance'].T  # transpose to [n_inputs, n_params]
                    self.true_relationships = importance_scores / importance_scores.sum(axis=0, keepdims=True)
            else:
                raise ValueError(f"Invalid value of rb_synth, should be integer between [0, 7]. Got {rb_synth}.")

            # TODO Not sure where the true q10 is set, hardcoding to 1.5 for now.
            ds["rb"] = ds["rb"] - ds["rb"].min() + 0.1  # Require non-negativity
        else:
            # same formula as paper
            ds["rb"] = 0.0075 * ds["sw_pot"] - 0.00375 * ds["dsw_pot"] + 1.03506858
        # Add noise
        ds["reco"] = ds["rb"] * (1.5 **(0.1 * (ds['ta'] - 15)))
        if reco_noise_std > 0.0:
            eps = torch.nn.init.trunc_normal_(torch.empty(ds["rb"].values.shape, requires_grad=False), mean=0, std=reco_noise_std, a=-0.95, b=0.95)
            ds["reco"] = ds["reco"] * (1 + eps.numpy())

        self._ds = ds
        self._features = [features] if isinstance(features, str) else features
        self._targets = [targets] if isinstance(targets, str) else targets
        self._train_time = train_time
        self._valid_time = valid_time
        self._test_time = test_time
        self._context_size = context_size
        self._batch_size = batch_size
        self._data_loader_kwargs = data_loader_kwargs

        self._ds_train = self._ds.sel(time=self._train_time).load()
        self._ds_valid = self._ds.sel(time=self._valid_time).load()
        self._ds_test = self._ds.sel(time=self._test_time).load()

        # If we want to test extrapolation, remove high values of (reco, rb, or ta)
        # from the training data.
        if remove_high == "reco":
            reco_thresh = np.quantile(self._ds_train["reco"].values, 1 - remove_high_frac)
            self._ds_train = self._ds_train.where(self._ds_train["reco"] <= reco_thresh, drop=True)
        elif remove_high == "rb":
            rb_thresh = np.quantile(self._ds_train["rb"].values, 1 - remove_high_frac)
            self._ds_train = self._ds_train.where(self._ds_train["rb"] <= rb_thresh, drop=True)
        elif remove_high == "ta":
            ta_thresh = np.quantile(self._ds_train["ta"].values, 1 - remove_high_frac)
            self._ds_train = self._ds_train.where(self._ds_train["ta"] <= ta_thresh, drop=True)
        elif remove_high == "none":
            pass
        else:
            raise ValueError("Invalid value of remove_high. Must be 'reco', 'rb', 'ta', or 'none'.")

        # Visualizations
        import os
        import sys
        sys.path.append('../')
        sys.path.append('../../')
        import visualization_utils
        fig, axes = plt.subplots(5, 3, figsize=(9, 15), sharex=True)  #, gridspec_kw={'hspace': 0.35, 'wspace': 0.12})
        variables = ["sw_pot", "dsw_pot", "ta", "rb", "reco"]
        datasets = [("train", self._ds_train), ("valid", self._ds_valid), ("test", self._ds_test)]
        for i, var in enumerate(variables):
            for j, (mode, ds) in enumerate(datasets):
                if var in ds:
                    ds[var].plot(ax=axes[i, j])
                    axes[i, j].set_title(f'{mode} {var}')
                else:
                    axes[i, j].set_title(f'{mode} {var} not found')
                    axes[i, j].axis('off')        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "features.png"))  #, dpi=300,bbox_inches='tight')
        plt.close()
        print("Plotted features", os.path.join(plot_dir, "features.png"))
        fig, axes = plt.subplots(len(self.features), len(self.features), figsize=(18, 18))
        for i in range(len(self.features)):
            for j in range(i+1, len(self.features)):
                visualization_utils.plot_single_scatter(axes[i,j], self._ds_train[self.features[i]].values, 
                                                        self._ds_train[self.features[j]].values,
                                                        x_label=self.features[i], y_label=self.features[j], 
                                                        title=f'{self.features[i]} vs {self.features[j]}', should_align=False)
        plt.savefig(os.path.join(plot_dir, "feat_correlations.png"))
        plt.close()        

        # Random subset for low-data regime
        if subset_frac < 1.0:
            n = self._ds_train.time.size
            self._ds_train = self._ds_train.isel(time=np.random.choice(n, size=int(subset_frac*n), replace=False))
            n = self._ds_valid.time.size
            self._ds_valid = self._ds_valid.isel(time=np.random.choice(n, size=int(subset_frac*n), replace=False))
            n = self._ds_test.time.size
            self._ds_test = self._ds_test.isel(time=np.random.choice(n, size=int(subset_frac*n), replace=False))

        # Register normalization parameters from training data.
        self._norm = Normalize()
        self._norm.register_xr(self._ds_train, self._features + self._targets)

        # These are constant kwargs to FDataset.
        self._datakwargs = {
            'features': self._features,
            'targets': self._targets,
            'context_size': self._context_size,
            'norm': self._norm
        }

    @property
    def features(self) -> List[str]:
        return self._features

    @property
    def targets(self) -> List[str]:
        return self._targets

    @property
    def num_features(self) -> int:
        return len(self._features)

    @property
    def num_targets(self) -> int:
        return len(self._targets)

    def train_dataloader(self) -> DataLoader:
        """"Get the training dataloader."""
        return DataLoader(
            FDataset(
                self._ds_train,
                **self._datakwargs),
            batch_size=self._batch_size,
            shuffle=True,
            multiprocessing_context=get_context('loky'),  # github.com/pytorch/pytorch/issues/44687
            **self._data_loader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        """"Get the validation dataloader."""
        return DataLoader(
            FDataset(
                self._ds_valid,
                **self._datakwargs),
            batch_size=self._batch_size,
            shuffle=True,  # Shuffle since the plot function only uses one batch - we want it to be representative
            multiprocessing_context=get_context('loky'),  # github.com/pytorch/pytorch/issues/44687,
            **self._data_loader_kwargs
        )

    def test_dataloader(self) -> DataLoader:
        """"Get the testing dataloader."""
        return DataLoader(
            FDataset(
                self._ds_test,
                **self._datakwargs),
            batch_size=self._batch_size,
            shuffle=True,
            multiprocessing_context=get_context('loky'),  # github.com/pytorch/pytorch/issues/44687
            **self._data_loader_kwargs
        )

    def target_xr(
            self,
            mode: str,
            varnames: Union[str, List[str]],
            num_epochs: int = 1) -> xr.Dataset:
        if mode not in ('train', 'valid', 'test'):
            raise ValueError(
                f'`mode` must be on of (`train` | `valid` | `test`), is `{mode}`.'
            )

        if mode == 'train':
            ds = self._ds_train
        elif mode == 'valid':
            ds = self._ds_valid
        elif mode == 'test':
            ds = self._ds_test
        else:
            raise ValueError(
                f'`mode` must be on of (`train` | `valid` | `test`), is `{mode}`.'
            )

        varnames = [varnames] if isinstance(varnames, str) else varnames

        ds_new = ds[varnames]

        for var in varnames:
            var_new = var + '_pred'
            dummy = ds[var].copy()
            dummy.values[:] = np.nan
            dummy = dummy.expand_dims(epoch=np.arange(num_epochs + 1))
            ds_new[var_new] = dummy.copy()

        return ds_new

    def add_scalar_record(self, ds: xr.Dataset, varname: str, x: Iterable) -> xr.Dataset:

        if isinstance(x, Tensor):
            x = x.detach().cpu().numpy()

        # Cut excess entries (NaNs).
        x = x[:x.argmin()]

        if 'iter' not in ds.coords:
            ds = ds.assign_coords({'iter': np.arange(len(x))})
        else:
            if len(ds['iter']) != len(x):
                raise ValueError(
                    f'dimension `iter` already exists in `ds`, but length ({len(ds["iter"])}) does '
                    f'not match length of `x` ({len(x)}).'
                )

        ds[varname] = ('iter', x)

        return ds

    def teardown(self) -> None:
        """Clean up after fit or test, called on every process in DDP."""
        self._ds_train.close()
        self._ds_valid.close()
        self._ds_test.close()
