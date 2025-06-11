from abc import ABC, abstractmethod
import os
import pickle
import tarfile
import urllib.request

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import functools as ft
import unet

class ImportanceSampler(eqx.Module):
    """Adaptive importance sampler described in "Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation" by Kingma and Gao (NeurIPS 2023)"""
    l_min: float
    l_max: float
    n_bins: int
    eta: float
    p: jnp.ndarray
    bin_edges: jnp.ndarray
    bin_width: float

    def __init__(self, l_min, l_max, n_bins, p=None, eta=0.999):
        self.n_bins = n_bins
        self.l_min = l_min
        self.l_max = l_max
        self.bin_width = (l_max - l_min) / n_bins
        self.eta = eta
        self.p = p if p is not None else jnp.ones(n_bins)
        self.bin_edges = jnp.linspace(l_min, l_max, n_bins + 1)

    def update(self, l, v):
        bin_indices = jnp.floor((l - self.l_min) / self.bin_width).astype(int)
        old_values = self.p[bin_indices]
        new_p = self.p.at[bin_indices].set(old_values * self.eta + (1 - self.eta) * v)
        return eqx.tree_at(lambda sampler: sampler.p, self, new_p)

    def prob(self, l):
        bin_indices = jnp.floor((l - self.l_min) / self.bin_width).astype(int)
        return self.p[bin_indices] / jnp.sum(self.p) / self.bin_width

    def transform(self, l):
        cumsums = self._compute_sums()
        normalized_l = (l - self.l_min) / (self.l_max - self.l_min)
        return jnp.interp(normalized_l, cumsums, self.bin_edges)

    def prob_and_transform(self, l):
        cumsums = self._compute_sums()
        normalized_l = (l - self.l_min) / (self.l_max - self.l_min)
        transformed = jnp.interp(normalized_l, cumsums, self.bin_edges)
        bin_indices = jnp.floor((transformed - self.l_min) / self.bin_width).astype(int)
        probs = self.p[bin_indices] / jnp.sum(self.p) / self.bin_width
        return probs, transformed

    def _compute_sums(self):
        total_mass = jnp.sum(self.p)
        cumsums = jnp.cumsum(self.p) / total_mass
        cumsums = jnp.concatenate([jnp.array([0.]), cumsums])
        return cumsums

@eqx.filter_jit
def update_sampler(loss_fn, data, key, sampler):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over l to reduce variance
    l = jr.uniform(tkey, (batch_size,), minval=sampler.l_min, maxval=sampler.l_min + (sampler.l_max - sampler.l_min) / batch_size)
    l = l + ((sampler.l_max - sampler.l_min) / batch_size) * jnp.arange(batch_size)
    loss_fn = jax.vmap(loss_fn)
    losses = loss_fn(data, l, losskey)
    return jnp.mean(losses), sampler.update(l, losses)

def cifar():
    filename = "cifar-10-python.tar.gz"
    url_dir = "https://www.cs.toronto.edu/~kriz"
    target_dir = os.getcwd() + "/data/cifar-10"
    url = f"{url_dir}/{filename}"
    tar_path = f"{target_dir}/{filename}"

    if not os.path.exists(tar_path):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, tar_path)
        print(f"Downloaded {url} to {tar_path}")

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    extracted_dir = os.path.join(target_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted_dir):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
            print(f"Extracted {tar_path} to {target_dir}")

    # List of data batch files
    batch_files = [os.path.join(target_dir, f'cifar-10-batches-py/data_batch_{i}') for i in range(1, 6)]
    
    # Load and concatenate all batch files
    data = []
    for file in batch_files:
        batch = unpickle(file)
        data.append(batch[b'data'])
    
    data = jnp.vstack(data)
    
    # Reshape and transpose to get the shape (60000, 3, 32, 32)
    data = data.reshape(-1, 3, 32, 32).astype('uint8')
    
    # Convert to jax.numpy array and return
    return jnp.array(data)

def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size

def euler_solver(drift_fn, state0, ts):
    """
    Args:
        drift_fn: (state, t) -> drift (PyTree, same structure as state)
        state0: initial state (array, tuple, etc.)
        ts: array of timepoints

    Returns:
        state_final (same PyTree as state0)
    """
    def scan_step(state, i):
        t, t_next = ts[i], ts[i+1]
        dt = t_next - t
        drift = drift_fn(state, t)
        def update_fn(_state, _drift):
            if isinstance(_state, jax.Array) and _state.dtype in [jnp.float32, jnp.float64]:
                return _state + _drift * dt
            else:
                # For PRNG keys (usually uint32), just take d (the new key)
                return _drift
        new_state = jax.tree_util.tree_map(update_fn, state, drift)
        return new_state, None

    state_final, _ = jax.lax.scan(scan_step, state0, jnp.arange(len(ts) - 1))
    return state_final


def normal_like(x, key):
    if isinstance(x, jnp.ndarray):
        return jr.normal(key, x.shape)
    else:
        keys = jr.split(key, jax.tree_util.tree_leaves(x).__len__())
        keys_tree = jax.tree.unflatten(jax.tree.structure(x), keys)
        return jax.tree.map(normal_like, x, keys_tree)

def euler_maruyama_solver(sde_terms, state0, ts, key):
    """
    Args:
        sde_terms: (state, t, noise) -> (drift, diffusion), all PyTrees matching state
        state0: initial state (PyTree)
        ts: array of timepoints
        key: jax.random.PRNGKey

    Returns:
        state_final (PyTree)
    """
    def scan_step(carry, i):
        state, key = carry
        t, t_next = ts[i], ts[i+1]
        dt = t_next - t
        key, subkey = jr.split(key)
        noise = normal_like(state, subkey)
        drift, diffusion = sde_terms(state, t, noise)
        def update_fn(_state, _drift, _diffusion):
            if isinstance(_state, jax.Array) and _state.dtype in [jnp.float32, jnp.float64]:
                return _state + _drift * dt + _diffusion * jnp.sqrt(jnp.abs(dt))
            else:
                # For PRNG keys (usually uint32), just take d (the new key)
                return _drift
        # Euler step: user must combine diffusion and noise however they want, but a typical use would be: s + d*dt + sqrt(dt) * diffusion*noise
        state_next = jax.tree_util.tree_map(update_fn, state, drift, diffusion)
        return (state_next, key), None

    (state_final, _), _ = jax.lax.scan(scan_step, (state0, key), jnp.arange(len(ts) - 1))
    return state_final

def gaussian_likelihood(x, sigma):
    return -0.5 * x.size * jnp.log(2 * jnp.pi * sigma ** 2) - 0.5 * jnp.sum((x / sigma)**2)

def divergence(fn, x, key, returnval=False, has_aux=False):
    """Stochastic estimation of divergence of fn at x using the Hutchinson's trace trick with a single Rademacher variable"""
    v = jr.rademacher(key, shape=x.shape, dtype=jnp.float32)
    if has_aux:
        fval, jvp, aux = jax.jvp(fn, (x,), (v,), has_aux=True)
    else:
        fval, jvp = jax.jvp(fn, (x,), (v,))
    div = jnp.sum(v * jvp)
    if returnval:
        if has_aux:
            return fval, div, aux
        else:
            return fval, div
    else:
        if has_aux:
            return div, aux
        else:
            return div

class SDE(ABC):
    def __init__(self, lambda_min, lambda_max, score_fn):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.score_fn = score_fn

    @abstractmethod
    def alpha_sigma(self, l):
        pass

    @abstractmethod
    def f(self, l):
        pass

    @abstractmethod
    def g2(self, l):
        "always equal to sigma(l)**2"
        pass

    def sample_prior(self, shape, key):
        _, sigma = self.alpha_sigma(self.lambda_min)
        return sigma * jr.normal(key, shape)

class VPSDE(SDE):
    def alpha_sigma(self, l):
        alpha = jnp.sqrt(jax.nn.sigmoid(l))
        sigma = jnp.sqrt(jax.nn.sigmoid(-l))
        return alpha, sigma
    
    def f(self, l):
        return 0.5 * jax.nn.sigmoid(-l)
    
    def g2(self, l):
        return jax.nn.sigmoid(-l)

class VESDE(SDE):
    def alpha_sigma(self, l):
        alpha = 1.
        sigma = jnp.exp(-0.5 * l)
        return alpha, sigma

    def f(self, l):
        return 0.
    
    def g2(self, l):
        return jnp.exp(-l)

def sde_from_checkpoint(checkpoint, key=jr.PRNGKey(5678)):
    model = unet.UNet(
        data_shape=(3, 32, 32),
        is_biggan=True,
        dim_mults=(1, 2, 2, 2),
        hidden_size=128,
        heads=8,
        dim_head=16,
        dropout_rate=0.1,
        num_res_blocks=4,
        attn_resolutions=[16],
        key=key
    )
    model = eqx.tree_deserialise_leaves(checkpoint, model)
    model = eqx.nn.inference_mode(model) # Needs to be applied, because the model is using dropout
    sde = VPSDE(
        lambda_min=-10.,
        lambda_max=10.,
        score_fn=lambda x, l: - model(l, x, key=None) / jnp.sqrt(jax.nn.sigmoid(-l)) # the models were trained with noise-parametrization, so `score = - eps/sigma`
    )
    return sde
