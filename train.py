"""Script for training the diffusion model"""
import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from utils import cifar, dataloader, ImportanceSampler, update_sampler, VPSDE
import unet

def single_loss_fn(model, sde, weight_fn, data, l, key):
    """
    weighted ELBO loss of the VP-SDE model using noise-paramterization and l=logSNR(t) noise schedule parametrization as recommended by "Understanding diffusion objectives as the elbo with simple data augmentation" by Kingma and Gao (NeurIPS 2023)
    Arguments:
      model: Callable
        noise-parametrized score model
      sde: SDE
        specification of the SDE model
      weight_fn: Callable
        what weight to assign to the loss based on the noise level.
      data: jnp.ndarray
        training data point
      l: jnp.float
        noise level represented with l=logSNR(t)
      key: jr.key
        random key used by the model
    """
    alpha, sigma = sde.alpha_sigma(l)
    mean = data * alpha
    noise = jr.normal(key, data.shape)
    y = mean + sigma * noise
    pred = model(l, y, key=key)
    weight = weight_fn(l)
    return weight * jnp.sum((pred - noise) ** 2)

def batch_loss_fn(model, sde, weight_fn, data, key, sampler):
    """
    Batched loss fn using importance weight resampling.
    Arguments:
      model: Callable
        noise-parametrized score model
      sde: SDE
        specification of the SDE model
      weight_fn: Callable
        what weight to assign to the loss based on the noise level.
      data: jnp.ndarray
        batch of training data points
      l: jnp.float
        noise level represented with l=logSNR(t)
      key: jr.key
        random key used by the model
      sampler: ImportanceSampler
        used for resampling noise levels to minimize variance
    """
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t to reduce variance
    l = jr.uniform(tkey, (batch_size,), minval=sampler.l_min, maxval=sampler.l_min + (sampler.l_max - sampler.l_min) / batch_size)
    l = l + ((sampler.l_max - sampler.l_min) / batch_size) * jnp.arange(batch_size)
    prob_l, reweighted_l = sampler.prob_and_transform(l)
    loss_fn = ft.partial(single_loss_fn, model, sde, weight_fn)
    loss_fn = jax.vmap(loss_fn)
    uniform_prob = 1 / (sampler.l_max - sampler.l_min)
    importance_weights = uniform_prob / prob_l
    return jnp.mean(loss_fn(data, reweighted_l, losskey) * importance_weights)

@eqx.filter_jit
def make_step(model, sde, weight_fn, data, key, sampler, opt_state, opt_update):
    """Model update function. Estimates the loss and performs a single update on model's parameters"""
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, sde, weight_fn, data, key, sampler)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state

if __name__ == "__main__":
    seed = 5678
    key = jr.PRNGKey(seed)
    model_key, train_key, loader_key, sampler_update_key = jr.split(key, 4)
    data = cifar()
    data_mean = jnp.mean(data, axis=(0, 2, 3), keepdims=True)
    data_std = jnp.std(data, axis=(0, 2, 3), keepdims=True)
    data_shape = data.shape[1:] #(3, 32, 32)
    train_data = (data - data_mean) / data_std

    model = unet.UNet(
        data_shape=data_shape,
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
    opt = optax.adabelief(3e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    total_value = 0
    total_size = 0
    num_steps = 1000000
    batch_size = 256
    print_every = 100

    LAMBDA_MIN = -10
    LAMBDA_MAX = 10

    sampler = ImportanceSampler(LAMBDA_MIN, LAMBDA_MAX, n_bins=100, eta=0.99, p=jnp.ones(100))
    sde = VPSDE(lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, score_fn=None)

    weight_fn = lambda l: 1. # unweighted training - optimized for likelihood
    #weight_fn = lambda l: jax.nn.sigmoid(-l + 2) # weighted training - optimized for `sample quality`

    for step, data in zip(
        range(num_steps), dataloader(train_data, batch_size, key=loader_key)
    ):
        value, model, train_key, opt_state = make_step(
            model, sde, weight_fn, data, train_key, sampler, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step={step} Loss={total_value / total_size:.5f}", flush=True)
            total_value = 0
            total_size = 0
            if step > 0:
                sampler_update_key = jr.split(sampler_update_key, 1)[0]
                eval_data = next(dataloader(train_data, 512, key=sampler_update_key))
                sampler_update_loss_fn = ft.partial(single_loss_fn, model, sde, weight_fn)
                uniform_loss, sampler = update_sampler(sampler_update_loss_fn, data, sampler_update_key, sampler)
                print(f"Step={step} Uniform loss={uniform_loss.item():.5f}", flush=True)
        if (step + 1) % 40000 == 0:
            eqx.tree_serialise_leaves(f"model_step_{step}.eqx", model)
