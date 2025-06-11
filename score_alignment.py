import jax
import jax.random as jr
import jax.numpy as jnp

from utils import divergence, euler_solver
from sampling import pf_ode_step

def score_alignment_drift_fn(sde, state, l):
    x, v = state
    drift_fn = lambda y: pf_ode_step(sde, l, y)
    return jax.jvp(drift_fn, (x,), (v,))

def score_alignment(sde, x0, n_steps, l_max):
    lambdas = jnp.linspace(sde.lambda_min, l_max, n_steps)
    v0 = sde.score_fn(x0, sde.lambda_min)
    drift_fn = lambda state, l: score_alignment_drift_fn(sde, state, l)
    x1, v1 = euler_solver(drift_fn, (x0, v0), lambdas)
    final_score = sde.score_fn(x1, l_max)
    return jnp.sum(final_score * v1)

def score_alignment_drift_no_score(sde, state, l):
    (x, v, omega, key) = state
    old_key, new_key = jr.split(key)
    x_drift_fn = lambda y: pf_ode_step(sde, l, y)
    def v_drift_fn(y):
        x_drift, v_drift = jax.jvp(x_drift_fn, (y,), (v,))
        return v_drift, x_drift
    v_drift, div, x_drift = divergence(v_drift_fn, x, old_key, returnval=True, has_aux=True)
    return x_drift, v_drift, -div, new_key

def score_alignment_no_score(sde, x0, n_steps, l_max, key):
    lambdas = jnp.linspace(sde.lambda_min, l_max, n_steps)
    v0 = sde.score_fn(x0, sde.lambda_min)
    omega0 = jnp.sum(v0 ** 2)
    drift_fn = lambda state, l: score_alignment_drift_no_score(sde, state, l)
    x1, v1, omega1, key1 = euler_solver(drift_fn, (x0, v0, omega0, key), lambdas)
    return omega1

if __name__ == "__main__":
    import equinox as eqx
    from utils import sde_from_checkpoint
    sde = sde_from_checkpoint("checkpoints/unweighted_model.eqx")
    key = jr.PRNGKey(5678)
    batch_size = 128

    prior_key, key = jr.split(key)
    latent = sde.sample_prior((batch_size, 3, 32, 32), prior_key)

    lambda_thres = 1.
    n_solver_steps = 1024

    score_alignment_fn = lambda x: score_alignment(
        sde, x, n_steps=n_solver_steps, l_max=lambda_thres)
    score_alignment_fn = eqx.filter_jit(jax.vmap(score_alignment_fn))

    score_alignment_no_score_fn = lambda x, key: score_alignment_no_score(
        sde, x, n_steps=n_solver_steps, l_max=lambda_thres, key=key)
    score_alignment_no_score_fn = eqx.filter_jit(jax.vmap(score_alignment_no_score_fn))
    
    score_alignment_key, key = jr.split(key)
    
    score_alignment_w_score = score_alignment_fn(latent)
    score_alignment_wo_score = score_alignment_no_score_fn(latent, jr.split(score_alignment_key, latent.shape[0]))

    corr = jnp.corrcoef(score_alignment_w_score, score_alignment_wo_score)[0, 1].item()
    print(f"Correlation of score alignment with and without score: {corr}", flush=True)
