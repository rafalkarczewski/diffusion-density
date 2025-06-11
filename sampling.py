import jax
import jax.numpy as jnp
import jax.random as jr

from utils import euler_solver, euler_maruyama_solver, divergence, gaussian_likelihood

def pf_ode_step(sde, l, x):
    # Probability flow ODE: dx/dl = f(l) x + 0.5 * g2(l) * score
    drift = sde.f(l) * x + 0.5 * sde.g2(l) * sde.score_fn(x, l)
    return drift

def hd_ode_step(sde, l, x):
    # High-density ODE: dx/dl = f(l)x + g2(l) * score
    drift = sde.f(l) * x + sde.g2(l) * sde.score_fn(x, l)
    return drift

def pf_ode_sampler(sde, x0, n_steps):
    """Simple PF-ODE sampling (no logp)"""
    lambdas = jnp.linspace(sde.lambda_min, sde.lambda_max, n_steps)
    drift_fn = lambda x, l: pf_ode_step(sde, l, x)
    return euler_solver(drift_fn, x0, lambdas)

def augmented_pf_ode_drift_fn(sde, state, l):
    """Evaluates the PF-ODE drift and the derivative of the marginal log-density (given by the negative divergence)"""
    x, logp, key = state
    new_key, old_key = jr.split(key)
    drift_fn = lambda y: pf_ode_step(sde, l, y)
    drift_eval, div = divergence(drift_fn, x, old_key, returnval=True)
    return (drift_eval, -div, new_key)

def pf_ode_sampler_with_logp(sde, x0, n_steps, key):
    """PF-ODE sampling with logp tracking (returns (x, logp))"""
    lambdas = jnp.linspace(sde.lambda_min, sde.lambda_max, n_steps)
    _, sigma_max = sde.alpha_sigma(sde.lambda_min)
    logp0 = gaussian_likelihood(x0, sigma_max)
    state0 = (x0, logp0, key)
    drift = lambda state, l: augmented_pf_ode_drift_fn(sde, state, l)
    final_x, final_logp, final_key = euler_solver(drift, state0, lambdas)
    return final_x, final_logp

def pf_ode_encoder_with_logp(sde, x0, n_steps, key):
    """PF-ODE encoding data to noise with logp tracking (returns (x, logp))"""
    lambdas = jnp.linspace(sde.lambda_max, sde.lambda_min, n_steps)
    state0 = (x0, 0., key)
    drift = lambda state, l: augmented_pf_ode_drift_fn(sde, state, l)
    encoded_x, dlogp, final_key = euler_solver(drift, state0, lambdas)
    _, sigma_max = sde.alpha_sigma(sde.lambda_min)
    prior_logp = gaussian_likelihood(encoded_x, sigma_max)
    final_logp = prior_logp - dlogp
    return encoded_x, final_logp

def hd_sampler(sde, x0, l_threshold, n_pf_ode_steps, n_hd_ode_steps):
    """High-density ODE sampler:
    1. Integrate PF-ODE from lambda_min to l_threshold
    2. Integrate HD-ODE from l_threshold to lambda_max
    """
    lambdas_pf = jnp.linspace(sde.lambda_min, l_threshold, n_pf_ode_steps)
    lambdas_hd = jnp.linspace(l_threshold, sde.lambda_max, n_hd_ode_steps)
    x_split = euler_solver(lambda x, l: pf_ode_step(sde, l, x), x0, lambdas_pf)
    x_final = euler_solver(lambda x, l: hd_ode_step(sde, l, x), x_split, lambdas_hd)
    return x_final

def sde_sampler(sde, x0, n_steps, key):
    """Stochastic SDE sampling using Euler-Maruyama."""
    lambdas = jnp.linspace(sde.lambda_min, sde.lambda_max, n_steps)
    def sde_terms(x, l, noise):
        drift = sde.f(l) * x + sde.g2(l) * sde.score_fn(x, l)
        diffusion = jnp.sqrt(sde.g2(l)) * noise
        return drift, diffusion
    return euler_maruyama_solver(sde_terms, x0, lambdas, key)

def sde_sampler_with_logp(sde, x0, n_steps, key):
    """Stochastic SDE sampling with logp tracking using Euler-Maruyama."""
    lambdas = jnp.linspace(sde.lambda_min, sde.lambda_max, n_steps)
    def sde_terms(state, l, noise):
        x, logp = state
        D = x0.size
        noise_x, noise_logp = noise
        # Note that noise_logp is ignored. The stochastic components of dxt and dlogpt are not independent
        # They are given by different transformations of the same D-dimensional Gaussian noise vector
        score = sde.score_fn(x, l)
        drift_x = sde.f(l) * x + sde.g2(l) * score
        drift_logp = - sde.f(l) * D + 0.5 * sde.g2(l) * jnp.sum(score ** 2)
        diffusion_x = jnp.sqrt(sde.g2(l)) * noise_x
        diffusion_logp = jnp.sqrt(sde.g2(l)) * jnp.sum(score * noise_x)
        return (drift_x, drift_logp), (diffusion_x, diffusion_logp)
    _, sigma_max = sde.alpha_sigma(sde.lambda_min)
    logp0 = gaussian_likelihood(x0, sigma_max)
    state0 = (x0, logp0)
    return euler_maruyama_solver(sde_terms, state0, lambdas, key)

if __name__ == "__main__":
    import equinox as eqx
    from utils import sde_from_checkpoint
    sde = sde_from_checkpoint("checkpoints/unweighted_model.eqx")
    key = jr.PRNGKey(5678)
    batch_size = 64
    n_ode_sampling_steps = 1024
    n_sde_sampling_steps = 1024

    prior_key, key = jr.split(key)
    latent = sde.sample_prior((batch_size, 3, 32, 32), prior_key)

    pf_sampling_fn = eqx.filter_jit(jax.vmap(lambda x, key: pf_ode_sampler_with_logp(sde, x, n_ode_sampling_steps, key)))
    ode_sampling_key, key = jr.split(key)
    ode_sample, logp_ode = pf_sampling_fn(latent, jr.split(ode_sampling_key, batch_size))
    regular_sampling_mean_logp = jnp.mean(logp_ode).item()

    pf_encoding_fn = eqx.filter_jit(jax.vmap(lambda x, key: pf_ode_encoder_with_logp(sde, x, n_ode_sampling_steps, key)))
    ode_encoding_key, key = jr.split(key)
    encoded, logp_ode_encoded = pf_encoding_fn(ode_sample, jr.split(ode_encoding_key, batch_size))

    print("Logp correlation betwen Rev-PF-ODE and Fwd-PF-ODE:", jnp.corrcoef(logp_ode, logp_ode_encoded)[0, 1], flush=True) # Running PF-ODE in both directions should yield approximately the same result

    sde_sampling_fn = eqx.filter_jit(jax.vmap(lambda x, key: sde_sampler_with_logp(sde, x, n_sde_sampling_steps, key)))
    sde_sampling_key, key = jr.split(key)
    sde_sample, logp_sde = sde_sampling_fn(latent, jr.split(sde_sampling_key, batch_size))

    ode_encoding_key, key = jr.split(key)
    encoded, logp_ode_encoded = pf_encoding_fn(sde_sample, jr.split(ode_encoding_key, batch_size))

    print("Logp correlation betwen Rev-SDE and Fwd-PF-ODE:", jnp.corrcoef(logp_sde, logp_ode_encoded)[0, 1], flush=True) # The logp estimate of a Rev-SDE sample should be close to the logp estimate of the sample encoded with PF-ODE

    print(f"Regular sampling mean logp: {regular_sampling_mean_logp:.2f}", flush=True)
    for l_thres in [8., 6., 4., 2.]:
        n_pf_ode_steps = int(1024 * (sde.lambda_max - l_thres) / (sde.lambda_max - sde.lambda_min))
        n_hd_ode_steps = int(1024 * (l_thres - sde.lambda_min) / (sde.lambda_max - sde.lambda_min))
        hd_sampling_fn = eqx.filter_jit(jax.vmap(lambda x: hd_sampler(sde, x, l_thres, n_pf_ode_steps, n_hd_ode_steps)))
        hd_samples = hd_sampling_fn(latent)
        ode_encoding_key, key = jr.split(key)
        _, logp = pf_encoding_fn(hd_samples, jr.split(ode_encoding_key, batch_size))
        mean_logp = jnp.mean(logp).item()
        print(f"LogSNR threshold: {l_thres}. HD sampling mean logp: {mean_logp:.2f}.", flush=True)
