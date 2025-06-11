import jax
import jax.numpy as jnp
import jax.random as jr

from scipy.stats import chi
from sampling import pf_ode_sampler, pf_ode_encoder_with_logp
from utils import euler_solver, euler_maruyama_solver, divergence

def prior_rescaling(sde, x0, target_norm):
    """Rescaling the noise vector"""
    norm = jnp.linalg.norm(x0)
    _, sigma_prior = sde.alpha_sigma(sde.lambda_min)
    target_norm = target_norm * sigma_prior
    return x0 * (target_norm / norm)

def prior_guidance_sampler(sde, x0, target_norm, n_steps):
    x0 = prior_rescaling(sde, x0, target_norm)
    return pf_ode_sampler(sde, x0, n_steps)

def density_guidance_ode_step(sde, l, x, q, noise_strength=0., key=jr.PRNGKey(1234)):
    D = x.size
    _, sigma = sde.alpha_sigma(l)
    quant = jnp.sqrt(2 * D) * jax.scipy.stats.norm.ppf(q)
    def noise_rescale_score():
        score, laplacian = divergence(lambda y: sde.score_fn(y, l), x, key, returnval=True)
        eta = 1 + (quant - (sigma * noise_strength) ** 2 * laplacian) / jnp.sum((sigma * score) ** 2)
        return eta * score
    def no_noise_rescale_score():
        score = sde.score_fn(x, l)
        eta = 1 + quant / jnp.sum((sigma * score) ** 2)
        return eta * score
    score = jax.lax.cond(jnp.abs(noise_strength) > 1e-6, noise_rescale_score, no_noise_rescale_score)
    return sde.f(l) * x + 0.5 * sde.g2(l) * score, score

def density_guidance_sampler(sde, x0, n_steps, q):
    target_norm = chi.ppf(0.5, df=x0.size)
    x0 = prior_rescaling(sde, x0, target_norm=target_norm) # We observed that staring at the median of logpT ensured consisted results of density guidance
    drift_fn = lambda x, l: density_guidance_ode_step(sde, l, x, q)[0]
    lambdas = jnp.linspace(sde.lambda_min, sde.lambda_max, n_steps)
    return euler_solver(drift_fn, x0, lambdas)

def stochastic_density_guidance_sampler(sde, x0, n_steps, noise_strength, q, key):
    target_norm = chi.ppf(0.5, df=x0.size)
    x0 = prior_rescaling(sde, x0, target_norm=target_norm)
    lambdas = jnp.linspace(sde.lambda_min, sde.lambda_max, n_steps)
    def sde_terms(state, l, noise):
        x, key = state
        noise_x, _ = noise
        new_key, old_key = jr.split(key)
        drift, score = density_guidance_ode_step(sde, l, x, q, noise_strength=noise_strength(l), key=old_key)
        projected_noise = noise_x - score * jnp.sum(score * noise_x)/jnp.sum(score * score)
        diffusion = noise_strength(l) * jnp.sqrt(sde.g2(l)) * projected_noise
        return (drift, new_key), (diffusion, None)
    state_key, solver_key = jr.split(key)
    state0 = (x0, state_key)
    final_x, final_key = euler_maruyama_solver(sde_terms, state0, lambdas, solver_key)
    return final_x

def indicator_fn(t, a, b):
    return jnp.where((t > a) & (t < b), 1., 0.)

if __name__ == "__main__":
    import equinox as eqx
    from utils import sde_from_checkpoint
    sde = sde_from_checkpoint("checkpoints/unweighted_model.eqx")
    key = jr.PRNGKey(5678)
    batch_size = 64
    n_ode_sampling_steps = 1024
    pf_encoding_fn = eqx.filter_jit(jax.vmap(lambda x, key: pf_ode_encoder_with_logp(sde, x, n_ode_sampling_steps, key)))

    prior_key, key = jr.split(key)
    latent = sde.sample_prior((batch_size, 3, 32, 32), prior_key)

    q_min = 0.001
    q_max = 0.999
    D = 3 * 32 * 32 # dimensionality of the data

    n_points = 6

    for target_norm in jnp.linspace(chi.ppf(q_min, df=D), chi.ppf(q_max, df=D), n_points):
        prior_guidance_sampling_fn = eqx.filter_jit(jax.vmap(lambda x: prior_guidance_sampler(sde, x, target_norm, n_steps=64)))
        samples = prior_guidance_sampling_fn(latent)
        ode_encoding_key, key = jr.split(key)
        encoded, logp_ode_encoded = pf_encoding_fn(samples, jr.split(ode_encoding_key, batch_size))
        mean_logp = jnp.mean(logp_ode_encoded).item()
        print(f"Prior guidance with q={chi.cdf(target_norm, df=D):g}; average logp: {mean_logp:.2f}", flush=True)

    for q in jnp.linspace(0.3, 0.7, n_points):
        density_guidance_sampling_fn = eqx.filter_jit(jax.vmap(lambda x: density_guidance_sampler(sde, x, n_steps=64, q=q)))
        samples = density_guidance_sampling_fn(latent)
        ode_encoding_key, key = jr.split(key)
        encoded, logp_ode_encoded = pf_encoding_fn(samples, jr.split(ode_encoding_key, batch_size))
        mean_logp = jnp.mean(logp_ode_encoded).item()
        print(f"Density guidance with q={q:g}; average logp: {mean_logp:.2f}", flush=True)

    n_points = 3

    early_noise_fn = lambda l: 0.2 * indicator_fn(l, -10, -4)
    late_noise_fn = lambda l: 0.3 * indicator_fn(l, -3, 9)

    for q in jnp.linspace(0.3, 0.7, n_points):
        stochastic_sampling_key, key = jr.split(key)
        early_noise_sdg_sampling_fn = eqx.filter_jit(jax.vmap(lambda x, key: stochastic_density_guidance_sampler(sde, x, n_steps=128, noise_strength=early_noise_fn, q=q, key=key)))
        late_noise_sdg_sampling_fn = eqx.filter_jit(jax.vmap(lambda x, key: stochastic_density_guidance_sampler(sde, x, n_steps=128, noise_strength=late_noise_fn, q=q, key=key)))
        early_noise_samples = early_noise_sdg_sampling_fn(latent, jr.split(stochastic_sampling_key, latent.shape[0]))
        late_noise_samples = late_noise_sdg_sampling_fn(latent, jr.split(stochastic_sampling_key, latent.shape[0]))
        ode_encoding_key, key = jr.split(key)
        encoded, logp_ode_encoded = pf_encoding_fn(early_noise_samples, jr.split(ode_encoding_key, batch_size))
        mean_logp = jnp.mean(logp_ode_encoded).item()
        print(f"Early noise stochastic density guidance with q={q:g}; average logp: {mean_logp:.2f}", flush=True)
        ode_encoding_key, key = jr.split(key)
        encoded, logp_ode_encoded = pf_encoding_fn(late_noise_samples, jr.split(ode_encoding_key, batch_size))
        mean_logp = jnp.mean(logp_ode_encoded).item()
        print(f"Late noise stochastic density guidance with q={q:g}; average logp: {mean_logp:.2f}", flush=True)
