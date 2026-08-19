# Load libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import validate_sample
from numpyro.distributions.util import promote_shapes # Useful for broadcasting
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.infer import init_to_median, init_to_sample, init_to_value
from jax.nn import sigmoid
from jax import random as jax_random # Renamed to avoid conflict if user has 'random' variable
import jax.numpy as jnp
import pandas as pd
import numpy as np
import pickle
import argparse
import arviz as az
import gc

class ZeroInflatedBetaBinomial(dist.Distribution):
    arg_constraints = {
        "gate": constraints.unit_interval,      # Probability of structural zero (psi)
        "total_count": constraints.nonnegative_integer,
        "concentration1": constraints.positive, # Alpha for BetaBinomial
        "concentration0": constraints.positive, # Beta for BetaBinomial
    }
    support = constraints.nonnegative_integer
    has_rsample = False # BetaBinomial itself doesn't have rsample

    def __init__(self, gate, total_count, concentration1, concentration0, validate_args=None):
        """
        Zero-Inflated Beta-Binomial distribution.
        Args:
            gate (Tensor): Probability of structural zero component (often denoted psi).
            total_count (Tensor): Number of trials.
            concentration1 (Tensor): alpha parameter of the Beta distribution.
            concentration0 (Tensor): beta parameter of the Beta distribution.
        """
        self.gate, self.total_count, self.concentration1, self.concentration0 = promote_shapes(
            gate, total_count, concentration1, concentration0
        )
        
        self._beta_binomial_dist = dist.BetaBinomial(
            concentration1=self.concentration1,
            concentration0=self.concentration0,
            total_count=self.total_count
        )
        
        super().__init__(
            batch_shape=self._beta_binomial_dist.batch_shape, # Broadcasting handled by promote_shapes and base dist
            event_shape=self._beta_binomial_dist.event_shape,
            validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        key_bern, key_bb = jax_random.split(key)
        
        # gate is Pr(structural zero). is_structural_zero = 1 if it's a structural zero.
        is_structural_zero = dist.Bernoulli(probs=self.gate).sample(key_bern, sample_shape)
        
        # Samples from the BetaBinomial component
        # These samples will have shape: sample_shape + batch_shape
        beta_binom_samples = self._beta_binomial_dist.sample(key_bb, sample_shape)
        
        return jnp.where(is_structural_zero == 1, 0, beta_binom_samples)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Log probability from the BetaBinomial component
        log_prob_beta_binom = self._beta_binomial_dist.log_prob(value)

        # log(gate) = log(P(structural zero))
        # log(1-gate) = log(P(not structural zero))
        log_gate = jnp.log(self.gate)
        log_one_minus_gate = jnp.log1p(-self.gate) # Numerically stable log(1-p)

        # For observations where value == 0:
        # log( P(structural_zero) + P(not_structural_zero) * P(BetaBinom=0) )
        # log( gate + (1-gate) * P(BetaBinom=0) )
        # This is logaddexp(log_gate, log_one_minus_gate + log_prob_beta_binom_at_0)
        # Note: log_prob_beta_binom already contains the P(BetaBinom=value), so when value is 0, it's P(BetaBinom=0)
        log_prob_if_zero = jnp.logaddexp(log_gate, log_one_minus_gate + log_prob_beta_binom)

        # For observations where value > 0:
        # log( P(not_structural_zero) * P(BetaBinom=value) )
        log_prob_if_not_zero = log_one_minus_gate + log_prob_beta_binom
        
        return jnp.where(value == 0, log_prob_if_zero, log_prob_if_not_zero)

    @property
    def mean(self):
        # E[Y] = (1 - gate) * E[BetaBinomial]
        return (1 - self.gate) * self._beta_binomial_dist.mean

    @property
    def variance(self):
        # Var[Y] = (1 - gate) * Var(BetaBinomial) + gate * (1 - gate) * (E[BetaBinomial])^2
        e_bb = self._beta_binomial_dist.mean
        var_bb = self._beta_binomial_dist.variance
        return (1 - self.gate) * var_bb + self.gate * (1 - self.gate) * jnp.square(e_bb)



# Overall model definition
def model(
    out_degree,
    C1_combined_numeric, C2_combined_numeric, C3_combined_numeric, C4_combined_numeric, C6_combined_numeric,
    C8_combined_numeric, H6_combined_numeric, H8_combined_numeric, age_group, vacc_status,
    weeks_since_vacc, variant, Regionskode, is_school_holiday,
    n_variants, n_age_groups, n_vacc_status, n_weeks_since_vacc, n_regions, n_school_holiday, n_trials_address
):
    variant          = jnp.asarray(variant, dtype=jnp.int32)
    age_group        = jnp.asarray(age_group, dtype=jnp.int32)
    vacc_status      = jnp.asarray(vacc_status, dtype=jnp.int32)
    weeks_since_vacc = jnp.asarray(weeks_since_vacc, dtype=jnp.int32)
    
    # Priors for variant (categorical)
    beta_variant      = numpyro.sample("beta_variant", dist.Normal(0, 1).expand((n_variants-1,)))
    beta_variant_full = numpyro.deterministic( "beta_variant_full", jnp.concatenate([jnp.array([0.0]), beta_variant]))
    
    holiday      = numpyro.sample("holiday", dist.Normal(0, 1).expand((n_school_holiday-1,)))
    holiday_full = numpyro.deterministic( "holiday_full", jnp.concatenate([jnp.array([0.0]), holiday]))
    
    # Priors for coefficients specific to each outcome
    beta    = numpyro.sample("beta_school", dist.Normal(0, 1).expand((6,)))  
    
    # Priors for age_group (categorical) with reference group
    # Set the first group (index 0) as the reference group
    beta_age_group      = numpyro.sample("beta_age_group", dist.Normal(0, 1).expand((n_age_groups - 1,)))
    # Pad with 0 for the reference group
    beta_age_group_full = numpyro.deterministic( "beta_age_group_full", jnp.concatenate([jnp.array([0.0]), beta_age_group]))
    
    beta_vacc_status    = numpyro.sample("beta_vacc_status", dist.Normal(0, 1).expand((n_vacc_status - 1,)))
    # Pad with 0 for the reference group
    beta_vacc_status_full = numpyro.deterministic( "beta_vacc_status_full", jnp.concatenate([jnp.array([0.0]), beta_vacc_status]))
    
    # random effects, by setting
    sigma_group = numpyro.sample("sigma_group", dist.HalfNormal(1.0))
    
    with numpyro.plate("groups",n_regions):
      group_intercept = numpyro.sample("group_intercept", dist.Normal(0,sigma_group))
    
    # intercepts
    intercept    = numpyro.sample("intercept", dist.Normal(0,2))
    
    # Priors for hurdle rates 
    hurdle    = numpyro.sample("hurdle", dist.Beta(1,1))
    
    # Overdispersion
    log_alpha    = numpyro.sample("log_alpha", dist.Normal(0,1))
    
    alpha    = numpyro.deterministic("alpha", jnp.exp(log_alpha))
    
    # Linear predictor for out_degree_school
    mu_raw = (
        intercept +
        group_intercept[Regionskode] +
        beta[0] * C1_combined_numeric +
        beta[1] * C2_combined_numeric +
        beta[2] * C4_combined_numeric +
        beta[3] * C6_combined_numeric +
        beta[4] * C8_combined_numeric +
        beta[5] * H6_combined_numeric +
        holiday_full[is_school_holiday] +
        beta_variant_full[variant] +
        beta_vacc_status_full[vacc_status] +
        beta_age_group_full[age_group]   # Use the full beta_age_group with reference group
    )
    mu = numpyro.deterministic("mu", jnp.exp(mu_raw))
    
    numpyro.sample(
       "out_degree", 
       dist.ZeroInflatedNegativeBinomial2(
           gate = hurdle,
           mean = mu,
           concentration = alpha
       ), 
       obs=out_degree)
    
   
def jittered_init(site, num_samples=10):
    base_init = init_to_median(num_samples=10)(site)
    if hasattr(base_init, 'shape'):
        jitter = 0.1 * jax_random.normal(jax_random.PRNGKey(0), base_init.shape)
        return base_init + jitter
    return base_init


def main(input_file, num_warmup=1000, num_samples=2000, num_chains=4, thinning=10,output_file='mcmc_object_zibb_total_transmission_re'):
    print(f"input_file: {input_file}")

    data_rt_ml                     = pd.read_csv(input_file)

    # variant
    custom_order_variant           = ['wildtype','Alpha','Delta','Eta','Omicron']
    cat_variant                    = pd.Categorical(data_rt_ml['variant'],categories=custom_order_variant)
    data_rt_ml['variant']          = cat_variant.codes

    # Regionskode
    custom_order_region            = [1081,1082,1083,1084,1085]
    cat_region                     = pd.Categorical(data_rt_ml['Regionskode'],categories=custom_order_region)
    data_rt_ml['Regionskode']     = cat_region.codes
    
    # age group
    cat_age                        = pd.Categorical(data_rt_ml['age_group'])
    cat_age                        = cat_age.reorder_categories(cat_age.categories[::-1])
    data_rt_ml['age_group']        = cat_age.codes

    # vaccination status
    custom_order_vacc_status       = ['unvaccinated','first_vacc','second_vacc']
    cat_vacc_status                = pd.Categorical(data_rt_ml['vacc_status'],categories=custom_order_vacc_status)
    data_rt_ml['vacc_status']      = cat_vacc_status.codes

    # time since vaccination
    custom_order_weeks_since_vacc  = ['0w','<3w','3-10w','10-18w','>18w']
    cat_weeks_since_vacc           = pd.Categorical(data_rt_ml['weeks_since_vacc'],categories=custom_order_weeks_since_vacc)
    data_rt_ml['weeks_since_vacc'] = cat_weeks_since_vacc.codes

    n_variants = int(np.max(data_rt_ml["variant"].to_numpy()) +1)
    n_regions = int(np.max(data_rt_ml["Regionskode"].to_numpy()) +1)
    n_age_groups = int(np.max(data_rt_ml["age_group"].to_numpy()) +1)
    n_vacc_status = int(np.max(data_rt_ml["vacc_status"].to_numpy()) +1)
    n_weeks_since_vacc = int(np.max(data_rt_ml["weeks_since_vacc"].to_numpy()) +1) 
    n_school_holiday = int(np.max(data_rt_ml["is_school_holiday"].to_numpy()) +1) 

    # Data preparation
    data = {
        "out_degree": jnp.array(data_rt_ml["out_degree"].to_numpy()), 
        "C1_combined_numeric": jnp.array(data_rt_ml["C1_combined_numeric"].to_numpy()),
        "C2_combined_numeric": jnp.array(data_rt_ml["C2_combined_numeric"].to_numpy()),
        "C3_combined_numeric": jnp.array(data_rt_ml["C3_combined_numeric"].to_numpy()),
        "C4_combined_numeric": jnp.array(data_rt_ml["C4_combined_numeric"].to_numpy()),
        "C6_combined_numeric": jnp.array(data_rt_ml["C6_combined_numeric"].to_numpy()),
        "C8_combined_numeric": jnp.array(data_rt_ml["C8_combined_numeric"].to_numpy()),
        "H6_combined_numeric": jnp.array(data_rt_ml["H6_combined_numeric"].to_numpy()),
        "H8_combined_numeric": jnp.array(data_rt_ml["H8_combined_numeric"].to_numpy()),
        "age_group": jnp.array(data_rt_ml["age_group"].to_numpy(),dtype=jnp.int32),
        "vacc_status": jnp.array(data_rt_ml["vacc_status"].to_numpy(),dtype=jnp.int32),
        "weeks_since_vacc": jnp.array(data_rt_ml["weeks_since_vacc"].to_numpy(),dtype=jnp.int32),
        "variant": jnp.array(data_rt_ml["variant"].to_numpy(),dtype=jnp.int32),
        "Regionskode": jnp.array(data_rt_ml["Regionskode"].to_numpy(),dtype=jnp.int32),
        "is_school_holiday": jnp.array(data_rt_ml["is_school_holiday"].to_numpy(),dtype=jnp.int32),
        "n_variants": n_variants,
        "n_age_groups": n_age_groups,
        "n_vacc_status": n_vacc_status,
        "n_weeks_since_vacc": n_weeks_since_vacc,
        "n_regions": n_regions,
        "n_school_holiday": n_school_holiday,
        "n_trials_address": jnp.array(data_rt_ml["hsize_small"].to_numpy(),dtype=jnp.int32)
    }

    # Running the MCMC
    numpyro.set_host_device_count(num_chains)
    #rng_key = jax_random.split(jax_random.PRNGKey(0),num_chains)  # Give each chain a truly independent seed
    rng_key = jax_random.PRNGKey(0)
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=50))         # Add small noise to break symmetry and avoid exact same starting points
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, thinning=thinning, num_chains=num_chains,chain_method='parallel')
    mcmc.run(rng_key, **data)

    del data
    gc.collect()

    summary = az.summary(mcmc, hdi_prob = 0.95)
    summary.to_csv(f"{output_file}_summary.csv")

    inference_data = az.from_numpyro(mcmc)
    inference_data.to_netcdf(f"{output_file}_results.nc")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multivariate Poisson/Negative Binomial regression model.")
    parser.add_argument("input_file", default="data_rt_ml.csv" ,help="Path to input CSV file")
    parser.add_argument("--num_warmup", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of chains")
    parser.add_argument("--thinning", type=int, default=10, help="Thinning parameter")
    parser.add_argument("--output_file", default="mcmc_object_rt_ml", help="Path to save output (optional)")
    
    args = parser.parse_args()
    
    main(
        input_file=args.input_file,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        output_file=args.output_file
    )

    
