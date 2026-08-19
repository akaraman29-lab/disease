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
    out_degree_school, out_degree_workplace, out_degree_address, out_degree_family, out_degree_other_ex_family,
    C1_combined_numeric, C2_combined_numeric, Infector_share_school, Infector_share_workplace,
    Infector_share_address, C3_combined_numeric, C4_combined_numeric, C6_combined_numeric,
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
    
    holiday_school      = numpyro.sample("holiday_school", dist.Normal(0, 1).expand((n_school_holiday-1,)))
    holiday_school_full = numpyro.deterministic( "holiday_school_full", jnp.concatenate([jnp.array([0.0]), holiday_school]))
    
    holiday_workplace      = numpyro.sample("holiday_workplace", dist.Normal(0, 1).expand((n_school_holiday-1,)))
    holiday_workplace_full = numpyro.deterministic( "holiday_workplace_full", jnp.concatenate([jnp.array([0.0]), holiday_workplace]))
    
    holiday_address      = numpyro.sample("holiday_address", dist.Normal(0, 1).expand((n_school_holiday-1,)))
    holiday_address_full = numpyro.deterministic( "holiday_address_full", jnp.concatenate([jnp.array([0.0]), holiday_address]))
    
    holiday_family      = numpyro.sample("holiday_family", dist.Normal(0, 1).expand((n_school_holiday-1,)))
    holiday_family_full = numpyro.deterministic( "holiday_family_full", jnp.concatenate([jnp.array([0.0]), holiday_family]))
    
    holiday_other      = numpyro.sample("holiday_other", dist.Normal(0, 1).expand((n_school_holiday-1,)))
    holiday_other_full = numpyro.deterministic( "holiday_other_full", jnp.concatenate([jnp.array([0.0]), holiday_other]))
    
    # Priors for coefficients specific to each outcome
    beta_school    = numpyro.sample("beta_school", dist.Normal(0, 1).expand((6,)))  # 7 coefficients for school
    beta_workplace = numpyro.sample("beta_workplace", dist.Normal(0, 1).expand((6,)))  # 7 coefficients for workplace
    beta_address   = numpyro.sample("beta_address", dist.Normal(0, 1).expand((6,)))  # 7 coefficients for address
    beta_family   = numpyro.sample("beta_family", dist.Normal(0, 1).expand((6,)))  # 7 coefficients for family
    beta_other     = numpyro.sample("beta_other", dist.Normal(0, 1).expand((6,)))  # 7 coefficients for other
    
    # Priors for age_group (categorical) with reference group
    # Set the first group (index 0) as the reference group
    beta_age_group      = numpyro.sample("beta_age_group", dist.Normal(0, 1).expand((n_age_groups - 1,)))
    # Pad with 0 for the reference group
    beta_age_group_full = numpyro.deterministic( "beta_age_group_full", jnp.concatenate([jnp.array([0.0]), beta_age_group]))
    
    beta_vacc_status    = numpyro.sample("beta_vacc_status", dist.Normal(0, 1).expand((n_vacc_status - 1,)))
    # Pad with 0 for the reference group
    beta_vacc_status_full = numpyro.deterministic( "beta_vacc_status_full", jnp.concatenate([jnp.array([0.0]), beta_vacc_status]))
    
    # random effects, by setting
    sigma_group_school = numpyro.sample("sigma_group_school", dist.HalfNormal(1.0))
    
    with numpyro.plate("groups_school",n_regions):
      group_intercept_school = numpyro.sample("group_intercept_school", dist.Normal(0,sigma_group_school))
    
    sigma_group_workplace = numpyro.sample("sigma_group_workplace", dist.HalfNormal(1.0))
    
    with numpyro.plate("groups_workplace",n_regions):
      group_intercept_workplace = numpyro.sample("group_intercept_workplace", dist.Normal(0,sigma_group_workplace))
    
    sigma_group_address = numpyro.sample("sigma_group_address", dist.HalfNormal(1.0))
    
    with numpyro.plate("groups_address",n_regions):
      group_intercept_address = numpyro.sample("group_intercept_address", dist.Normal(0,sigma_group_address))
      
    sigma_group_family = numpyro.sample("sigma_group_family", dist.HalfNormal(1.0))
    
    with numpyro.plate("groups_family",n_regions):
      group_intercept_family = numpyro.sample("group_intercept_family", dist.Normal(0,sigma_group_family))
    
    sigma_group_other = numpyro.sample("sigma_group_other", dist.HalfNormal(1.0))
    
    with numpyro.plate("groups_other",n_regions):
      group_intercept_other = numpyro.sample("group_intercept_other", dist.Normal(0,sigma_group_other))

    # intercepts
    intercept_school    = numpyro.sample("intercept_school", dist.Normal(0,2))
    intercept_workplace = numpyro.sample("intercept_workplace", dist.Normal(0,2))
    intercept_address   = numpyro.sample("intercept_address", dist.Normal(0,2))
    intercept_family    = numpyro.sample("intercept_family", dist.Normal(0,2))
    intercept_other     = numpyro.sample("intercept_other", dist.Normal(0,2))
    
    # Priors for hurdle rates 
    hurdle_school    = numpyro.sample("hurdle_school", dist.Beta(1,1))
    hurdle_workplace = numpyro.sample("hurdle_workplace", dist.Beta(1,1))
    hurdle_address   = numpyro.sample("hurdle_address", dist.Beta(1,1))
    hurdle_family    = numpyro.sample("hurdle_family", dist.Beta(1,1))
    hurdle_other     = numpyro.sample("hurdle_other", dist.Beta(1,1))
    
    # Overdispersion
    log_alpha_school    = numpyro.sample("log_alpha_school", dist.Normal(0,1))
    log_alpha_workplace = numpyro.sample("log_alpha_workplace", dist.Normal(0,1))
    log_alpha_address   = numpyro.sample("log_alpha_address", dist.Normal(0,14))
    log_alpha_family    = numpyro.sample("log_alpha_family", dist.Normal(0,1))
    log_alpha_other     = numpyro.sample("log_alpha_other", dist.Normal(0,1))

    alpha_school    = numpyro.deterministic("alpha_school", jnp.exp(log_alpha_school))
    alpha_workplace = numpyro.deterministic("alpha_workplace", jnp.exp(log_alpha_workplace))
    alpha_address   = numpyro.deterministic("alpha_address", jnp.exp(log_alpha_address))
    alpha_family    = numpyro.deterministic("alpha_family", jnp.exp(log_alpha_family))
    alpha_other     = numpyro.deterministic("alpha_other", jnp.exp(log_alpha_other))

    # Linear predictor for out_degree_school
    mu_school_raw = (
        intercept_school +
        group_intercept_school[Regionskode] +
        beta_school[0] * C1_combined_numeric +
        beta_school[1] * C2_combined_numeric +
        beta_school[2] * C4_combined_numeric +
        beta_school[3] * C6_combined_numeric +
        beta_school[4] * C8_combined_numeric +
        beta_school[5] * H6_combined_numeric +
        holiday_school_full[is_school_holiday] +
        beta_variant_full[variant] +
        beta_vacc_status_full[vacc_status] +
        beta_age_group_full[age_group]   # Use the full beta_age_group with reference group
    )
    
    mu_school = numpyro.deterministic("mu_school", jnp.exp(mu_school_raw))
    
    numpyro.sample(
       "out_degree_school", 
       dist.ZeroInflatedNegativeBinomial2(
           gate = hurdle_school,
           mean = mu_school,
           concentration = alpha_school#[age_group]
       ), 
       obs=out_degree_school)
    
   # Linear predictor for out_degree_workplace
    mu_workplace_raw = (
        intercept_workplace +
        group_intercept_workplace[Regionskode] +
        beta_workplace[0] * C1_combined_numeric +
        beta_workplace[1] * C2_combined_numeric +
        beta_workplace[2] * C4_combined_numeric +
        beta_workplace[3] * C6_combined_numeric +
        beta_workplace[4] * C8_combined_numeric +
        beta_workplace[5] * H6_combined_numeric +
        holiday_workplace_full[is_school_holiday] +
        beta_variant_full[variant] +
        beta_vacc_status_full[vacc_status] +
        beta_age_group_full[age_group]    # Use the full beta_age_group with reference group
    )
    
    mu_workplace = numpyro.deterministic("mu_workplace", jnp.exp(mu_workplace_raw))
    
    numpyro.sample(
        "out_degree_workplace", 
        dist.ZeroInflatedNegativeBinomial2(
            gate = hurdle_workplace,
            mean = mu_workplace,
            concentration = alpha_workplace#[age_group]
        ), 
        obs=out_degree_workplace)
    
    # Linear predictor for out_degree_address
    mu_address_raw = (
        intercept_address +
        group_intercept_address[Regionskode] +
        beta_address[0] * C1_combined_numeric +
        beta_address[1] * C2_combined_numeric +
        beta_address[2] * C4_combined_numeric +
        beta_address[3] * C6_combined_numeric +
        beta_address[4] * C8_combined_numeric +
        beta_address[5] * H6_combined_numeric +
        holiday_address_full[is_school_holiday] +
        beta_variant_full[variant] +
        beta_vacc_status_full[vacc_status] +
        beta_age_group_full[age_group]   # Use the full beta_age_group with reference group
    )
    mu_address = numpyro.deterministic("mu_address", jnp.exp(mu_address_raw))
    
    prob_address = sigmoid(mu_address_raw)  # convert to (0,1), need to use mu_address_raw
        
    # Define a small epsilon for numerical stability 
    epsilon = 1e-6 
    
    # Calculate concentration1 and concentration0 with epsilon 
    c1 = (prob_address * alpha_address) + epsilon 
    c0 = ((1 - prob_address) * alpha_address) + epsilon 
    
    numpyro.sample(
        "out_degree_address",
        ZeroInflatedBetaBinomial(
            gate=hurdle_address,
            total_count=n_trials_address,
            concentration1=c1,
            concentration0=c0
        ),
        obs=out_degree_address)
        
        
    # Linear predictor for out_degree_family
    mu_family_raw = (
        intercept_family +
        group_intercept_family[Regionskode] +
        beta_family[0] * C1_combined_numeric +
        beta_family[1] * C2_combined_numeric +
        beta_family[2] * C4_combined_numeric +
        beta_family[3] * C6_combined_numeric +
        beta_family[4] * C8_combined_numeric +
        beta_family[5] * H6_combined_numeric +
        holiday_family_full[is_school_holiday] +
        beta_variant_full[variant] +
        beta_vacc_status_full[vacc_status] +
        beta_age_group_full[age_group]    # Use the full beta_age_group with reference group
    )
    
    mu_family = numpyro.deterministic("mu_family", jnp.exp(mu_family_raw))
    
    numpyro.sample(
        "out_degree_family", 
        dist.ZeroInflatedNegativeBinomial2(
            gate = hurdle_family,
            mean = mu_family,
            concentration = alpha_family
        ), 
        obs=out_degree_family)    
    
    # Linear predictor for out_degree_other
    mu_other_raw = (
        intercept_other + 
        group_intercept_other[Regionskode] +
        beta_other[0] * C1_combined_numeric +
        beta_other[1] * C2_combined_numeric +
        beta_other[2] * C4_combined_numeric +
        beta_other[3] * C6_combined_numeric +
        beta_other[4] * C8_combined_numeric +
        beta_other[5] * H6_combined_numeric +
        holiday_other_full[is_school_holiday] +
        beta_variant_full[variant] +
        beta_vacc_status_full[vacc_status] +
        beta_age_group_full[age_group]   # Use the full beta_age_group with reference group
    )
    mu_other = numpyro.deterministic("mu_other", jnp.exp(mu_other_raw))
    
    numpyro.sample(
        "out_degree_other", 
        dist.ZeroInflatedNegativeBinomial2(
            gate = hurdle_other,
            mean = mu_other, 
            concentration = alpha_other#[age_group]         
        ), 
        obs=out_degree_other_ex_family)
    
def jittered_init(site, num_samples=10):
    base_init = init_to_median(num_samples=10)(site)
    if hasattr(base_init, 'shape'):
        jitter = 0.1 * jax_random.normal(jax_random.PRNGKey(0), base_init.shape)
        return base_init + jitter
    return base_init


def main(input_file, num_warmup=1000, num_samples=2000, num_chains=4, thinning=10,output_file=None):
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
        "out_degree_school": jnp.array(data_rt_ml["out_degree_school"].to_numpy()),
        "out_degree_workplace": jnp.array(data_rt_ml["out_degree_workplace"].to_numpy()),
        "out_degree_address": jnp.array(data_rt_ml["out_degree_address"].to_numpy()),
        "out_degree_family": jnp.array(data_rt_ml["out_degree_family"].to_numpy()),
        "out_degree_other_ex_family": jnp.array(data_rt_ml["out_degree_other_ex_family"].to_numpy()),
        "C1_combined_numeric": jnp.array(data_rt_ml["C1_combined_numeric"].to_numpy()),
        "C2_combined_numeric": jnp.array(data_rt_ml["C2_combined_numeric"].to_numpy()),
        "Infector_share_school": jnp.array(data_rt_ml["Infector_share_school"].to_numpy()),
        "Infector_share_workplace": jnp.array(data_rt_ml["Infector_share_workplace"].to_numpy()),
        "Infector_share_address": jnp.array(data_rt_ml["Infector_share_address"].to_numpy()),
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

    
