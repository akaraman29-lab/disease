// utility functions
functions {
  int num_zeros(int[] y) {
    int sum = 0;
    for (n in 1:size(y))
      sum += (y[n]==0);
    return sum;
  }
}


// The input data is a vector 'y' of length 'N'.
data {
    int<lower=0> N;        // number of observations
    int<lower=0> y[N];     // observed counts
}

transformed data {
  int<lower=0> N_zero = num_zeros(y);
  array[N-N_zero] int<lower=1> y_nonzero;
  int N_nonzero = 0;
  
  for (n in 1:N) {
    if (y[n] == 0) continue;
    N_nonzero += 1;
    y_nonzero[N_nonzero] = y[n];
  }
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
    real<lower=0, upper=1> theta; 
    real<lower=0> lambda; 
}

transformed parameters {
  real<lower=0> R; 
  
  R = (1-theta) * lambda;
}

// The model to be estimated. We model the output
model {
    // Priors (can be adjusted depending on prior knowledge)
    lambda ~ normal(0, 5);          
    
    // Likelihood
    target += N_zero * log_sum_exp(log(theta),log1m(theta)+poisson_lpmf(0|lambda));
    target += N_nonzero * log1m(theta);
    target += poisson_lpmf(y_nonzero|lambda);
}

