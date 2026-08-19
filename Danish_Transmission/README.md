## Danish_Transmission_NPIs_Cov19
Code repository for inferring transmission networks from Danish SARS-CoV-2 viral genome data and modelling the impact of NPIs in Denmark.

## Authors
* Jacob-Curran Sebastian (<jaccur@itu.dk>)
* Christian Morgenstern (<c.morgenstern@imperial.ac.uk>)

## Age-structure matrices
The folder age_matrices contains median age-structure matrices for sampled infector-infectee pairs in our analysis. Those age matrices with the suffix "_1_year" correspond to the age groupings presented in the manuscript (and below), but we also include age groups of 5 and 10 years, which are given in these folders. 

![plot](./Figures/age_matrices_all_settings_granular.png)

## Packages and Dependencies

Analysis conducted using Python version 3.11.13 using the following packages:

- Numpy v2.3.3
- Pandas v2.2.3
- tqdm v4.67.1
- biopython v1.85
- regex v2.5.161
- hammingdist v1.3.0
- polars v1.33.1
- geopandas v1.1.1
- jupyter v1.1.1
- matplotlib v3.10.6
- scipy v1.16.2
- seaborn v0.13.2
- networkx v3.5
 
Typical install time < 1 minute
