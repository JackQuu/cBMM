library(mirt)

# Simulated data generation
set.seed(123)
n_items <- 100
n_people <- 20

# Generate true discrimination (a) and difficulty (d) parameters
# Note: mirt uses intercept form; typically P(x=1) = g + (1-g) / (1 + exp(-(a*theta + d)))
true_a <- matrix(rlnorm(n_items, meanlog = 0, sdlog = 0.3))
true_d <- matrix(rnorm(n_items, mean = 0, sd = 1))

# Generate simulated response matrix with simdata()
# Here we simulate a unidimensional (Nfact = 1) 2PL model
data <- simdata(a = true_a, d = true_d, N = n_people, itemtype = '2PL')

# Fit a unidimensional model (1 factor)
# '2PL' specifies the item parameterization for each item
model_2PL <- mirt(data, 1, itemtype = '2PL')
