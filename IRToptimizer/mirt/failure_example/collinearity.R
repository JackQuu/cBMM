# Load packages
library(mirt)

# 1. Create a normal dataset (500 persons, 5 items)
set.seed(123)
a <- matrix(rlnorm(5, 0, 0.2))
d <- matrix(rnorm(5))
dat <- simdata(a, d, 500, itemtype = '2PL')
colnames(dat) <- paste0("Item_", 1:5)

# 2. Construct perfectly duplicated columns
# Make Item_6 an exact duplicate of Item_5
dat_redundant <- cbind(dat, Item_6 = dat[, "Item_5"])

# Inspect first rows to confirm Item_5 and Item_6 are identical
head(dat_redundant[, c("Item_5", "Item_6")])

# 3. Try fitting the model
# Case A: normal data
message("--- Fitting normal data ---")
mod_normal <- mirt(dat, 1, itemtype = '2PL')

# Case B: data with duplicated columns
message("\n--- Fitting data with duplicated columns ---")
# Watch console output; warnings or errors often appear
mod_redundant <- mirt(dat_redundant, 1, itemtype = '2PL')

# 4. Check results
# Inspect parameter estimates for duplicated columns
print(coef(mod_redundant, simplify = TRUE)$items)

# Check whether the information matrix is valid (if singular, SE will be NA)
message("\n--- Checking standard errors (SE) ---")
print(head(data.frame(extract.mirt(mod_redundant, 'SE'))))
