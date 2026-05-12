library(mirt)

# Build simulated data
# Item_A: normal item
# Item_B: contains NA; remaining scores are all 1 (may trigger errors)
set.seed(123)
n_people <- 10
sim_data <- data.frame(
  Item_A = sample(c(0, 1), n_people, replace = TRUE),
  Item_B = c(1, 1, NA, 1, 1, NA, 1, 1, 1, 1)  # only 1 and NA
)

print("Simulated data:")
print(sim_data)

# Fitting the model
tryCatch({
  fit <- mirt(sim_data, 1)
}, error = function(e) {
  cat("\nCaught error:\n", e$message, "\n")
})
