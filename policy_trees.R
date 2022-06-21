# Susan Athey Causal Forests for CATE E[Y2 - Y1|X=x]: https://grf-labs.github.io/grf/
# Kallus Recursive Partitioning for Personalization using Observational Data: https://github.com/Aida-Rahmattalabi/PersonalizationTrees
# Nuti Explainable Bayesian Decision Tree Algorithm: https://github.com/UBS-IB/bayesian_tree
# Dunnhumby Grocery store Data: https://www.dunnhumby.com/source-files/
# Estimating Treatment Effects with Causal Forests: https://arxiv.org/pdf/1902.07409.pdf

install.packages("grf")           #generalized random forests
install.packages("tidyverse")
install.packages("readxl")
library(padr)
library(grf)
library(tidyr)
library(readxl)
library(tidyverse)

# clear memory
gc()

# import grocery data
setwd("C:/Users/mlentini/OneDrive - Edmund Optics, Inc/Documents/Rowan/Thesis/Notebooks/Dunnhumby Grocery Stores/dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey CSV")
data.demog <- read.csv('hh_demographic.csv')
data.trans <- read.csv('transaction_data.csv') # has 2 million rows
data.prods <- read.csv('product.csv')

# data preparation
# look at each shopping trip
# did customer purchase strawberries
# if yes, use sales value as price
# if not, impute price as mode of sales value

# mean price of strawberries
price.mean <- mean(
  as.numeric(
    unlist(
      data.trans %>% 
        inner_join(., data.prods, by = "PRODUCT_ID") %>% 
        filter(SUB_COMMODITY_DESC == "STRAWBERRIES") %>% 
        select(SALES_VALUE) 
    )
  )
)

# joins and transformations
data <- data.trans %>% 
  inner_join(., data.demog, by = "household_key") %>%
  inner_join(., data.prods, by = "PRODUCT_ID") %>%
  group_by(BASKET_ID, AGE_DESC, MARITAL_STATUS_CODE, INCOME_DESC, HOMEOWNER_DESC, HH_COMP_DESC, HOUSEHOLD_SIZE_DESC, KID_CATEGORY_DESC, household_key) %>%
  summarise(
    purchased = case_when("STRAWBERRIES" %in% SUB_COMMODITY_DESC ~ 1, TRUE ~ 0),
    price = dplyr::case_when("STRAWBERRIES" %in% SUB_COMMODITY_DESC ~ SALES_VALUE, TRUE ~ price.mean)
  ) 


# check for missing values
any(is.na(data))

write.csv(data, "cleaned_strawberries.csv")

############################### Example taken from https://grf-labs.github.io/grf/ ############################### 
# Generate data.
n <- 2000
p <- 10
X <- matrix(rnorm(n * p), n, p) # n x p matrix generated from a normal distribution
X.test <- matrix(0, 101, p) # 101 by p matrix all 0s
X.test[, 1] <- seq(-2, 2, length.out = 101) # change 1st column to linear space from -2 to 2 with 101 data points

# Train a causal forest.
W <- rbinom(n, 1, 0.4 + 0.2 * (X[, 1] > 0)) # sample from binomial distribution n times, sample size = 1, prob = 0.4 or 0.6. Data should be roughly balanced, check table(W)
Y <- pmax(X[, 1], 0) * W + X[, 2] + pmin(X[, 3], 0) + rnorm(n) # pmax gives which ever value is greater, 0 or the value in the vector
tau.forest <- causal_forest(X, Y, W) # gives features' importance

# X: Covariates used in causal forest
# Y: The outcome (must be a numeric vector with no NAs)
# W: The treatment assignment (must be binary or real numeric vector with no NAs)

# Estimate treatment effects for the training data using out-of-bag prediction.
tau.hat.oob <- predict(tau.forest)
hist(tau.hat.oob$predictions)

# Estimate treatment effects for the test sample.
tau.hat <- predict(tau.forest, X.test)
plot(X.test[, 1], tau.hat$predictions, ylim = range(tau.hat$predictions, 0, 2), xlab = "x", ylab = "tau", type = "l")
lines(X.test[, 1], pmax(0, X.test[, 1]), col = 2, lty = 2) # the treatment effect

# Estimate the conditional average treatment effect on the full sample (CATE).
average_treatment_effect(tau.forest, target.sample = "all")

# Estimate the conditional average treatment effect on the treated sample (CATT).
average_treatment_effect(tau.forest, target.sample = "treated")

# Add confidence intervals for heterogeneous treatment effects; growing more trees is now recommended.
tau.forest <- causal_forest(X, Y, W, num.trees = 4000)
tau.hat <- predict(tau.forest, X.test, estimate.variance = TRUE)
sigma.hat <- sqrt(tau.hat$variance.estimates)
plot(X.test[, 1], tau.hat$predictions, ylim = range(tau.hat$predictions + 1.96 * sigma.hat, tau.hat$predictions - 1.96 * sigma.hat, 0, 2), xlab = "x", ylab = "tau", type = "l")
lines(X.test[, 1], tau.hat$predictions + 1.96 * sigma.hat, col = 1, lty = 2)
lines(X.test[, 1], tau.hat$predictions - 1.96 * sigma.hat, col = 1, lty = 2)
lines(X.test[, 1], pmax(0, X.test[, 1]), col = 2, lty = 1)

# In some examples, pre-fitting models for Y and W separately may
# be helpful (e.g., if different models use different covariates).
# In some applications, one may even want to get Y.hat and W.hat
# using a completely different method (e.g., boosting).

# Generate new data.
n <- 4000
p <- 20
X <- matrix(rnorm(n * p), n, p)
TAU <- 1 / (1 + exp(-X[, 3]))
W <- rbinom(n, 1, 1 / (1 + exp(-X[, 1] - X[, 2])))
Y <- pmax(X[, 2] + X[, 3], 0) + rowMeans(X[, 4:6]) / 2 + W * TAU + rnorm(n)

forest.W <- regression_forest(X, W, tune.parameters = "all")
W.hat <- predict(forest.W)$predictions

forest.Y <- regression_forest(X, Y, tune.parameters = "all")
Y.hat <- predict(forest.Y)$predictions

forest.Y.varimp <- variable_importance(forest.Y)

# Note: Forests may have a hard time when trained on very few variables
# (e.g., ncol(X) = 1, 2, or 3). We recommend not being too aggressive
# in selection.
selected.vars <- which(forest.Y.varimp / mean(forest.Y.varimp) > 0.2)

tau.forest <- causal_forest(X[, selected.vars], Y, W,
                            W.hat = W.hat, Y.hat = Y.hat,
                            tune.parameters = "all")

# Check whether causal forest predictions are well calibrated.
test_calibration(tau.forest)

############################################### Causal Forest Dunnhumby Grocery Data ######################################################

# split into 50% train 50% test
# each row is a shopping trip. if an item was purchased in a shopping trip assign 1 otherwise 0. 
# when item is not purchased there is not price. The price can be imputed using the average or mode of previous transactions
# predict increase in revenue using causal forests


X <- data %>% select(-SALES_VALUE)
W <- data$SALES_VALUE # price is the treatment
Y <- 
