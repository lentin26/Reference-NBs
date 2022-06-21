# Susan Athey Causal Forests for CATE E[Y2 - Y1|X=x]: https://grf-labs.github.io/grf/
# Kallus Recursive Partitioning for Personalization using Observational Data: https://github.com/Aida-Rahmattalabi/PersonalizationTrees
# Nuti Explainable Bayesian Decision Tree Algorithm: https://github.com/UBS-IB/bayesian_tree
# Dunnhumby Grocery store Data: https://www.dunnhumby.com/source-files/
# Estimating Treatment Effects with Causal Forests: https://arxiv.org/pdf/1902.07409.pdf

install.packages("grf")           #generalized random forests
install.packages("tidyverse")
install.packages("readxl")
install.packages("ggcorrplot")

rm(list = ls())

library(grf)
library(tidyr)
library(tidyverse)
library(dplyr)
library(grf)
library(binr)
library(timereg)
library(policytree)
library(DiagrammeR)
library(data.table)
library(ggcorrplot)

# Synthetic dataset 1 - linear probit model with no confounding
set.seed(39)
n <- 5 # samples
h <- -1
X0 <- rnorm(n, 5, 1)
X1 <- rnorm(n, 5, 1)
g <- X0
P <- rnorm(n, 5, 1)
e <- rnorm(n, 0, 1)

# descrete price
t = as.numeric(quantile(P, probs=seq(0.1, 0.9, length.out=9)))
P.descrete <- sample(
  t, 
  size = n, 
  replace = TRUE,
  pnorm(seq(0.1, 0.9, length.out=9), 5, 1)
)

# binary indicator variable for purchase
Y <- as.numeric(g + h*P.descrete + e > 0) 

# 1vA binary treatment effects with causal forest
X <- data.frame(X0, X1)
W <- as.numeric(P.descrete == t[1])
R <- Y*P.descrete
tau.forest <- causal_forest(X, R, W)

# Estimate treatment effects for the training data using out-of-bag prediction.
tau.hat.oob <- predict(tau.forest)
hist(tau.hat.oob$predictions)

# maybe do this instead on 1vA?
multi.forest <- grf::multi_arm_causal_forest(X, R, as.factor(P))

# assign best treatment
# Compute doubly robust scores.
dr.scores <- double_robust_scores(multi.forest)
head(dr.scores)

# Fit a depth-2 tree on the doubly robust scores.
tree <- policy_tree(X, dr.scores, depth = 3)
plot(tree)

# assign treatments
predict <- predict(tree, X)
P.CT <- tree$action.names[predict]
Y.CT <- as.numeric(g + h*P.CT + e > 0) 

# revenue
R.CT <- sum(Y.CT*P.CT)/length(Y.CT)
print(c(R.CT, R.optimal))

################# Causal Trees Revenue ################# 
Predict.CT.Revenue <- function(data, k){
  e <- unlist(data %>% select(e), use.names=FALSE)
  R <- unlist(data %>% select(R), use.names=FALSE)
  P <- unlist(data %>% select(P), use.names=FALSE)
  h <- unlist(data %>% select(h), use.names=FALSE)
  g <- unlist(data %>% select(g), use.names=FALSE)
  X <- data %>% select(-c(R, P, Y, g, h, e))
  
  multi.forest <- grf::multi_arm_causal_forest(X, R, as.factor(P))
  
  # assign best treatment
  # Compute doubly robust scores.
  dr.scores <- double_robust_scores(multi.forest)
  
  # Fit a depth-2 tree on the doubly robust scores.
  tree <- policy_tree(X, dr.scores, depth = k)
  
  # assign treatments
  predict <- predict(tree, X)
  P.CT <- tree$action.names[predict]
  Y.CT <- as.numeric(g + h*P.CT + e > 0) 
  
  # revenue
  R.CT <- sum(Y.CT*P.CT)/length(Y.CT)
  
  # optimal revenue based on underlying probability distribution
  x <- data %>% group_by(P) %>% summarise(Mean = mean(Y, na.rm=TRUE))
  R.optimal <- max(x$P*x$Mean)
  
  return(list(R.optimal, R.CT))
}
################# Causal Trees Revenue #################

# dataset 1 - linear probit model with no confounding
generate.dataset.1 <- function(n){
  h <- -1
  X0 <- rnorm(n, 5, 1)
  X1 <- rnorm(n, 5, 1)
  g <- X0
  P <- rnorm(n, 5, 1)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  x <- data.frame(P) %>% group_by(cut(P, 9)) %>% mutate(P.descrete = min(P))
  P.descrete <- x$P.descrete
  
  # binary indicator variable for purchase
  Y <- as.numeric(g + h*P.descrete + e > 0) 
  
  # 1vA binary treatment effects with causal forest
  X <- data.frame(X0, X1)
  R <- Y*P.descrete
  #Y <- as.factor(Y)
  P <- P.descrete
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X, P, R, Y, g, h, e)) 
}

# Dataset 2 - higher dimension probit model with sparse linear interaction
generate.dataset.2 <- function(n){
  set.seed(NULL)
  p <- 20
  X <- matrix(rnorm(n*p), n, p)
  g <- 5
  beta <- matrix(rep(c(rnorm(5), rep(0, 15)), n), n, p)
  
  matrix(rnorm(20), 1, 20) %>% matrix(c(rnorm(5), rep(0, 15)))
  
  h <- -1.5*X %*% t(beta)
  P <- rnorm(n, 0, 2)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  x <- data.frame(P) %>% group_by(cut(P, 9)) %>% mutate(P.descrete = min(P))
  P <- x$P.descrete
  
  # binary indicator variable for purchase
  Y.star <- g + h*P + e
  Y <- as.numeric(Y.star > 0) 
  
  # data 
  X <- data.frame(X)
  R <- Y*P
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X, P, R, Y, g, h, e)) 
}

# dataset 3 probit model with step interaction
generate.dataset.3 <- function(n){
  set.seed(NULL)
  X0 <- rnorm(n, 0, 1)
  X1 <- rnorm(n, 0, 1)
  g <- 5
  
  h <- case_when(
    X0 < -1 ~ -1.2,
    (-1 <= X0 & X0 < 0) ~ -1.1,
    (0 <= X0 & X0 < 1) ~ -0.9,
    (1 <= X0 ~ -0.8)
    )

  P <- rnorm(n, X0 + 5, 2)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  x <- data.frame(P) %>% group_by(cut(P, 9)) %>% mutate(P.descrete = min(P))
  P <- x$P.descrete
  
  # binary indicator variable for purchase
  Y.star <- g + h*P + e
  Y <- as.numeric(Y.star > 0) 
  
  # data 
  R <- Y*P
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X0, X1, P, R, Y, g, h, e)) 
}

# dataset 4 probit model with multidimensional step interaction
generate.dataset.4 <- function(n){
  set.seed(NULL)
  X0 <- rnorm(n, 0, 1)
  X1 <- rnorm(n, 0, 1)
  g <- 5
  
  h <- case_when(
    X0 < -1 ~ -1.25,
    (-1 <= X0 & X0 < 0) ~ -1.1,
    (0 <= X0 & X0 < 0.1) ~ -0.9,
    (1 <= X0) ~ 0.75,
    (X1 < 0) ~ -0.1,
    (0 <= X1) ~ 0.1
  )
  
  P <- rnorm(n, X0 + 5, 2)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  x <- data.frame(P) %>% group_by(cut(P, 9)) %>% mutate(P.descrete = min(P))
  P <- x$P.descrete
  
  # binary indicator variable for purchase
  Y.star <- g + h*P + e
  Y <- as.numeric(Y.star > 0) 
  
  # data 
  R <- Y*P
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X0, X1, P, R, Y, g, h, e)) 
}

# dataset 5 linear probit model with observed confounding
generate.dataset.5 <- function(n){
  set.seed(NULL)
  X0 <- rnorm(n, 5, 1)
  X1 <- rnorm(n, 5, 1)
  g <- X0
  
  h <- -1
  
  P <- rnorm(n, X0 + 5, 2)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  x <- data.frame(P) %>% group_by(cut(P, 9)) %>% mutate(P.descrete = min(P))
  P <- x$P.descrete
  
  # binary indicator variable for purchase
  Y.star <- g + h*P + e
  Y <- as.numeric(Y.star > 0) 
  
  # data 
  R <- Y*P
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X0, X1, P, R, Y, g, h, e)) 
}

# dataset 6 probit model with non-linear interaction
generate.dataset.6 <- function(n){
  set.seed(NULL)
  X0 <- rnorm(n, 0, 1)
  X1 <- rnorm(n, 0, 1)
  
  g <- abs(X0 + X1)
  h <- -abs(X0 + X1)
  
  P <- rnorm(n, X0 + 5, 2)
  e <- rnorm(n, 0, 1)
  
  # descrete price
  x <- data.frame(P) %>% group_by(cut(P, 9)) %>% mutate(P.descrete = min(P))
  P <- x$P.descrete
  
  # binary indicator variable for purchase
  Y.star <- g + h*P + e
  Y <- as.numeric(Y.star > 0) 
  
  # data 
  R <- Y*P
  
  # covariates, treatment, outcome (revenue)
  return(data.frame(X0, X1, P, R, Y, g, h, e)) 
}
#################### visualize ####################
# datasets
data <- generate.dataset.1(5000)
data <- generate.dataset.2(5000)
data <- generate.dataset.3(5000)
data <- generate.dataset.4(5000)
data <- generate.dataset.5(5000)
data <- generate.dataset.6(5000)

# overall, customers more likely to not buy
hist(data$Y)
mean(data$Y)

# demand curve
data %>% group_by(P) %>% summarise(Mean = mean(Y, na.rm=TRUE)) %>%
  ggplot(aes(x=P, y=Mean)) + geom_line()

# revenue curve
data %>% group_by(P) %>% summarise(Mean = mean(Y, na.rm=TRUE)) %>%
  ggplot(aes(x=P, y=P*Mean)) + geom_line()

# optimal revenue from data
x <- data %>% group_by(P) %>% summarise(Mean = mean(Y, na.rm=TRUE))
max(x$P*x$Mean)

data$optimal_price <- -data$g/data$h
mean(data$optimal_price)

# customers less likely to buy if prices are higher 
data %>% ggplot(aes(x=as.factor(Y), y=P)) + geom_boxplot(color="red", fill="orange", alpha=0.2)

#  box plot price assignments
data %>% group_by(P) %>% summarise(Count = n(), na.rm=TRUE) %>% 
  ggplot(aes(x=as.factor(round(P,1)), y=Count)) + geom_bar(stat="identity")

# price assignment
data %>% ggplot(aes(x=X0, y=X1, color=P)) + geom_point()

# correlation plot
ggcorrplot(cor(data %>% select(-c(g, h, e, R))))

# sample from generative dataset 10 times and aveage compute optimal 
results <- c()
for (i in 1:10){
  data <- generate.dataset.2(5000)
  x <- data %>% group_by(P) %>% summarise(Mean = mean(Y, na.rm=TRUE))
  results <- c(results, max(x$P*x$Mean))
}
mean(results)
##################### results ##################### 
get.results <- function(n, dataset.generator){
  # function to find min, max and mean
  multi.fun <- function(x) {
    c(min = min(x), mean = mean(x), max = max(x))
  }
  
  # generate and test multiple samples
  result = list()
  for (i in 1:10){
    data <- dataset.generator(n)
    result <- rbindlist(list(result, Predict.CT.Revenue(data, k = 5)))
  }
  
  # summarize and result results
  colnames(result) <- c('Optimal', 'CT')
  return(sapply(result, multi.fun))
}

# get results
get.results(10, generate.dataset.1)
get.results(100, generate.dataset.2)


############## delete ############## 
data <- generate.dataset.1(100)

# covariates X, treatment P, outcome R
R <- unlist(data %>% select(R), use.names=FALSE)
P <- unlist(data %>% select(P), use.names=FALSE)
X <- data %>% select(-c(R, P, Y, g, h, e))

# train tree
k <- 3
multi.forest <- grf::multi_arm_causal_forest(X, R, as.factor(P))
dr.scores <- double_robust_scores(multi.forest)
tree <- policy_tree(X, dr.scores, depth = k)

# plot
plot(tree)
