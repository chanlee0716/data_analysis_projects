# Load packages
library(MASS)
library(ggplot2)
library(car)
library(lme4) # needed for modelling with random effects
library(tidyverse); library(dplyr); library(broom)
library(patchwork); library(gridExtra); library(grid)
library(janitor); library(splines)
library(glmnet)
library(caret)
library(pROC)

# Reading in the csv file
data <- read.csv("data/05b_analysis_file_update.csv")

# Preparing data
data$homelessness_rate <- data$pit_tot_hless_pit_hud / data$dem_pop_pop_census
data_2017 <- data[data$year == 2017, ]
data_2017 <- data_2017[!is.na(data_2017$homelessness_rate), ] # lose two rows

g1 <- ggplot(data = data_2017, aes(x = homelessness_rate)) +
  geom_density()

data_2017$homelessness_rate_log <- log(data_2017$homelessness_rate)
g2 <- ggplot(data = data_2017, aes(x = homelessness_rate_log)) +
  geom_density()


# 1) Small epsilon adjustment if needed (avoid exactly 0 or 1)
epsilon <- 1e-6
data_2017$homelessness_rate_adj <- 
  pmin(pmax(data_2017$homelessness_rate, epsilon), 1 - epsilon)

# 2) Logit transform
data_2017$homelessness_rate_logit <- log(
  data_2017$homelessness_rate_adj / (1 - data_2017$homelessness_rate_adj)
)
g3 <- ggplot(data = data_2017, aes(x = homelessness_rate_logit)) +
  geom_density()