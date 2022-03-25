library(readr)
require(magrittr)
require(tidyverse)
library(MASS)

data_solution = read_csv("Logistic_solution.csv")
data_dual = read_csv("Logistic_dual.csv")
data_solution %<>% mutate(across(where(is.character), as.factor))
data_dual %<>% mutate(across(where(is.character), as.factor))
data_solution <- data_solution[,-1]
data_dual <- data_dual[,-1]

data.frame(colnames(data_solution))




mod1 = glm(data_solution$SOLUTIONS ~., family = "binomial", data = data_solution)
summary(mod1)
mod1_aic <- stepAIC(mod1, direction = "forward")
summary(mod1_aic)

mod2 = glm(data_dual$COMMODITY_DUAL~., family = "binomial", data = data_dual)
summary(mod2)
mod2_aic <- stepAIC(mod2, direction = "forward")
summary(mod2_aic)



mod3 = glm(SOLUTIONS ~., family = "binomial", data = data_solution[,c(2, 21, 22, 23)])
summary(mod3)
mod3_aic <- stepAIC(mod3, direction = "forward")
summary(mod3_aic)

mod4 = glm(COMMODITY_DUAL~., family = "binomial", data =  data_dual[,c(18,19,20,28)])
summary(mod4)
mod4_aic <- stepAIC(mod4, direction = "forward")
summary(mod4)
