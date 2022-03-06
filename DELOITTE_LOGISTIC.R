library(readr)
Logistic_commodity <- read_csv("Logistic_commodity.csv")
View(Logistic_commodity)


mod1 = glm(COMMODITY_DUAL~., family = "binomial", data = Logistic_commodity)
summary(mod1)
exp(coef(mod1))
exp(coef(mod1)[1]+1.632923e-01*1) / (1+ exp(coef(mod1)[1] + 1.632923e-01*1)) 
Logistic_solutions <- read_csv("Logistic_solutions.csv")

View(Logistic_solutions)
mod2 = glm(SOLUTIONS~., family = "binomial", data = Logistic_solutions)
summary(mod2)
