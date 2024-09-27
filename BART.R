library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)
library(stacks)
library(dbarts)
#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare")

train <- vroom("train.csv")
test <- vroom("test.csv")


#Clean Data

train <- train %>% mutate(count = log(count)) %>% select(-casual, -registered)

recipe_3 <- recipe(count~., data = train) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              holiday=factor(holiday),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy"))) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_mutate(datetime_hour=factor(datetime_hour))

recipe_1 <- recipe(count~., data = train) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_rm(datetime) %>% 
  step_mutate(working_hour = workingday * datetime_hour) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              holiday=factor(holiday),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy")))%>%
  step_mutate(datetime_hour=factor(datetime_hour)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

recipe_2 <- recipe(count~., data = train) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_rm(datetime, holiday, temp) %>% 
  step_mutate(working_hour = workingday * datetime_hour) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy"))) %>%
  step_mutate(datetime_hour=factor(datetime_hour),
              datetime_dow = factor(datetime_dow))


# Model
bart_model <- bart2(formula = count~., data = baked) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

bart_workflow <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(recipe_2)

bart_fit <- bart_workflow %>% 
  fit(data = train)

bart_predict <- bart_fit %>% predict(new_data = test)

bart_kaggle_submission <- bart_predict %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=bart_kaggle_submission, file="./Submissions/BartPreds5.csv", delim=",")



