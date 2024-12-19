library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)
library(stacks)
library(dbarts)
library(recipes)
#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare")

train <- vroom("train.csv")
test <- vroom("test.csv")


#Clean Data

train_casual <- train %>% mutate(casual = log(casual+0.0001)) %>% select(-count, -registered)
train_registered <- train %>% mutate(registered = log(registered+0.0001)) %>% select(-count, -casual)

recipe_casual <- recipe(casual ~ ., data = train_casual) %>% 
  step_date(datetime, features = c("year", "dow")) %>% 
  step_time(datetime, features = "hour") %>% 
  step_mutate(working_hour = workingday * datetime_hour) %>% 
  step_rm(datetime, holiday, temp) %>% 
  step_mutate(
    season = factor(season, labels = c("Spring", "Summer", "Fall", "Winter")),
    workingday = factor(workingday),
    weather = factor(ifelse(weather == 4, 3, weather), labels = c("Sunny", "Cloudy", "Rainy")),
    datetime_hour = factor(datetime_hour),
    datetime_dow = factor(datetime_dow),
    datetime_year = factor(datetime_year))

recipe_registered <- recipe(registered ~ ., data = train_registered) %>% 
  step_date(datetime, features = c("year", "dow")) %>% 
  step_time(datetime, features = "hour") %>% 
  step_mutate(working_hour = workingday * datetime_hour) %>% 
  step_rm(datetime, holiday, temp) %>% 
  step_mutate(
    season = factor(season, labels = c("Spring", "Summer", "Fall", "Winter")),
    workingday = factor(workingday),
    weather = factor(ifelse(weather == 4, 3, weather), labels = c("Sunny", "Cloudy", "Rainy")),
    datetime_hour = factor(datetime_hour),
    datetime_dow = factor(datetime_dow),
    datetime_year = factor(datetime_year))

# Model
bart_model <- parsnip::bart(trees = 1000) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression") %>% 
  translate()

bart_workflow_casual <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(recipe_casual)

bart_workflow_registered <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(recipe_registered)

bart_fit_cas <- bart_workflow_casual %>% 
  fit(data = train_casual)

bart_fit_reg <- bart_workflow_registered %>% 
  fit(data = train_registered)

bart_predict_cas <- bart_fit_cas %>% predict(new_data = test)
bart_predict_reg <- bart_fit_reg %>% predict(new_data = test)

bart_predict_joined <- exp(bart_predict_cas) + exp(bart_predict_reg) + 0.0002

bart_kaggle_submission <- bart_predict_joined %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=bart_kaggle_submission, file="./Submissions/SplitPreds2.csv", delim=",")
