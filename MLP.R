library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)
library(stacks)
library(dbarts)
library(xgboost)
library(mgcv)
#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare")

train <- vroom("train.csv")
test <- vroom("test.csv")


#Clean Data

train <- train %>% mutate(count = log(count)) %>% select(-casual, -registered)

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

folds <- vfold_cv(train, v = 10, repeats=1)

mlp_model <- mlp(hidden_units = tune(),
                 penalty = tune(),
                 epochs = 100) %>% 
  set_engine("nnet") %>% 
  set_mode("regression") %>% 
  translate()

mlp_workflow <- workflow() %>% 
  add_recipe(recipe_1) %>% 
  add_model(mlp_model)

mlp_tuning_grid <- grid_regular(hidden_units(), penalty(), levels = 5)

untunedModel <- control_stack_grid() 

mlp_models <- mlp_workflow %>%
  tune_grid(resamples=folds,
            grid=mlp_tuning_grid,
            metrics=metric_set(rmse),
            control = untunedModel)

mlp_bestTune <- mlp_models %>% select_best(metric="rmse")

mlp_fit <- mlp_workflow %>% 
  finalize_workflow(mlp_bestTune) %>% 
  fit(data = train)

mlp_predict <- mlp_fit %>% predict(new_data = test)


mlp_kaggle_submission <- mlp_predict %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=mlp_kaggle_submission, file="./Submissions/MLPPreds3.csv", delim=",")
