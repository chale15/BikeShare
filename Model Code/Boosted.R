library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)
library(stacks)
library(dbarts)
library(xgboost)

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

bt_model <- boost_tree(mtry = tune(),
                        min_n = tune(),
                        trees = 500,
                       learn_rate = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression") %>% 
  translate()

bt_workflow <- workflow() %>% 
  add_recipe(recipe_1) %>% 
  add_model(bt_model)

bt_tuning_grid <- grid_regular(mtry(range=c(1, 9)), min_n(), learn_rate(), levels = 5)

untunedModel <- control_stack_grid() 

bt_models <- bt_workflow %>%
  tune_grid(resamples=folds,
            grid=bt_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)

bt_bestTune <- bt_models %>% select_best(metric="rmse")

bt_fit <- bt_workflow %>% 
  finalize_workflow(bt_bestTune) %>% 
  fit(data = train)

bt_predict <- bt_fit %>% predict(new_data = test)



bt_kaggle_submission <- bt_predict %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=bt_kaggle_submission, file="./Submissions/BoostedPreds1.csv", delim=",")

