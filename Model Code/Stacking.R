library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(rpart)
library(ranger)
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

recipe_2 <- recipe(count ~ ., data = train) %>% 
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


folds <- vfold_cv(train, v = 10, repeats=1)

untunedModel <- control_stack_grid() 
tunedModel <- control_stack_resamples()

#Penalized Regression Model
preg_model <- linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

preg_workflow <- workflow() %>% 
  add_recipe(recipe_1) %>% 
  add_model(preg_model)

preg_tuning_grid <- grid_regular(penalty(), mixture(), levels = 10)

preg_models <- preg_workflow %>%
  tune_grid(resamples=folds,
            grid=preg_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)


#Linear Regression Model
lreg <- linear_reg() %>% set_engine("lm")

lreg_workflow <- workflow() %>% 
  add_model(lreg) %>% 
  add_recipe(recipe_3)

lreg_model <- fit_resamples(lreg_workflow,
                            resamples=folds,
                            metrics=metric_set(rmse, mae, rsq),
                            control = tunedModel)



#Regression Tree
rt_model<- decision_tree(tree_depth=tune(),
                         cost_complexity = tune(),
                         min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

rt_workflow <- workflow() %>% 
  add_recipe(recipe_2) %>% 
  add_model(rt_model)

rt_tuning_grid <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels = 5)

rt_models <- rt_workflow %>% 
  tune_grid(resamples=folds,
            grid=rt_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)

bestTune <- rt_models %>% select_best(metric="rmse")

#Random Forest
rf_model <- rand_forest(mtry = tune(),
                       min_n = tune(),
                       trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_workflow <- workflow() %>% 
  add_recipe(recipe_2) %>% 
  add_model(rf_model)

rf_tuning_grid <- grid_regular(mtry(range=c(1, 9)), min_n(), levels = 5)

rf_models <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=rf_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)

#BART Model
bart_model <- parsnip::bart(trees = 1000) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression") %>% 
  translate()

bart_workflow <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(recipe_2)

bart_models <- fit_resamples(bart_workflow, 
                resamples=folds,
                metrics= metric_set(rmse, mae, rsq),
                control = tunedModel)

#Boosted Model
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

bt_models <- bt_workflow %>%
  tune_grid(resamples=folds,
            grid=bt_tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)

#MLP Model
mlp_model <- mlp(hidden_units = tune(),
                 penalty = tune(),
                 epochs = 500) %>% 
  set_engine("nnet") %>% 
  set_mode("regression") %>% 
  translate()

mlp_workflow <- workflow() %>% 
  add_recipe(recipe_1) %>% 
  add_model(mlp_model)

mlp_tuning_grid <- grid_regular(hidden_units(), penalty(), levels = 5)

mlp_models <- mlp_workflow %>%
  tune_grid(resamples=folds,
            grid=mlp_tuning_grid,
            metrics=metric_set(rmse),
            control = untunedModel)


#Stacked Model
my_stack <- stacks() %>% 
  add_candidates(lreg_model) %>% 
  add_candidates(rf_models) %>% 
  add_candidates(bart_models) %>% 
#  add_candidates(bt_models) %>% 
  add_candidates(mlp_models)

stack_model <- my_stack %>% 
  blend_predictions() %>% 
  fit_members()

#stack_data <- as_tibble(my_stack)

final_preds <- stack_model %>% predict(new_data=test)

#Kaggle Submission
stacking_kaggle_submission <- final_preds %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=stacking_kaggle_submission, file="./Submissions/StackedPreds38.csv", delim=",")





