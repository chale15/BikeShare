library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(glmnet)
library(DataExplorer)
library(rpart)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare")

train <- vroom("train.csv")
test <- vroom("test.csv")


#Clean Data

train <- train %>% mutate(count = log(count)) %>% select(-casual, -registered)

model_2 <- decision_tree(tree_depth=tune(),
                         cost_complexity = tune(),
                         min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

recipe_2 <- recipe(count~., data = train) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_rm(datetime, holiday, temp) %>% 
  step_mutate(working_hour = workingday * datetime_hour) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
#              holiday=factor(holiday),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy"))) %>%
  step_mutate(datetime_hour=factor(datetime_hour),
              datetime_dow = factor(datetime_dow)) #%>%
  #step_dummy(all_nominal_predictors()) %>% 
  #step_normalize(all_numeric_predictors())

prepped <- prep(recipe_2)
baked <- bake(prepped, new_data = train)

workflow_2 <- workflow() %>% 
  add_recipe(recipe_2) %>% 
  add_model(model_2)

tuning_param_grid <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- workflow_2 %>% 
  tune_grid(resamples=folds,
            grid=tuning_param_grid,
            metrics=metric_set(rmse))

bestTune <- cv_results %>% select_best(metric="rmse")

final_wf <- workflow_2 %>% finalize_workflow(bestTune) %>% 
  fit(data = train)

final_preds <- final_wf %>% predict(new_data = test)


#Format and Write for Kaggle Submission

cross_val_kaggle_submission <- final_preds %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=cross_val_kaggle_submission, file="./Submissions/TreePreds9.csv", delim=",")

