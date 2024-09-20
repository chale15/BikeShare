library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(glmnet)
library(DataExplorer)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare")

train <- vroom("train.csv")
test <- vroom("test.csv")


#Clean Data

train <- train %>% mutate(count = log(count)) %>% select(-casual, -registered)


#Feature Engineering

recipe_1 <- recipe(count~., data = train) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              holiday=factor(holiday),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy"))) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_rm(datetime) %>% 
  step_mutate(datetime_hour=factor(datetime_hour)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())


model_1 <-linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

workflow_1 <- workflow() %>% 
  add_recipe(recipe_1) %>% 
  add_model(model_1)

tuning_param_grid <- grid_regular(penalty(), mixture(), levels = 10)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- workflow_1 %>% 
  tune_grid(resamples=folds,
            grid=tuning_param_grid,
            metrics=metric_set(rmse, mae, rsq))

collect_metrics(cv_results) %>% 
  filter(.metric=="rmse") %>% 
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) + geom_line()

bestTune <- cv_results %>% select_best(metric="rmse")

final_wf <- workflow_1 %>% finalize_workflow(bestTune) %>% 
  fit(data = train)

final_preds <- final_wf %>% predict(new_data = test)


#Format and Write for Kaggle Submission

cross_val_kaggle_submission <- final_preds %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=cross_val_kaggle_submission, file="./Submissions/CrossValPreds6.csv", delim=",")
