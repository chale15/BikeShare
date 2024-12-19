library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)
library(GGally)
library(poissonreg)
library(lubridate)
library(glmnet)
library(car)
library(ggfortify)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare")

train <- vroom("train.csv")
test <- vroom("test.csv")


#Clean Data

train <- train %>% mutate(count = log(count)) %>% select(-casual, -registered)


#Feature Engineering

bike_recipe <- recipe(count~., data = train) %>% 
  step_mutate(season=factor(season, labels=c("Spring","Summer","Fall","Winter")),
              holiday=factor(holiday),
              workingday=factor(workingday),
              weather= factor(ifelse(weather==4,3,weather), labels=c("Sunny","Cloudy","Rainy"))) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features="hour") %>% 
  step_mutate(datetime_hour=factor(datetime_hour))


#Create and Execute Workflow

lin_model_1 <-linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

workflow_1 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(lin_model_1) %>% 
  fit(data = train)

lin_preds_1 <- predict(workflow_1, new_data = test)


#Format and Write for Kaggle Submission

recipe_kaggle_submission <- lin_preds_1 %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=recipe_kaggle_submission, file="recipePreds2.csv", delim=",")


#Feature Engineering 2

bike_recipe2 <- recipe(count~., data = train) %>% 
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


#Create and Execute Workflow

penalized_model_1 <-linear_reg(penalty = .05, mixture = .5) %>% 
  set_engine("glmnet")

penalized_workflow_1 <- workflow() %>% 
  add_recipe(bike_recipe2) %>% 
  add_model(penalized_model_1) %>% 
  fit(data = train)

penalized_preds_1 <- predict(penalized_workflow_1, new_data = test)


#Format and Write for Kaggle Submission

recipe_kaggle_submission <- penalized_preds_1 %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=recipe_kaggle_submission, file="PenalizedPreds3.csv", delim=",")


#Selecting Lambda values

prepped_recipe <- prep(bike_recipe2)
train1 <- bake(prepped_recipe, new_data=train)
train1 <- train1 %>% mutate(count1 = count) %>% select(-count) %>% mutate(count=count1) %>% select(-count1)

train_x <- as.matrix(train1[, 1:40]) # predictors
train_y <- as.matrix(train1[, 41]) # response

set.seed(50)
train_ridge_cv <- cv.glmnet(x = train_x, 
                          y = train_y, 
                          type.measure = "mse", 
                          alpha = 0)

train_lasso_cv <- cv.glmnet(x = train_x, 
                            y = train_y, 
                            type.measure = "mse", 
                            alpha = 1)

train_en_cv <- cv.glmnet(x = train_x, 
                            y = train_y, 
                            type.measure = "mse", 
                            alpha = .5)


lambda_ridge <- 0.06053913

lambda_lasso <- 0.00056459

lambda_en <- 0.001028867
nu_en <- 0.5

#Kaggle Submission Ridge Regression

penalized_model_2 <-linear_reg(penalty = lambda_ridge, mixture = 0) %>% 
  set_engine("glmnet")

penalized_workflow_2 <- workflow() %>% 
  add_recipe(bike_recipe2) %>% 
  add_model(penalized_model_2) %>% 
  fit(data = train)

penalized_preds_2 <- predict(penalized_workflow_2, new_data = test)


#Format and Write for Kaggle Submission

recipe_kaggle_submission <- penalized_preds_2 %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=recipe_kaggle_submission, file="PenalizedPreds4.csv", delim=",")


#Kaggle Submission LASSO

penalized_model_3 <-linear_reg(penalty = lambda_lasso, mixture = 1) %>% 
  set_engine("glmnet")

penalized_workflow_3 <- workflow() %>% 
  add_recipe(bike_recipe2) %>% 
  add_model(penalized_model_3) %>% 
  fit(data = train)

penalized_preds_3 <- predict(penalized_workflow_3, new_data = test)


#Format and Write for Kaggle Submission

recipe_kaggle_submission <- penalized_preds_3 %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=recipe_kaggle_submission, file="PenalizedPreds5.csv", delim=",")

#Kaggle Submission Elastic Net

penalized_model_4 <-linear_reg(penalty = lambda_en, mixture = 0.5) %>% 
  set_engine("glmnet")

penalized_workflow_4 <- workflow() %>% 
  add_recipe(bike_recipe2) %>% 
  add_model(penalized_model_4) %>% 
  fit(data = train)

penalized_preds_4 <- predict(penalized_workflow_4, new_data = test)


#Format and Write for Kaggle Submission

recipe_kaggle_submission <- penalized_preds_4 %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=recipe_kaggle_submission, file="PenalizedPreds6.csv", delim=",")

#Kaggle Submission Guessing

penalized_model_5 <-linear_reg(penalty = .0015, mixture = 0.85) %>% 
  set_engine("glmnet")

penalized_workflow_5 <- workflow() %>% 
  add_recipe(bike_recipe2) %>% 
  add_model(penalized_model_5) %>% 
  fit(data = train)

penalized_preds_5 <- predict(penalized_workflow_5, new_data = test)


#Format and Write for Kaggle Submission

recipe_kaggle_submission <- penalized_preds_5 %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=recipe_kaggle_submission, file="PenalizedPreds12.csv", delim=",")
