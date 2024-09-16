library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)
library(GGally)
library(poissonreg)
library(lubridate)

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

lin_model_1 <-linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

workflow_1 <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(lin_model_1) %>% 
  fit(data = train)

lin_preds_1 <- predict(workflow_1, new_data = test)

recipe_kaggle_submission <- lin_preds_1 %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=recipe_kaggle_submission, file="recipePreds2.csv", delim=",")

