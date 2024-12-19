library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)
library(GGally)
library(poissonreg)
library(lubridate)

#Import Training and Testing Data
train <- vroom("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/train.csv")
test <- vroom("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/test.csv")

#Wrangling
train$season <- as.factor(train$season)
levels(train$season) <- c("Spring", "Summer", "Fall", "Winter")
train$weather <- as.factor(train$weather)
levels(train$weather) <- c("Sunny", "Cloudy", "Light Rain", "Heavy Rain")
train$workingday <- factor(train$workingday)
train$holiday <- factor(train$holiday)
train <- train %>% select(-casual, -registered)
train$dayOfWeek <- wday(train$datetime, label=TRUE)

test$season <- as.factor(test$season)
levels(test$season) <- c("Spring", "Summer", "Fall", "Winter")
test$weather <- as.factor(test$weather)
levels(test$weather) <- c("Sunny", "Cloudy", "Light Rain", "Heavy Rain")
test$workingday <- factor(test$workingday)
test$holiday <- factor(test$holiday)
test$dayOfWeek <- wday(test$datetime, label=TRUE)


#EDA Plots
weather_bar <- ggplot(data = train, mapping = aes(x = weather)) + 
  geom_bar(fill = "deepskyblue") + 
  labs(title = "Weather Bar Plot",x = "Type of Weather", y="Count")

temp_count <- ggplot(data = train, mapping = aes(x = temp, y = count)) +
  geom_point(size=1, colour = "deepskyblue4", alpha = 0.8) + geom_smooth(se=FALSE, color = "red", lwd = 1) +
  labs(title = "Bike Rentals by Temperature", x = "Temperature (Celsius)", y = "Count")

count_over_time <- ggplot(data = train, mapping = aes(x = datetime, y = count)) +
  geom_point(size=1, colour = "forestgreen", alpha = 0.6) + geom_smooth(se=FALSE, color = "red", lwd = 1) +
  labs(title = "Bike Rentals over Time", x = "Date (Year-Month)", y = "Count")

count_by_season <- ggplot(data = train, mapping = aes(x=season, y = count)) + 
  geom_boxplot(fill = "darkgreen", alpha = .7) + 
  labs(title = "Bike Rental by Season", x = "Season", y = "Count")

four_plot <- (weather_bar + temp_count)/(count_over_time + count_by_season)

ggsave("four_plots.jpg", plot = four_plot, path = "~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/")


#Linear Regression
my_lm <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(data = train, formula = count ~ .)

lm_predict <- predict(my_lm, new_data= test)

#lm_predict

#For Kaggle Submission
lin_kaggle_submission <- lm_predict %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=pmax(0, count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=lin_kaggle_submission, file="~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/linearPredsFactor.csv", delim=",")


#Poisson Regression
pois_model <- poisson_reg() %>% 
  set_engine("glm") %>% 
  set_mode("regression") %>% 
  fit(data = train, formula = count ~ .)

pois_predict <- predict(pois_model, new_data = test)

#pois_predict

#For Kaggle Submission

pois_kaggle_submission <- pois_predict %>% 
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count=.pred) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=pois_kaggle_submission, file="~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/poissonPreds1.csv", delim=",")


#Linear Regression on log(count)

log_lm <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(data = train, formula = log(count) ~ .)

log_lm_predict <- predict(log_lm, new_data= test)

#For kaggle submission

log_lin_kaggle_submission <- log_lm_predict %>% 
  bind_cols(., test) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count=exp(count)) %>%
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=log_lin_kaggle_submission, file="~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/log_linearPreds2.csv", delim=",")

