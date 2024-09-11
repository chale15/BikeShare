library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(patchwork)
library(GGally)

#Import Training and Testing Data
train <- vroom("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/train.csv")
test <- vroom("~/Desktop/Fall 2024/Stat 348/GitHubRepos/BikeShare/test.csv")

#Refactor
train$season <- factor(train$season, levels = c("Spring", "Summer", "Fall", "Winter"))
train$weather <- factor(train$weather, levels = c("Sunny", "Cloudy", "Light Rain", "Heavy Rain"))


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

