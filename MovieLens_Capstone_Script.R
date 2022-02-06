#################################
# 1. Downloading & Preparing Data
#################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(recommenderlab)
library(ggplot2)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#####################
# 2. Data Exploration
#####################

# Checking the number of unique users that provided ratings and how many unique movies were rated
edx %>% summarize(n_users = n_distinct(userId),n_movies = n_distinct(movieId))

# Plotting top 20 movies
top_20 <- edx %>% group_by(title) %>% summarize(count=n()) %>% top_n(20,count) %>% arrange(desc(count))

top_20 %>%
  ggplot(aes(x=reorder(title, count), y=count)) +
  geom_bar(stat="identity",color= "purple") + coord_flip(y=c(0, 40000)) +
  labs(x="", y="Number of ratings") +
  geom_text(aes(label= count), hjust=-0.1, size=3) +
  labs(title="Top 20 movies title based \n on number of ratings")

# Exploring the number of ratings by movieId
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "green") + 
  scale_x_log10() + 
  ggtitle("Exploring Movies") + 
  labs(x= "movieId", y= "Number of Ratings")

# Exploring the number of ratings by userId
edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "orange") + 
  scale_x_log10() + 
  ggtitle("Exploring Users") + 
  labs(x= "userId", y= "Number of Ratings")

# Exploring the number of ratings by userId
edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "orange") + 
  scale_x_log10() + 
  ggtitle("Exploring Users") +
  labs(x= "userId", y= "Number of Ratings")

######################################
# 3. Preparing to build the Algorithm
#####################################

# RMSE Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Test & Training Sets
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

######################
# 4. Building Models
######################

# Baseline Model
mu_baseline<- mean(train_set$rating)                                                           # Average Rating
baseline_RMSE<- RMSE(test_set$rating, mu_baseline)                                             # Baseline RMSE
RMSE_results <- tibble(Method = "Baseline Model", RMSE = baseline_RMSE)                              # Create a table with RMSE's
RMSE_results

# Movie Effect Model
mu_least_squares <- mean(train_set$rating) 

LSE_rating_movies<- train_set %>% group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_least_squares))

LSE_prediction_movies<- mu_least_squares + test_set %>% 
  left_join(LSE_rating_movies, by='movieId') %>% .$b_i

LSE_RMSE_movies <- RMSE(LSE_prediction_movies, test_set$rating)

RMSE_results <- bind_rows(RMSE_results, tibble(Method="Movie Effect Model", RMSE = LSE_RMSE_movies ))
RMSE_results %>% knitr::kable()

# User + Movie Effect
LSE_rating_users <- train_set %>% left_join(LSE_rating_movies, by='movieId') %>% 
  group_by(userId) %>% summarize(b_u = mean(rating - mu_least_squares - b_i))

LSE_prediction_users <- test_set %>% left_join(LSE_rating_movies, by='movieId') %>% 
  left_join(LSE_rating_users, by='userId') %>% mutate(pred = mu_least_squares + b_i + b_u) %>% .$pred

LSE_RMSE_users <- RMSE(LSE_prediction_users, test_set$rating)
RMSE_results <- bind_rows(RMSE_results, tibble(Method="Movie + User Effect Model", RMSE = LSE_RMSE_users ))

RMSE_results %>% knitr::kable()

# Regularization
lambdas <- seq(0, 10, 0.25)
best_lambda <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, best_lambda)
lambda <- lambdas[which.min(best_lambda)]
lambda

RMSE_results <- bind_rows(RMSE_results,
                          tibble(Method="Regularized Movie + User Effect Model",  
                                 RMSE = min(best_lambda)))

RMSE_results %>% knitr::kable()

# Matrix Factorization

# Calculating Movie Effect to be used in Matrix Factorization
b_i <- edx %>% group_by(movieId) %>%     
  summarize(b_i = sum(rating - mu_least_squares)/(n()+lambda))

# Calculating Movie + User Effect to be used in Matrix Factorization
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu_least_squares)/(n()+lambda))

# Adding the Residuals to "edx"
edx_residual <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(residual = rating - mu_least_squares - b_i - b_u) %>%
  select(userId, movieId, residual)

# Preparing the datasets to be used with recommenderlab packagae 
write.table(train_set , file = "trainset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(test_set , file = "testset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)

edx_mf <- as.matrix(edx_residual)
validation_mf <- validation %>% select(userId, movieId, rating)
validation_mf<- as.matrix(validation_mf)

write.table(validation_mf , file = "validation.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
train_set_mf<- data_file("trainset.txt")
test_set_mf<- data_file("testset.txt")
validation_mf<- data_file("validation.txt")

# Building Recommender object & tuning it's parameters
r<- Reco()
tuning_mf <- r$tune(train_set_mf, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                              costp_l1 = 0, costq_l1 = 0,
                                              nthread = 1, niter = 10))

# Training the Recommender model
r$train(train_set_mf, opts = c(tuning_mf$min, nthread = 1, niter = 20))


# Making prediction on the test_set % calculating RMSE
pred_file <- tempfile()
r$predict(test_set_mf, out_file(pred_file))
predicted_residuals_mf <- scan(pred_file)
mf_RMSE <- RMSE(predicted_residuals_mf, test_set$rating)
RMSE_results <- bind_rows(RMSE_results, tibble(Method="Matrix Factorization Model",  RMSE = mf_RMSE))

RMSE_results %>% knitr::kable()

##############################################
# 5. Final model performance on validation set
##############################################

# Applying the best performing model on the validation set & calculating RMSE

# Preparing Validation Set
validation_rating <- read.table("validation.txt", header = FALSE, sep = " ")$V3

# Making predictions on the validation set
r$predict(validation_mf, out_file(pred_file))
predicted_residuals_mf_validation <- scan(pred_file)

# Calculating RMSE
mf_RMSE_validation <- RMSE(predicted_residuals_mf_validation, validation_rating)
RMSE_results <- bind_rows(RMSE_results, tibble(Method="Validation using Matrix Factorization",  RMSE = mf_RMSE_validation))

options(digits = 8)
RMSE_results %>% knitr::kable()