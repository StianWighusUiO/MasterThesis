library(readr)
library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)
library(glmnet)
library(caret)
library(bnlearn)

set.seed(15052025)

time_start <- Sys.time()

train <- read_delim("mnist_train.csv", delim = ",", col_names = TRUE)
test <- read_delim("mnist_test.csv", delim = ",", col_names = TRUE)

n_levels <- 3

train <- train %>% 
  mutate(label = as.factor(label)) %>% 
  mutate(across(!all_of("label"), .fns = ~./255)) %>% 
  mutate(across(!all_of("label"), .fns = ~cut.default(.,
                                                      breaks = seq(0, 1, 1/n_levels),
                                                      labels = 1:n_levels,
                                                      include.lowest = TRUE,
                                                      right = TRUE,
                                                      ordered_result = FALSE))) %>% 
  as.data.frame()

test <- test %>% 
  mutate(label = as.factor(label)) %>% 
  mutate(across(!all_of("label"), .fns = ~./255)) %>% 
  mutate(across(!all_of("label"), .fns = ~cut.default(.,
                                                      breaks = seq(0, 1, 1/n_levels),
                                                      labels = 1:n_levels,
                                                      include.lowest = TRUE,
                                                      right = TRUE,
                                                      ordered_result = FALSE))) %>% 
  as.data.frame()


cat("Main data retrival and preprocessing:")
Sys.time()-time_start


# Naive Bayes
time_start <- Sys.time()
nb_structure <- naive.bayes(train, "label")
nb_model <- bn.fit(nb_structure, train, method = "bayes")
nb_pred <- predict(nb_model, test)
print(confusionMatrix(nb_pred, test$label))

cat("Naive Bayes:")
Sys.time() - time_start

# Tree-augmented naive Bayes
time_start <- Sys.time()
TAN_structure <- tree.bayes(train, "label")
TAN_model <- bn.fit(TAN_structure, train, method = "bayes")

TAN_pred <- predict(TAN_model, test)
print(confusionMatrix(TAN_pred, test$label))

cat("Tree-Augmented Naive Bayes:")
Sys.time() - time_start

plot_digit_gg <- function(vec) {
  mat <- matrix(vec, nrow = 28, byrow = TRUE)
  df <- as.data.frame(as.table(mat))
  colnames(df) <- c("y", "x", "value")
  
  ggplot(df, aes(x = as.numeric(x), y = as.numeric(y), fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black") +
    coord_fixed() +
    theme_void() +
    theme(legend.position = "none") +
    scale_y_reverse()
}

x <- test

full_list <- list(c())
for (j in 0:9) {
  y <- x %>% 
    filter(label == j)
  t <- y %>% 
    mutate_all(as.numeric) %>% 
    summarise_all(mean)
  temp1 <- list(plot_digit_gg(as.numeric(t[-1])))
  temp2 <- lapply(1:10, function(i) {
    plot_digit_gg(as.numeric(y[i, -1]))
  })
  full_list <- c(full_list, temp1, temp2)
}

plot_grid(plotlist = full_list[-1], nrow = 10)
