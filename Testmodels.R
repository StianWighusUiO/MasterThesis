library(readr)
library(dplyr)
library(purrr)
library(glmnet)
library(caret)
library(bnlearn)
library(rsample)
library(parallel)
library(tidyr)
library(doMC)
library(ggplot2)

set.seed(15052025)

time_start <- Sys.time()

train <- read_delim("zip.train", delim = " ", col_names = FALSE)
test <- read_delim("zip.test", delim = " ", col_names = FALSE)

new_colnames <- c("label", 
                  paste0("X", rep(1:16, each = 16), "x", rep(1:16, times = 16)))

train <- train %>% 
  select(!X258) %>% 
  rename_with(~ new_colnames, everything()) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~round((.+1)/2))) %>% 
  mutate_all(.funs = ~as.factor(.)) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~factor(., levels = c("0", "1")))) %>% 
  as.data.frame()

test <- test %>% 
  rename_with(~ new_colnames, everything()) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~round((.+1)/2))) %>% 
  mutate_all(.funs = ~as.factor(.)) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~factor(., levels = c("0", "1")))) %>% 
  as.data.frame()

cat("Main data retrival and preprocessing:\t", Sys.time()-time_start, "\n")
rm(list = c("new_colnames", "time_start"))

# Naive Bayes
time_start <- Sys.time()
nb_structure <- naive.bayes(train, "label")
nb_model <- bn.fit(nb_structure, train, method = "bayes")

nb_pred <- predict(nb_model, test)
print(confusionMatrix(nb_pred, test$label))

cat("Naive Bayes:\t", Sys.time() - time_start, "\n")

# Tree-augmented naive Bayes
time_start <- Sys.time()
TAN_structure <- tree.bayes(train, "label")
TAN_model <- bn.fit(TAN_structure, train, method = "bayes")

TAN_pred <- predict(TAN_model, test)
print(confusionMatrix(TAN_pred, test$label))

cat("Tree-Augmented Naive Bayes:\t", Sys.time() - time_start, "\n")

# Logistic regression with no interaction, lasso penalty
time_start <- Sys.time()
x_train <- model.matrix(label ~ ., train)[, -1]
y_train <- as.numeric(as.character(train$label))

x_test <- model.matrix(label ~ ., test)[, -1]
y_test <- as.numeric(as.character(test$label))

logit_cv <- cv.glmnet(x_train,
                      y_train, 
                      family = "multinomial",
                      alpha = 1
)

logit_pred <- predict(logit_cv$glmnet.fit, 
                      newx = x_test,
                      s = logit_cv$lambda.1se,
                      type = "class"
  
)

conf_matrix_logit <- confusionMatrix(as.factor(logit_pred), as.factor(y_test))
print(conf_matrix_logit)

cat("Logistic regression, no interactions:\t", Sys.time() - time_start, "\n")

# Logistic regression with TAN interactions
time_start <- Sys.time()
x <- as.data.frame(TAN_structure$arcs)
x <- filter(x, from != "label" & to != "label")
x <- map2_chr(x$from, x$to, \(x, y) paste0(x, ":", y))
formula_chr <- paste0("label ~ . + ", paste(x, collapse = " + "))

x_TANinter_train <- model.matrix(as.formula(formula_chr), train)[, -1]
y_TANinter_train <- as.numeric(as.character(train$label))

x_TANinter_test <- model.matrix(as.formula(formula_chr), test)[, -1]
y_TANinter_test <- as.numeric(as.character(test$label))

logit_TANinter_cv <- cv.glmnet(x_TANinter_train, 
                               y_TANinter_train, 
                               family = "multinomial", 
                               alpha = 1
)

logit_TANinter_pred <- predict(logit_TANinter_cv$glmnet.fit, 
                               newx = x_TANinter_test, 
                               s = logit_TANinter_cv$lambda.1se, 
                               type = "class"
)

conf_matrix_logit_TANinter <- confusionMatrix(as.factor(logit_TANinter_pred), as.factor(y_TANinter_test))
print(conf_matrix_logit_TANinter)

cat("Logistic regression, TAN interactions:\t", Sys.time() - time_start, "\n")

# Logistic regression with pairwise interactions
time_start <- Sys.time()
x_inter_train <- model.matrix(label ~ .^2, train)[, -1]
y_inter_train <- as.numeric(as.character(train$label))

x_inter_test <- model.matrix(label ~ .^2, test)[, -1]
y_inter_test <- as.numeric(as.character(test$label))

logit_inter_cv <- cv.glmnet(x_inter_train, 
                            y_inter_train, 
                            family = "multinomial", 
                            alpha = 1
)

logit_inter_pred <- predict(logit_inter_cv$glmnet.fit, 
                            newx = x_inter_test, 
                            s = logit_inter_cv$lambda.1se,
                            type = "class"
)

conf_matrix_logit_inter <- confusionMatrix(as.factor(logit_inter_pred), as.factor(y_inter_test))
print(conf_matrix_logit_inter)

cat("Logistic regression, all pariwise interactions:\t", Sys.time() - time_start, "\n")


# Convergence

# Precompute model matrices for logistic regression
x_full <- model.matrix(label ~ ., train)[, -1]
# y_full <- as.numeric(as.character(train$label))

run_models <- function(train) {
  # test <- test[test_idx, names(train)]
  
  
  # Naive Bayes
  nb_structure <- naive.bayes(train, "label")
  nb_model <- bn.fit(nb_structure, train, method = "bayes")
  nb_pred <- predict(nb_model, test)
  acc_NB <- mean(nb_pred == test$label)
  
  # TAN
  TAN_structure <- tree.bayes(train, "label")
  TAN_model <- bn.fit(TAN_structure, train, method = "bayes")
  TAN_pred <- predict(TAN_model, test[, ])
  acc_TAN <- mean(TAN_pred == test$label)
  
  # Logistic no interactions
  x_logit_train <- model.matrix(as.formula(formula_chr), train)[, -1]
  y_logit_train <- as.numeric(as.character(train$label))
  
  x_logit_test <- model.matrix(as.formula(formula_chr), test[])[, -1]
  y_logit_test <- as.numeric(as.character(test$label))
  
  logit_model <- glmnet(x_logit_train, 
                                 y_logit_train, 
                                 family = "multinomial", 
                                 alpha = 1, 
                                 lambda = logit_cv$lambda
  )
  
  logit_pred <- predict(logit_model, 
                                 newx = x_logit_test,
                                 s = logit_cv$lambda.1se,
                                 type = "class"
  )
  acc_logit <- mean(logit_pred == test$label)
  
  
  # Logistic TAN interactions
  x <- as.data.frame(TAN_structure$arcs)
  x <- filter(x, from != "label" & to != "label")
  x <- map2_chr(x$from, x$to, \(x, y) paste0(x, ":", y))
  formula_chr <- paste0("label ~ . + ", paste(x, collapse = " + "))
  
  x_TANinter_train <- model.matrix(as.formula(formula_chr), train)[, -1]
  y_TANinter_train <- as.numeric(as.character(train$label))
  
  x_TANinter_test <- model.matrix(as.formula(formula_chr), test[])[, -1]
  y_TANinter_test <- as.numeric(as.character(test$label))
  
  logit_TANinter_model <- glmnet(x_TANinter_train, 
                                 y_TANinter_train, 
                                 family = "multinomial", 
                                 alpha = 1, 
                                 lambda = logit_TANinter_cv$lambda
  )

  logit_TANinter_pred <- predict(logit_TANinter_model, 
                                 newx = x_TANinter_test,
                                 s = logit_TANinter_cv$lambda.1se,
                                 type = "class"
  )
  acc_logitTANinter <- mean(logit_TANinter_pred == test$label)

  
  return(c(acc_NB, acc_TAN, acc_logit, acc_logitTANinter))
}

n_folds <- 5
n_resamples <- 3

set.seed(15052025)
folds <- vfold_cv(train, v = n_folds, repeats = n_resamples, strata = "label")

results <- data.frame(Model = character(), Fold = integer(), Resample = integer(), Accuracy = numeric())

cl <- makeCluster(detectCores() - 1)
time_start <- Sys.time()
for (i in 1:n_resamples) {
  # train_data <- data.frame()
  # colnames(train_data) <- colnames(train)
  index <- c()
  
  for (j in 1:n_folds) {
    if ((j == n_folds) & (i > 1)) {
      next
    }
    new_train_idx <- setdiff(1:length(train$label), folds$splits[[j + (i - 1) * n_folds]][[2]])
    index <- union(index, new_train_idx)
    
    acc_values <- run_models(train[index, ])
    
    temp_results <- data.frame(
      Model = c("NB", "TAN", "Logistic", "Logistic TAN"),
      Fold = j,
      Resample = i,
      Accuracy = acc_values
    )
    
    results <- rbind(results, temp_results)
    print(paste("Fold", j, "done"))
  }
  print(paste("Resample", i, "done"))
}
stopCluster(cl)
 
print(results)
print(Sys.time() - time_start)

# Plot convergence

results_summary <- results %>%
  group_by(Model, Fold) %>%
  summarise(
    Avg_Accuracy = mean(Accuracy),
    SD = sd(Accuracy),
    SE = SD / sqrt(n()),
    .groups = "drop"
  )

ggplot(results_summary, aes(x = Fold, y = Avg_Accuracy, color = Model, group = Model)) +
  geom_line(size = 1) +   
  geom_point(size = 2) +  
  geom_errorbar(aes(ymin = Avg_Accuracy - SE, ymax = Avg_Accuracy + SE, width = 0.2)) + 
  theme_minimal() +   
  labs(
    y = "Average Accuracy ?? SE",
    color = "Model"
  ) +
  theme(legend.position = "top")


# Plot numbers

plot_number <- function(
    v,
    px = 16,
    n_levels = length(levels(train[, 2]))
) {
  
  par(mar = rep(0, 4))
  
  m <- matrix(v, px)
  m <- m[, ncol(m):1]
  
  image(m, col = grDevices::gray.colors(n = n_levels, start = 0, end = 1)[n_levels:1])

}

m <- test[nb_pred != test$label, -1] %>% mutate_all(~as.numeric(as.character(.)))


for (i in 1:351) {
  v <- as.vector(m[i, ])
  v <- as.numeric(v) - 1

  plot_number(v)
}










