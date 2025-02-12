library(dplyr)

train <- read.csv("mnist_train.csv", header = TRUE)
train <- train %>% 
  mutate(
    across(.cols = label, .fns = as.factor),
    across(.cols = !all_of("label"), .fns = ~./255)
  )

train <- train %>%
  mutate(across(
    .cols = where(is.numeric) & !all_of("label"),
    .fns = ~ if_else(. %in% c(0, 1), NaN, .)
  ))

train %>% 
  select(ends_with("x28")) %>% 
  hist.data.frame(n.unique = 1)

train %>% 
  group_by() %>%
  summarise(n=n_distinct())

write.csv(train, "train_data_part_1.csv", quote = FALSE, sep = ",", row.names = FALSE)


# Cont. New data
library(bnlearn)
library(glmnet)
library(caret)

train <- read.csv("zip.train", header = FALSE, sep = "")
test <- read.csv("zip.test", header = FALSE, sep = "")

new_colnames <- c("label", paste0("X", rep(1:16, each = 16), "x", rep(1:16, times = 16)))
train <- train %>% 
  rename_with(~ new_colnames, everything()) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~round((.+1)/2))) %>% 
  mutate_all(.funs = ~as.factor(.))

test <- test %>% 
  rename_with(~ new_colnames, everything()) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~round((.+1)/2))) %>% 
  mutate_all(.funs = ~as.factor(.)) %>% 
  mutate(X1x1 = )

train <- train %>% 
  select(!"X1x1")
test <- test %>% 
  select(!"X1x1")


# Naive Bayes
nb_structure <- naive.bayes(train, "label")
nb_model <- bn.fit(nb_structure, train)

nb_pred <- predict(nb_model, test)
print(confusionMatrix(nb_pred, test$label))


# Logistic regression
x_train <- model.matrix(label ~ ., train)[, -1]
y_train <- as.numeric(as.character(train$label))

x_test <- model.matrix(label ~ ., test)[, -1]
y_test <- as.numeric(as.character(test$label))

logit_model <- glmnet(x_train, y_train, family = "multinomial", alpha = 1)

logit_pred <- predict(logit_model, newx = x_test, type = "class")

conf_matrix_logit <- confusionMatrix(factor(logit_pred), factor(y_test))
print(conf_matrix_logit)