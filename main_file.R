library(readr)
library(dplyr)
library(purrr)
library(glmnet)
library(caret)
library(bnlearn)
library(tidyr)
library(ggplot2)
library(RColorBrewer)
library(cowplot)

set.seed(15052025)

time_start <- Sys.time()

train <- read_delim("zip.train", delim = " ", col_names = FALSE)
test <- read_delim("zip.test", delim = " ", col_names = FALSE)

new_colnames <- c("label", 
                  paste0("X", rep(1:16, each = 16), "x", rep(1:16, times = 16)))

n_levels <- 3

train <- train %>% 
  select(!X258) %>% 
  rename_with(~ new_colnames, everything()) %>% 
  mutate(across("label", .fns = ~as.factor(.))) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~cut.default(.,
                                                              breaks = seq(-1, 1, 2/n_levels),
                                                              ordered_result = TRUE,
                                                              include.lowest = TRUE,
                                                              right = TRUE
                                                              ))) %>%
  as.data.frame()

test <- test %>% 
  rename_with(~ new_colnames, everything()) %>% 
  mutate(across("label", .fns = ~as.factor(.))) %>% 
  mutate(across(.cols = !all_of("label"), .fns = ~cut.default(.,
                                                              breaks = seq(-1, 1, 2/n_levels),
                                                              ordered_result = TRUE,
                                                              include.lowest = TRUE,
                                                              right = TRUE))) %>%
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

# Logistic regression with no interaction, lasso penalty
set.seed(150525)
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

cat("Logistic regression, no interactions:")
Sys.time() - time_start

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

cat("Logistic regression, TAN interactions:")
Sys.time() - time_start


# Functions
stratified_sample <- function(data, size) {
  if (size >= nrow(data)) return(data)
  class_counts <- table(data$label)
  class_props <- class_counts / sum(class_counts)
  class_sample_sizes <- round(size * class_props)
  
  class_sample_sizes[class_sample_sizes == 0 & class_counts > 0] <- 1
  
  sampled_data <- do.call(rbind, lapply(names(class_sample_sizes), function(cl) {
    class_data <- data[data$label == cl, ]
    n <- min(nrow(class_data), class_sample_sizes[cl])
    class_data[sample(nrow(class_data), n), ]
  }))
  
  return(sampled_data)
}

run_naive_bayes <- function(train, test) {
  structure <- naive.bayes(train, "label")
  model <- bn.fit(structure, train, method = "bayes")
  pred <- predict(model, test)
  
  cm <- confusionMatrix(as.factor(pred), as.factor(test$label))
  list(acc = cm$overall["Accuracy"], class = cm$byClass[, 1])
}

run_tan <- function(train, test) {
  structure <- tree.bayes(train, "label")
  model <- bn.fit(structure, train, method = "bayes")
  pred <- predict(model, test)
  
  cm <- confusionMatrix(as.factor(pred), as.factor(test$label))
  list(acc = cm$overall["Accuracy"], class = cm$byClass[, 1], structure = structure)
}

run_logistic <- function(train, test, cv_fit) {
  x_train <- model.matrix(label ~ ., train)[, -1]
  y_train <- as.numeric(as.character(train$label))
  
  x_test <- model.matrix(label ~ ., test)[, -1]
  y_test <- as.numeric(as.character(test$label))
  
  model <- glmnet(x_train, y_train, family = "multinomial", alpha = 1, lambda = cv_fit$lambda)
  pred <- predict(model, newx = x_test, s = cv_fit$lambda.1se, type = "class")
  
  cm <- confusionMatrix(as.factor(pred), as.factor(test$label))
  list(acc = cm$overall["Accuracy"], class = cm$byClass[, 1])
}

run_logistic_tan <- function(train, test, tan_structure, cv_fit) {
  arcs <- as.data.frame(tan_structure$arcs)
  interactions <- arcs[arcs$from != "label" & arcs$to != "label", ]
  interaction_terms <- mapply(function(x, y) paste0(x, ":", y), interactions$from, interactions$to)
  
  formula_chr <- paste0("label ~ . + ", paste(interaction_terms, collapse = " + "))
  
  x_train <- model.matrix(as.formula(formula_chr), train)[, -1]
  y_train <- as.numeric(as.character(train$label))
  
  x_test <- model.matrix(as.formula(formula_chr), test)[, -1]
  y_test <- as.numeric(as.character(test$label))
  
  model <- glmnet(x_train, y_train, family = "multinomial", alpha = 1, lambda = cv_fit$lambda)
  pred <- predict(model, newx = x_test, s = cv_fit$lambda.1se, type = "class")
  
  cm <- confusionMatrix(as.factor(pred), as.factor(test$label))
  list(acc = cm$overall["Accuracy"], class = cm$byClass[, 1])
}

run_models <- function(train, test, logit_cv, logit_TANinter_cv, verbose = FALSE) {
  train$label <- as.factor(train$label)
  test$label <- as.factor(test$label)
  
  model_times <- numeric(4)
  
  if (verbose) message("Running Naive Bayes...")
  t_nb <- system.time({
    nb_results <- run_naive_bayes(train, test)
  })
  model_times[1] <- t_nb["elapsed"]
  
  if (verbose) message("Running TAN...")
  t_tan <- system.time({
    tan_results <- run_tan(train, test)
  })
  model_times[2] <- t_tan["elapsed"]
  
  if (verbose) message("Running Logistic Regression...")
  t_logit <- system.time({
    logit_results <- run_logistic(train, test, logit_cv)
  })
  model_times[3] <- t_logit["elapsed"]
  
  if (verbose) message("Running Logistic TAN...")
  t_tan_logit <- system.time({
    logit_tan_results <- run_logistic_tan(train, test, tan_results$structure, logit_TANinter_cv)
  })
  model_times[4] <- t_tan_logit["elapsed"]
  
  acc_all <- c(nb_results$acc, tan_results$acc, logit_results$acc, logit_tan_results$acc)
  class_all <- c(nb_results$class, tan_results$class, logit_results$class, logit_tan_results$class)
  
  return(list(
    acc = acc_all,
    class = class_all,
    time = model_times
  ))
}


set.seed(15052025)

sample_sizes <- round(10^(seq(1.5, 4, 0.1)))
sample_sizes <- sample_sizes[sample_sizes <= nrow(train)]
n_repeats <- 100

results_list <- vector("list", length(sample_sizes) * n_repeats)
class_results_list <- vector("list", length(sample_sizes) * n_repeats)

counter <- 1
time_start <- Sys.time()

for (size in sample_sizes) {
  for (r in 1:n_repeats) {
    sampled_data <- stratified_sample(train, size)    
    from_func <- run_models(
      train = sampled_data,
      test = test,
      logit_cv = logit_cv,
      logit_TANinter_cv = logit_TANinter_cv,
      verbose = TRUE
      )
    acc_values <- from_func[["acc"]]
    class_values <- from_func[["class"]]
    time_values <- from_func[["time"]]
    
    results_list[[counter]] <- data.frame(
      Model = c("NB", "TAN", "Logistic", "Logistic TAN"),
      Size = size,
      Repeat = r,
      Accuracy = acc_values,
      Time = time_values
    )
    
    class_results_list[[counter]] <- data.frame(
      Model = rep(c("NB", "TAN", "Logistic", "Logistic TAN"), each = 10),
      Class = rep(0:9, 4),
      Size = size,
      Repeat = r,
      Accuracy = class_values
    )
    
    message(sprintf("Sample size %d, repeat %d done", size, r))
    counter <- counter + 1
  }
}

results <- do.call(rbind, results_list)
class_results <- do.call(rbind, class_results_list)

print(results)
print(Sys.time() - time_start)

write_csv(results, "results.csv")
write_csv(class_results, "class_results.csv")

# Plot convergence accuracy
# pdf("learning_curve.pdf")

results$LogSize <- log10(results$Size)

ggplot(results, aes(x = LogSize, y = Accuracy, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1) +
  stat_summary(fun = mean, geom = "point", size = 1.25) +
  scale_x_continuous(
    breaks = seq(1, 4, 0.5),
    labels = scales::math_format(10^.x)
  ) +
  scale_y_continuous(
    breaks = seq(0.3, 1, 0.05)
  ) +
  scale_color_manual(
    values = c(
      "Logistic" = "#D73027",
      "Logistic TAN" = "#67001F",
      "NB" = "#1F78B4",
      "TAN" = "#08457E"
    )
  ) +
  labs(
    x = "Sample size",
    y = "Average Accuracy",
    color = "Model"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

# dev.off()

# Plot time
pdf("runtime_curve.pdf")

results$LogSize <- log10(results$Size)

ggplot(results, aes(x = LogSize, y = Time, color = Model)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1) +
  stat_summary(fun = mean, geom = "point", size = 1.25) +
  scale_x_continuous(
    breaks = seq(1, 4, 0.5),
    labels = scales::math_format(10^.x)
  ) +
  scale_y_continuous(
    trans = "log10",
    # breaks = seq(0, 26, 2),
    # labels = scales::math_format(10^.x)
  ) +
  scale_color_manual(
    values = c(
      "Logistic" = "#D73027",
      "Logistic TAN" = "#67001F",
      "NB" = "#1F78B4",
      "TAN" = "#08457E"
    )
  ) +
  labs(
    x = "Sample size",
    y = "Average Time (s)",
    color = "Model"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

dev.off()

# Plot class convergence for digits

class_results$LogSize <- log10(class_results$Size)

cclass_results <- class_results %>%
  mutate(
    Class = as.factor(Class),
    LineType = ifelse(as.numeric(as.character(Class)) %% 2 == 0, "even", "odd")
  )

# pdf("class_accuracy_curve.pdf")

# Set class color palette
class_palette <- brewer.pal(10, "Paired")

ggplot(cclass_results, aes(x = LogSize, y = Accuracy, color = Class, linetype = LineType)) +
  stat_summary(fun = mean, geom = "line", linewidth = 0.7) +
  stat_summary(fun = mean, geom = "point", size = 0.7) +
  scale_color_manual(values = class_palette) +
  scale_linetype_manual(values = c("even" = "solid", "odd" = "twodash")) +
  scale_x_continuous(
    breaks = seq(1, 4, 0.5),
    labels = scales::math_format(10^.x)
  ) +
  scale_y_continuous(limits = c(0.2, 1), breaks = seq(0.2, 1, 0.1)) +
  facet_wrap(~ Model, ncol = 2) +
  labs(
    title = "Per-Class Accuracy by Sample Size",
    x = "Sample size",
    y = "Average Class Accuracy",
    color = "Class",
    linetype = "Parity"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",
    strip.text = element_text(face = "bold")
  )

# dev.off()

# Digit plotting

plot_digit_gg <- function(vec, pix = 16) {
  mat <- matrix(vec, nrow = pix, byrow = TRUE)
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

set.seed(150525)
pdf("Data_example_with_discritization.pdf")
# x <- rbn(TAN_model, 2007)
# x <- rbn(nb_model, 2007)
x <- train
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
dev.off()


# Function to plot a digit image
plot_digit_gg <- function(vec, pix = 16) {
  mat <- matrix(vec, nrow = pix, byrow = TRUE)
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

generate_digit_plots <- function(data) {
  plot_list <- list()
  
  for (digit in 0:9) {
    digit_data <- data %>% filter(label == digit)
    
    avg_image <- digit_data %>%
      mutate_all(as.numeric) %>%
      summarise_all(mean) %>%
      select(-label) %>%
      as.numeric()
    
    mean_plot <- plot_digit_gg(avg_image)
    
    example_plots <- lapply(1:10, function(i) {
      plot_digit_gg(as.numeric(digit_data[i, -1]))
    })
    
    plot_list <- c(plot_list, list(mean_plot), example_plots)
  }
  return(plot_list)
}

save_digit_grid <- function(data, filename = "digit_grid.pdf", seed = 150525) {
  set.seed(seed)
  plot_list <- generate_digit_plots(data)
  pdf(filename)
  plot_grid(plotlist = plot_list, nrow = 10)
  # dev.off()
}

save_digit_grid(test, "test_digits.pdf")
dev.off()
save_digit_grid(rbn(nb_model, 2007), "nb_generated_digits.pdf")
dev.off()
save_digit_grid(rbn(TAN_model, 2007), "tan_generated_digits.pdf")
dev.off()
