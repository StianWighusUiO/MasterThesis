library(readr)
library(dplyr)
library(purrr)
library(ggplot2)
library(glmnet)
library(caret)
library(bnlearn)
library(tidyr)
library(RColorBrewer)
library(cowplot)
library(scales)

time_start <- Sys.time()

train <- read_delim("zip.train", delim = " ", col_names = FALSE)
test <- read_delim("zip.test", delim = " ", col_names = FALSE)

new_colnames <- c("label", paste0("X", rep(1:16, each = 16), "x", rep(1:16, times = 16)))
n_levels <- 3

train <- train %>%
  select(!X258) %>%
  rename_with(~ new_colnames, everything()) %>%
  mutate(across("label", as.factor)) %>%
  mutate(across(-label, ~ cut(.x, breaks = seq(-1, 1, 2 / n_levels), ordered_result = TRUE, include.lowest = TRUE))) %>%
  as.data.frame()

test <- test %>%
  rename_with(~ new_colnames, everything()) %>%
  mutate(across("label", as.factor)) %>%
  mutate(across(-label, ~ cut(.x, breaks = seq(-1, 1, 2 / n_levels), ordered_result = TRUE, include.lowest = TRUE))) %>%
  as.data.frame()

cat("Data loaded and preprocessed:\n")
print(Sys.time() - time_start)

excluded_digit <- "5"
train_ood <- train %>% filter(label != excluded_digit)
train_ood$label <- droplevels(train_ood$label)

compute_entropy <- function(probs) {
  -sum(probs * log(probs + 1e-10))
}

results_list <- list()

logsumexp <- function(log_probs) {
  max_log <- max(log_probs)
  max_log + log(sum(exp(log_probs - max_log)))
}

# Naive Bayes
nb_structure <- naive.bayes(train_ood, "label")
nb_model <- bn.fit(nb_structure, train_ood, method = "bayes")
nb_preds <- predict(nb_model, test, prob = TRUE)


y <- factor((0:9)[-6], (0:9)[-6])

nb_logLik <- lapply(1:2007, function(i) {
  lapply(y, function(j) {
    x <- data.frame(label = j, test[i, -1])
    names(x) <- names(test)
    logLik(nb_model, x)
  })
})
nb_probs <- sapply(nb_logLik, function(i) {
  logsumexp(unlist(i))
})

nb_posteriors <- attr(nb_preds, "prob")
nb_entropy <- apply(nb_posteriors, 2, compute_entropy)


results_list[["nb"]] <- data.frame(
  model = "nb",
  true_label = test$label,
  predicted_label = nb_preds,
  log_prob = nb_probs,
  entropy = nb_entropy,
  is_ood = test$label == excluded_digit
)

cat("Naive Bayes done\n")

# TAN
tan_structure <- tree.bayes(train_ood, "label")
tan_model <- bn.fit(tan_structure, train_ood, method = "bayes")
tan_preds <- predict(tan_model, test, prob = TRUE)

y <- factor((0:9)[-6], (0:9)[-6])

tan_logLik <- lapply(1:2007, function(i) {
  lapply(y, function(j) {
    x <- data.frame(label = j, test[i, -1])
    names(x) <- names(test)
    logLik(tan_model, x)
  })
})
tan_probs <- sapply(tan_logLik, function(i) {
  logsumexp(unlist(i))
})

tan_posteriors <- attr(tan_preds, "prob")
tan_entropy <- apply(tan_posteriors, 2, compute_entropy)



results_list[["tan"]] <- data.frame(
  model = "tan",
  true_label = test$label,
  predicted_label = tan_preds,
  log_prob = tan_probs,
  entropy = tan_entropy,
  is_ood = test$label == excluded_digit
)

cat("TAN done\n")

x_train <- model.matrix(label ~ ., train_ood)[, -1]
y_train <- as.numeric(as.character(train_ood$label))

x_test <- model.matrix(label ~ ., test)[, -1]
y_test <- as.numeric(as.character(test$label))

# Logistic Regression
logit_cv <- cv.glmnet(x_train, y_train, family = "multinomial", alpha = 1)

logit_probs <- predict(logit_cv$glmnet.fit, newx = x_test, s = logit_cv$lambda.1se, type = "response")[,,1]
logit_preds <- colnames(logit_probs)[apply(logit_probs, 1, which.max)]



results_list[["logreg"]] <- data.frame(
  model = "logreg",
  true_label = test$label,
  predicted_label = logit_preds,
  log_prob = log(apply(logit_probs, 1, max)),
  entropy = apply(logit_probs, 1, compute_entropy),
  is_ood = test$label == excluded_digit
)

cat("Logistic regression done\n")

# Logistic Regression with TAN interactions
interactions <- as.data.frame(tan_structure$arcs) %>%
  filter(from != "label" & to != "label") %>%
  transmute(term = paste0(from, ":", to)) %>%
  pull(term)

formula_tan <- paste0("label ~ . + ", paste(interactions, collapse = " + "))

x_TAN_train <- model.matrix(as.formula(formula_tan), train_ood)[, -1]
y_TAN_train <- as.numeric(as.character(train_ood$label))

x_TAN_test <- model.matrix(as.formula(formula_tan), test)[, -1]
y_TAN_test <- as.numeric(as.character(test$label))

logit_tan_cv <- cv.glmnet(x_TAN_train, y_TAN_train, family = "multinomial", alpha = 1)
logit_tan_probs <- predict(logit_tan_cv$glmnet.fit, newx = x_TAN_test, s = logit_tan_cv$lambda.1se, type = "response")[,,1]
logit_tan_preds <- colnames(logit_tan_probs)[apply(logit_tan_probs, 1, which.max)]

results_list[["logreg_tan"]] <- data.frame(
  model = "logreg_tan",
  true_label = test$label,
  predicted_label = logit_tan_preds,
  log_prob = log(apply(logit_tan_probs, 1, max)),
  entropy = apply(logit_tan_probs, 1, compute_entropy),
  is_ood = test$label == excluded_digit
)

cat("Logistic regression + TAN done\n")

# === Combine and save results ===
df <- bind_rows(results_list)

write_csv(df, "ood_results.csv")
cat("Saved: ood_results.csv\n")

# === Plotting ===
df <- df %>%
  mutate(
    is_ood = factor(is_ood, levels = c(FALSE, TRUE), labels = c("Seen", "OOD")),
    model = factor(model, levels = c("nb", "tan", "logreg", "logreg_tan"),
                   labels = c("Naive Bayes", "TAN", "LogReg", "LogReg+TAN"))
  )

# Separate generative and discriminative models
generative_df <- df %>% filter(model %in% c("Naive Bayes", "TAN"))
discriminative_df <- df %>% filter(model %in% c("LogReg", "LogReg+TAN")) %>% mutate(prob = exp(log_prob), .keep = "unused")

# === Plot: Generative models (using log P(x)) ===
ggplot(generative_df, aes(x = log_prob, after_stat(density), fill = is_ood)) +
  geom_histogram(, alpha = 0.7, position = "identity", bins = 40) +
  facet_wrap(~ model, ncol = 2) +
  scale_fill_manual(values = c("Seen" = "#0072B2", "OOD" = "#D55E00")) +
  labs(
    title = "Generative Models: log P(x) ??? Seen vs. OOD",
    x = "log P(x)",
    y = "Density",
    fill = "Sample"
  ) +
  theme_minimal(base_size = 13)
ggsave("generative_logpx_histogram.pdf", width = 8, height = 5)


# === Plot: Discriminative models (using max P(y|x)) ===
ggplot(discriminative_df, aes(x = prob, after_stat(density), fill = is_ood)) +
  geom_histogram(alpha = 0.7, position = "identity", bins = 30) +
  facet_wrap(~ model, ncol = 2) +
  scale_fill_manual(values = c("Seen" = "#0072B2", "OOD" = "#D55E00")) +
  labs(
    title = "Discriminative Models: max P(y|x) ??? Seen vs. OOD",
    x = "max P(y | x)",
    y = "Density",
    fill = "Sample"
  ) +
  theme_minimal(base_size = 13)
ggsave("discriminative_p_histogram.pdf", width = 8, height = 5)


# --- Entropy Boxplot ---
ggplot(generative_df, aes(x = model, y = entropy, fill = is_ood)) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.5) +
  scale_fill_manual(values = c("Seen" = "#0072B2", "OOD" = "#D55E00")) +
  labs(
    title = "Entropy (Generative Models)",
    x = "Model",
    y = "Entropy",
    fill = "Sample"
  ) +
  theme_minimal(base_size = 13)
ggsave("generative_entropy_boxplot.pdf", width = 8, height = 5)

ggplot(discriminative_df, aes(x = model, y = entropy, fill = is_ood)) +
  geom_boxplot(alpha = 0.8, outlier.size = 0.5) +
  scale_fill_manual(values = c("Seen" = "#0072B2", "OOD" = "#D55E00")) +
  labs(
    title = "Entropy (Discriminative Models)",
    x = "Model",
    y = "Entropy",
    fill = "Sample"
  ) +
  theme_minimal(base_size = 13)
ggsave("discriminative_entropy_boxplot.pdf", width = 8, height = 5)
