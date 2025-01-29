time_start <- Sys.time()

train <- read.csv("mnist_train.csv", header = TRUE)
train <- train %>% 
  mutate(
    across(.cols = label, .fns = as.factor),
    across(.cols = !all_of("label"), .fns = ~./255)
  )

cat("Main data retrival and preprocessing:\t", Sys.time() - time_start, "\n")
time_start <- Sys.time()

train_2_levles <- train %>% 
  mutate(
    across(.cols = !all_of("label"), .fns = ~round(.))
  )

cat("Data with two-level categories:\t", Sys.time() - time_start, "\n")
time_start <- Sys.time()

train_2_levles <- train_2_levles %>% 
  select(where(fn = ~(n_distinct(.) > 1)))

pc(
  suffStat = list(dm = as.matrix(train_2_levles[ , !names(train_2_levles) %in% "label"]), adaptDF = FALSE),
  indepTest = disCItest,
  alpha = 0.05,
  labels = colnames(train_2_levles)[!colnames(train_2_levles) %in% "label"],
  verbose = TRUE
)