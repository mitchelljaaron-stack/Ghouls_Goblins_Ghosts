# GhoulsGoblinsGhosts_Analysis_1st_Trial

library(glmnet)
library(tidyverse)
library(tidymodels) 
library(vroom) 
library(patchwork)
library(ggplot2) 
library(recipes) 
library(embed) 
library(dials)
library(tune)

train_data <- vroom("train.csv") 
test_data <- vroom("test.csv")

# Feature Engineering 
train_data <- train_data %>%
  mutate(
    type = as.factor(type),
    color = as.factor(color))

test_data <- test_data %>%
  mutate(color = as.factor(color))


# Recipe
my_recipe <- recipe(type ~ ., data = train_data) %>%
  update_role(id, new_role = "id variable") %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_normalize(all_predictors())

DataExplorer::plot_correlation(train_data)

# mod and wf

rf_mod <- rand_forest( mtry = tune(), min_n = tune(), trees = 500 ) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod) 

tuning_grid <- grid_regular(
  mtry(range = c(1, 10)),
  min_n(range = c(2, 10)),
  levels = 4
)


## Split data for CV
folds <- vfold_cv(train_data, v = 4) 


metrics_multiclass <- metric_set(accuracy, mn_log_loss, roc_auc)


## Run the CV 

CV_results <- rf_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metrics_multiclass,
    control = control_grid(save_pred = TRUE)
  )

## Find Best Tuning Parameters 
bestTune <- CV_results %>% select_best(metric = "roc_auc")

## Finalize the Workflow & fit it 
final_wf <- rf_wf %>%
  finalize_workflow(bestTune)%>%
  fit(data=train_data) 

## Predict 
final_predictions <- final_wf %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data %>% select(id)) %>%
  rename(type = .pred_class) %>%
  select(id, type)


# Export processed dataset 

vroom_write(x = final_predictions, file = "./ggg_rf_model_preds_a.csv", delim = ",")

