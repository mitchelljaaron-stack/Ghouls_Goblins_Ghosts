library(discrim)
library(glmnet)
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(ggplot2)
library(recipes)
library(embed)
library(themis)

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

# Feature Engineering

train_data <- train_data %>%
  mutate(
    type = as.factor(type),
    color = as.factor(color))

test_data <- test_data %>%
  mutate(color = as.factor(color))

ggplot(data = train_data, aes(bone_length, hair_length))+
  geom_point()

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


# Create recipe
my_recipe <- recipe(type ~ ., data = train_data) %>%
  # Collapse rare categories (<0.1%)
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  # Target encoding
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_smote(all_outcomes(), neighbors=2)



nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Grid of values to tune over
tuning_grid <- grid_regular(
  Laplace(range = c(0, 1)),
  smoothness(range = c(0,1)),
  levels = 5
)

## Split data for CV
folds <- vfold_cv(train_data, v = 4, repeats=2)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

## Predict
final_predictions <- final_wf %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data %>% select(id)) %>%
  rename(type = .pred_class) %>%
  select(id, type)

# Export processed dataset
vroom_write(x = final_predictions, file = "./ggg_nb_model_preds_c.csv", delim = ",")
