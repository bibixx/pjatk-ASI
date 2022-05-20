# %%
import pandas as pd
import numpy as np

# %%
import logging

# %%
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# %%
csv_url = ("dataset_short.csv")
try:
  dataset = pd.read_csv(csv_url, sep=',')
except Exception as e:
  logger.exception(f'Oj, coś poszło nie tak. Error: {e}')

# %%
dataset.shape

# %%
dataset.info()

# %%
dataset.drop('isFlaggedFraud', axis=1, inplace=True)

# %%
dataset.info()

# %%
data = dataset.sample(frac=0.75, random_state=2137)

# %%
data.head()

# %%
data.reset_index(drop=True, inplace=True)
data.head()

# %%
data_unseen = dataset.drop(data.index)

# %%
data_unseen.head()

# %%
data_unseen.reset_index(drop=True, inplace=True)
data_unseen.head()

# %%
print(f'Dane treningowe: {data.shape}')
print(f'Dane testowe: {data_unseen.shape}')

# %%
print(set(data['isFraud']))

print(set(data_unseen['isFraud']))

# %%
data.head()

# %%
from pycaret.classification import *

# %%
exp1 = setup(
  data=data,
  target='isFraud',
  session_id=2137,
  log_experiment=True,
  experiment_name="fraud",
  silent=True
)

# %%
best = compare_models()

# %%
import datetime
formatted_now = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M")
model_name = f"fraud_{formatted_now}"

# %%
# final = finalize_model(best)
# save_model(final, model_name=f"{model_name}_compare")

# %%
xgboost = create_model('xgboost')
tuned_xgboost = tune_model(xgboost)
# evaluate_model(tuned_lightgbm)
# predict_model(tuned_lightgbm)
# save_model(tuned_lightgbm, model_name=f"{model_name}_lightgbm")

# %%
plot_model(tuned_xgboost, plot='feature')

# %%
evaluate_model(tuned_xgboost)


