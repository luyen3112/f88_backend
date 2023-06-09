import os
import pandas as pd
import numpy as np
import sys

sys.path.append(r'C:\Users\luyen\KLTN\fake\imbDRL-master')

import tensorflow_datasets as tfds
from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val
from imbDRL.utils import rounded_dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU on structured data

episodes = 1_000_000  # Total number of episodes
warmup_steps = 1_070_000  # Amount of warmup steps to collect data with random policy
memory_length = warmup_steps  # Max length of the Replay Memory
batch_size = 32
collect_steps_per_episode = 2000
collect_every = 1500

target_update_period = 1000  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 1  # Soften the target model update
n_step_update = 1

layers = [Dense(526, activation="relu"), Dropout(0.2),
          Dense(526, activation="relu"), Dropout(0.2),
          Dense(526, activation="relu"), Dropout(0.2),
          Dense(526, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]  # No activation, pure Q-values

learning_rate = 0.0001  # Learning rate
gamma = 0.2  # Discount factor
min_epsilon = 0.001  # Minimal and final chance of choosing random action
decay_episodes = episodes // 10  # Number of episodes to decay from 1.0 to `min_epsilon``

min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes

df = pd.read_parquet(r"C:\Users\luyen\KLTN\df_new.parquet", engine='fastparquet')

train = df[df["CREATE_DATE"] <  "2022-02-01"]
val = df[(df["CREATE_DATE"] >=  "2022-02-01") & (df["CREATE_DATE"] <  "2022-08-01")]
test = df[df["CREATE_DATE"] >=  "2022-08-01"]

X_train = train.drop(["CREATE_DATE", "Good_Bad"], axis = 1)
X_val = val.drop(["CREATE_DATE", "Good_Bad"], axis = 1)
X_test = test.drop(["CREATE_DATE", "Good_Bad"], axis = 1)

y_train = train["Good_Bad"]
y_val = val["Good_Bad"]
y_test = test["Good_Bad"]

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_val = pd.get_dummies(X_val)

for i in X_train.columns:
    if i not in X_test.columns:
        X_test[i] = 0
    if i not in X_val.columns:
        X_val[i] = 0
X_test = X_test[X_train.columns]
X_val = X_val[X_train.columns]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = X_train.columns
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)
X_val = pd.DataFrame(scaler.transform(X_val), columns=cols)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_val = X_val.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_val = y_val.to_numpy()

model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

imb_ratio = 0.13

model.compile_model(X_train, y_train, layers, imb_ratio = imb_ratio)
model.q_net.summary()
model.train(X_val, y_val, "F1")

stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.824172, 'F1': 0.781395, 'Precision': 0.730435, 'Recall': 0.84, 'TP': 84, 'TN': 131, 'FP': 31, 'FN': 16}
