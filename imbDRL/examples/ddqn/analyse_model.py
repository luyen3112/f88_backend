import os
import pandas as pd
import sys

sys.path.append(r'C:\Users\luyen\KLTN\fake\imbDRL-master')
from imbDRL.agents.ddqn import TrainDDQN

from imbDRL.metrics import (classification_metrics, network_predictions,
                            plot_confusion_matrix, plot_pr_curve,
                            plot_roc_curve)
from imbDRL.utils import rounded_dict
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU on structured data

min_class = [1]  # Minority classes, same setup as in original paper
maj_class = [0]  # Majority classes
fp_model = r"C:\Users\luyen\KLTN\fake\imbDRL-master\models\20230603_003037.pkl"
# fp_model = r"C:\Users\luyen\KLTN\fake\imbDRL-master\models\20230604_172702.pkl"
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
print(X_train.columns)

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

import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.externals.joblib import dump, load
joblib.dump(scaler, r'C:\Users\luyen\KLTN\fake\imbDRL-master\w\std_scaler.save')

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_val = X_val.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_val = y_val.to_numpy()

network = TrainDDQN.load_network(fp_model)

y_pred_train = network_predictions(network, X_train)
y_pred_test = network_predictions(network, X_test)
y_pred_val = network_predictions(network, X_val)

stats = classification_metrics(y_train, y_pred_train)
# print(f"Train: {rounded_dict(stats)}")
stats.to_excel(r"C:\Users\luyen\KLTN\train_q.xlsx")
stats = classification_metrics(y_val, y_pred_val)
# print(f"Val: {rounded_dict(stats)}")
stats.to_excel(r"C:\Users\luyen\KLTN\val_q.xlsx")
stats = classification_metrics(y_test, y_pred_test)
# print(f"Test:  {rounded_dict(stats)}")
stats.to_excel(r"C:\Users\luyen\KLTN\test_q.xlsx")

# plot_pr_curve(network, X_test, y_test, X_train, y_train)
# plot_roc_curve(network, X_test, y_test, X_train, y_train)
# plot_confusion_matrix(stats.get("TP"), stats.get("FN"), stats.get("FP"), stats.get("TN"))
