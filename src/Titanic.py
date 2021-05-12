import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from optbinning import OptimalBinning
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
from sklearn import (
    ensemble,
    linear_model,
    impute,
    svm,
    discriminant_analysis,
    naive_bayes,
    neighbors,
    gaussian_process,
    tree,
    model_selection,
)
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import *
from sklearn.metrics import accuracy_score, make_scorer
from tensorflow.keras.metrics import *
import warnings

warnings.filterwarnings("ignore")

# Import data
df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")
train_shape = df_train.shape
test_shape = df_test.shape

# Let's merge train and test for future feature engineering
full_df = pd.concat([df_train, df_test]).reset_index(drop=True)

# This is a validation sample, so we can avoid overfitting
df_train_test = df_train.sample(frac=0.2, random_state=123)
y_train_test = df_train_test[["Survived", "PassengerId"]]
df_train_test = df_train_test.drop(["Survived"], axis=1)
list_index = df_train_test.index.values.tolist()
df_train_train = df_train[~df_train.index.isin(list_index)]
full_df_model = pd.concat([df_train_test, df_train_train])

# Functions
def create_family_rate(df, train):
    MEAN_SURVIVAL_RATE = round(np.mean(train["Survived"]), 4)

    df["Family_Friends_Surv_Rate"] = MEAN_SURVIVAL_RATE
    df["Surv_Rate_Invalid"] = 1

    for _, grp_df in df[
        ["Survived", "Family_Name", "Fare", "Ticket", "PassengerId"]
    ].groupby(["Family_Name", "Fare"]):
        if len(grp_df) > 1:
            if grp_df["Survived"].isnull().sum() != len(grp_df):
                for ind, row in grp_df.iterrows():
                    df.loc[
                        df["PassengerId"] == row["PassengerId"],
                        "Family_Friends_Surv_Rate",
                    ] = round(grp_df["Survived"].mean(), 4)
                    df.loc[
                        df["PassengerId"] == row["PassengerId"], "Surv_Rate_Invalid"
                    ] = 0

    for _, grp_df in df[
        [
            "Survived",
            "Family_Name",
            "Fare",
            "Ticket",
            "PassengerId",
            "Family_Friends_Surv_Rate",
        ]
    ].groupby("Ticket"):
        if len(grp_df) > 1:
            for ind, row in grp_df.iterrows():
                if (row["Family_Friends_Surv_Rate"] == 0.0) | (
                    row["Family_Friends_Surv_Rate"] == MEAN_SURVIVAL_RATE
                ):
                    if grp_df["Survived"].isnull().sum() != len(grp_df):
                        df.loc[
                            full_df["PassengerId"] == row["PassengerId"],
                            "Family_Friends_Surv_Rate",
                        ] = round(grp_df["Survived"].mean(), 4)
                        df.loc[
                            full_df["PassengerId"] == row["PassengerId"],
                            "Surv_Rate_Invalid",
                        ] = 0

    return df


def clean_ticket(each_ticket):
    prefix = re.sub(r"[^a-zA-Z]", "", each_ticket)
    if prefix:
        return prefix
    else:
        return "NUM"


def fare_cat(fare):
    if fare <= 7.0:
        return 1
    elif fare <= 39 and fare > 7.0:
        return 2
    else:
        return 3


def knn_imputer(df, features, k=10):
    imputer = impute.KNNImputer(n_neighbors=k, missing_values=np.nan)
    imputer.fit(df[features])
    df.loc[:, features] = pd.DataFrame(
        imputer.transform(df[features]), index=df.index, columns=features
    )

    return df


def create_optmal_binning(df):
    optb = OptimalBinning(name="Age", dtype="numerical", solver="cp")
    x = df[: train_shape[0]]["Age"].values
    y_train = df[: train_shape[0]]["Survived"]
    y = y_train[y_train.index.isin(df_train.index)]
    optb.fit(x, y)

    return optb


def transform_variable(df):
    list_index = df.index.values.tolist()
    col = df["Age"].values
    x_transform = optb.transform(col, metric="event_rate")
    x_transform = pd.Series(x_transform, index=list_index)
    x_transform.value_counts()
    x_transform = x_transform.rename("Age_Band")
    df = pd.concat((df, x_transform), axis=1)

    return df


def get_accuracy(prediction):
    score = round(accuracy_score(prediction, y_test) * 100, 2)
    print("Accuracy", score)

    return score


def create_model():
    metrics = ["accuracy", Precision(), Recall()]
    model = Sequential()
    model.add(Input(shape=X_train.shape[1], name="Input_"))
    model.add(
        layers.Dense(
            8,
            activation="relu",
            kernel_initializer="glorot_normal",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(
        layers.Dense(
            16,
            activation="relu",
            kernel_initializer="glorot_normal",
            kernel_regularizer=regularizers.l2(0.1),
        )
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(
            16,
            activation="relu",
            kernel_initializer="glorot_normal",
            kernel_regularizer=regularizers.l2(0.1),
        )
    )
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_normal"))

    optimize = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimize, loss="binary_crossentropy", metrics=metrics)

    return model


# Encoding variable Sex
full_df.loc[:, "Sex"] = (full_df.loc[:, "Sex"] == "female").astype(int)

# Creating variable Title
full_df["Title"] = full_df["Name"]
full_df["Title"] = full_df["Name"].str.extract("([A-Za-z]+)\.", expand=True)

# Replacing rare titles
mapping = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Major": "Other",
    "Col": "Other",
    "Dr": "Other",
    "Rev": "Other",
    "Capt": "Other",
    "Jonkheer": "Royal",
    "Sir": "Royal",
    "Lady": "Royal",
    "Don": "Royal",
    "Countess": "Royal",
    "Dona": "Royal",
}
full_df.replace({"Title": mapping}, inplace=True)
full_df["Title_C"] = full_df["Title"]
full_df["Embarked"] = full_df["Embarked"].fillna("S")
full_df = pd.get_dummies(
    full_df, columns=["Embarked", "Title_C"], prefix=["Emb", "Title"], drop_first=False
)
title_dict = {"Mr": 1, "Miss": 2, "Mrs": 3, "Other": 4, "Royal": 5, "Master": 6}
full_df["Title"] = full_df["Title"].map(title_dict).astype("int")
full_df["Family_Size"] = full_df["Parch"] + full_df["SibSp"] + 1
full_df["Fsize_Cat"] = full_df["Family_Size"].map(
    lambda val: "Alone" if val <= 1 else ("Small" if val < 5 else "Big")
)
full_df["isAlone"] = full_df.Family_Size.apply(lambda x: 1 if x == 1 else 0)
Fsize_dict = {"Alone": 3, "Small": 2, "Big": 1}
full_df["Fsize_Cat"] = full_df["Fsize_Cat"].map(Fsize_dict).astype("int")

# Using the variable Name to create Family Name and Name Length
full_df["Name_Length"] = full_df.Name.str.replace("[^a-zA-Z]", "").str.len()
full_df["Family_Name"] = full_df["Name"].str.extract(
    "([A-Za-z]+.[A-Za-z]+)\,", expand=True
)
full_df_model["Family_Name"] = full_df_model["Name"].str.extract(
    "([A-Za-z]+.[A-Za-z]+)\,", expand=True
)
full_df = create_family_rate(full_df, df_train)
full_df_model = create_family_rate(full_df_model, df_train_train)
full_df = full_df.drop(["Name", "Family_Name"], axis=1)
full_df_model = full_df_model.drop(["Name", "Family_Name"], axis=1)

# Cleaning variable Cabin
full_df["Cabin_Clean"] = full_df["Cabin"].fillna("U")
full_df["Cabin_Clean"] = full_df["Cabin_Clean"].str.strip(" ").str[0]
cabin_dict = {"A": 9, "B": 8, "C": 7, "D": 6, "E": 5, "F": 4, "G": 3, "T": 2, "U": 1}
full_df["Cabin_Clean"] = full_df["Cabin_Clean"].map(cabin_dict).astype("int")
full_df.drop(["Cabin"], axis=1, inplace=True)

# Ticket
full_df["Tkt_Clean"] = full_df.Ticket.apply(clean_ticket)
full_df["Ticket_Frequency"] = full_df.groupby("Ticket")["Ticket"].transform("count")
full_df.drop(["Ticket"], axis=1, inplace=True)
full_df = pd.get_dummies(
    full_df, columns=["Tkt_Clean"], prefix=["Tkt"], drop_first=True
)

# Fare
full_df.loc[:, "Fare_Cat"] = full_df["Fare"].apply(fare_cat).astype("int")
full_df.loc[:, "Fare_Family_Size"] = full_df["Fare"] / full_df["Family_Size"]
full_df.loc[:, "Fare_Cat_Pclass"] = full_df["Fare_Cat"] * full_df["Pclass"]
full_df.loc[:, "Fare_Cat_Title"] = full_df["Fare_Cat"] * full_df["Title"]
full_df.loc[:, "Fsize_Cat_Title"] = full_df["Fsize_Cat"] * full_df["Title"]
full_df.loc[:, "Fsize_Cat_Fare_Cat"] = full_df["Fare_Cat"] / full_df[
    "Fsize_Cat"
].astype("int")
full_df.loc[:, "Pclass_Title"] = full_df["Pclass"] * full_df["Title"]
full_df.loc[:, "Fsize_Cat_Pclass"] = full_df["Fsize_Cat"] * full_df["Pclass"]

# Cleaning Data
colsToRemove = []
cols = [
    "Tkt_AQ",
    "Tkt_AS",
    "Tkt_C",
    "Tkt_CA",
    "Tkt_CASOTON",
    "Tkt_FC",
    "Tkt_FCC",
    "Tkt_Fa",
    "Tkt_LINE",
    "Tkt_LP",
    "Tkt_NUM",
    "Tkt_PC",
    "Tkt_PP",
    "Tkt_PPP",
    "Tkt_SC",
    "Tkt_SCA",
    "Tkt_SCAH",
    "Tkt_SCAHBasle",
    "Tkt_SCOW",
    "Tkt_SCPARIS",
    "Tkt_SCParis",
    "Tkt_SOC",
    "Tkt_SOP",
    "Tkt_SOPP",
    "Tkt_SOTONO",
    "Tkt_SOTONOQ",
    "Tkt_SP",
    "Tkt_STONO",
    "Tkt_STONOQ",
    "Tkt_SWPP",
    "Tkt_WC",
    "Tkt_WEP",
    "Fare_Cat",
    "Fare_Family_Size",
    "Fare_Cat_Pclass",
    "Fare_Cat_Title",
    "Fsize_Cat_Title",
    "Fsize_Cat_Fare_Cat",
    "Pclass_Title",
    "Fsize_Cat_Pclass",
]
for col in cols:
    if full_df[col][: train_shape[0]].std() == 0:
        colsToRemove.append(col)
full_df.drop(colsToRemove, axis=1, inplace=True)
features = ["Survived", "Family_Friends_Surv_Rate", "Surv_Rate_Invalid"]
df = full_df.copy()
df.loc[df.PassengerId.isin(full_df_model.PassengerId), features] = full_df_model[
    features
]
passenger_list = full_df_model["PassengerId"].tolist()
full_df_model = df[df["PassengerId"].isin(passenger_list)]
imp_features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Title",
    "Name_Length",
    "Emb_C",
    "Emb_Q",
    "Emb_S",
    "Family_Size",
    "Fsize_Cat",
    "Family_Friends_Surv_Rate",
    "Surv_Rate_Invalid",
    "Cabin_Clean",
    "Ticket_Frequency",
    "Tkt_AS",
    "Tkt_C",
    "Tkt_CA",
    "Tkt_CASOTON",
    "Tkt_FC",
    "Tkt_FCC",
    "Tkt_Fa",
    "Tkt_LINE",
    "Tkt_NUM",
    "Tkt_PC",
    "Tkt_PP",
    "Tkt_PPP",
    "Tkt_SC",
    "Tkt_SCA",
    "Tkt_SCAH",
    "Tkt_SCAHBasle",
    "Tkt_SCOW",
    "Tkt_SCPARIS",
    "Tkt_SCParis",
    "Tkt_SOC",
    "Tkt_SOP",
    "Tkt_SOPP",
    "Tkt_SOTONO",
    "Tkt_SOTONOQ",
    "Tkt_SP",
    "Tkt_STONO",
    "Tkt_SWPP",
    "Tkt_WC",
    "Tkt_WEP",
    "Fare_Cat",
    "Fare_Family_Size",
    "Fare_Cat_Pclass",
    "Fare_Cat_Title",
    "Fsize_Cat_Title",
    "Fsize_Cat_Fare_Cat",
    "Pclass_Title",
    "Fsize_Cat_Pclass",
]

full_df = knn_imputer(full_df, imp_features)
full_df_model = knn_imputer(full_df_model, imp_features)

# Age
optb = create_optmal_binning(full_df)
full_df = transform_variable(full_df)
optb = create_optmal_binning(full_df_model)
full_df_model = transform_variable(full_df_model)

# Child/Senior
full_df["Child"] = full_df["Age"].map(lambda val: 1 if val < 18 else 0)
full_df["Senior"] = full_df["Age"].map(lambda val: 1 if val > 70 else 0)

# Standardization
from sklearn.preprocessing import StandardScaler

scaler_cols = [
    "Age",
    "Fare",
    "Name_Length",
    "Family_Size",
    "Ticket_Frequency",
    "Fare_Family_Size",
    "Fare_Cat_Pclass",
]
std = StandardScaler()
std.fit(full_df[scaler_cols])
df_std = pd.DataFrame(
    std.transform(full_df[scaler_cols]), index=full_df.index, columns=scaler_cols
)
full_df.drop(scaler_cols, axis=1, inplace=True)
full_df = pd.concat((full_df, df_std), axis=1)

# Final data
features = ["Survived", "Family_Friends_Surv_Rate", "Surv_Rate_Invalid", "Age_Band"]
df = full_df.copy()
df.loc[df.PassengerId.isin(full_df_model.PassengerId), features] = full_df_model[
    features
]
passenger_list = full_df_model["PassengerId"].tolist()
full_df_model = df[df["PassengerId"].isin(passenger_list)]
df_train_final = full_df[: train_shape[0]]
df_test_final = full_df[train_shape[0] :]
df_test_final.drop(["Survived"], axis=1, inplace=True)
passenger_train = df_train_train["PassengerId"].tolist()
df_train = full_df_model[full_df_model["PassengerId"].isin(passenger_train)]
passenger_test = df_train_test["PassengerId"].tolist()
df_test = full_df_model[full_df_model["PassengerId"].isin(passenger_test)]
df_test.loc[
    df_test.PassengerId.isin(y_train_test.PassengerId), "Survived"
] = y_train_test["Survived"]
X_train = df_train.drop(["Survived", "PassengerId"], axis=1)
y_train = df_train["Survived"]
X_test = df_test.drop(["Survived", "PassengerId"], axis=1)
y_test = df_test["Survived"]
all_passenger = passenger_train + passenger_test
df_train_final = full_df[full_df["PassengerId"].isin(all_passenger)]
df_test_final = full_df[~full_df["PassengerId"].isin(all_passenger)]

# Modelling
ada_boost = ensemble.AdaBoostClassifier()
ada_boost.fit(X_train, y_train)
prediction = ada_boost.predict(X_test)
ada_boost_score = get_accuracy(prediction)

bagging = ensemble.BaggingClassifier()
bagging.fit(X_train, y_train)
prediction = bagging.predict(X_test)
bagging_score = get_accuracy(prediction)

gradient_boosting = ensemble.GradientBoostingClassifier()
gradient_boosting.fit(X_train, y_train)
prediction = gradient_boosting.predict(X_test)
gradient_boosting_score = get_accuracy(prediction)

extra_trees = ensemble.ExtraTreesClassifier()
extra_trees.fit(X_train, y_train)
prediction = extra_trees.predict(X_test)
extra_trees_score = get_accuracy(prediction)

random_forest = ensemble.RandomForestClassifier()
random_forest.fit(X_train, y_train)
prediction = random_forest.predict(X_test)
random_forest_score = get_accuracy(prediction)

gaussian_pr = gaussian_process.GaussianProcessClassifier()
gaussian_pr.fit(X_train, y_train)
prediction = gaussian_pr.predict(X_test)
gaussian_pr_score = get_accuracy(prediction)

logistic_regression_cv = linear_model.LogisticRegressionCV(max_iter=100000)
logistic_regression_cv.fit(X_train, y_train)
prediction = logistic_regression_cv.predict(X_test)
logistic_regression_cv_score = get_accuracy(prediction)

logistic_regression = linear_model.LogisticRegression(random_state=1, max_iter=10000)
logistic_regression.fit(X_train, y_train)
prediction = logistic_regression.predict(X_test)
logistic_regression_score = get_accuracy(prediction)

ridge = linear_model.RidgeClassifierCV()
ridge.fit(X_train, y_train)
prediction = ridge.predict(X_test)
ridge_score = get_accuracy(prediction)

perceptron = linear_model.Perceptron()
perceptron.fit(X_train, y_train)
prediction = perceptron.predict(X_test)
perceptron_score = get_accuracy(prediction)

passive_aggressive = linear_model.PassiveAggressiveClassifier()
passive_aggressive.fit(X_train, y_train)
prediction = passive_aggressive.predict(X_test)
passive_aggressive_score = get_accuracy(prediction)

sdg = linear_model.SGDClassifier()
sdg.fit(X_train, y_train)
prediction = sdg.predict(X_test)
sdg_score = get_accuracy(prediction)

gaussian_nb = naive_bayes.GaussianNB()
gaussian_nb.fit(X_train, y_train)
prediction = gaussian_nb.predict(X_test)
gaussian_nb_score = get_accuracy(prediction)

bernoulli_nb = naive_bayes.BernoulliNB()
bernoulli_nb.fit(X_train, y_train)
prediction = bernoulli_nb.predict(X_test)
bernoulli_nb_score = get_accuracy(prediction)

knn = neighbors.KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
knn_score = get_accuracy(prediction)

svc = svm.SVC(random_state=1, kernel="linear")
svc.fit(X_train, y_train)
prediction = svc.predict(X_test)
svc_score = get_accuracy(prediction)

svc_linear = svm.LinearSVC(random_state=1, max_iter=100000)
svc_linear.fit(X_train, y_train)
prediction = svc_linear.predict(X_test)
svc_linear_score = get_accuracy(prediction)

svc_nu = svm.NuSVC(probability=True)
svc_nu.fit(X_train, y_train)
prediction = svc_nu.predict(X_test)
svc_nu_score = get_accuracy(prediction)

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
prediction = decision_tree.predict(X_test)
decision_tree_score = get_accuracy(prediction)

linear_discriminant = discriminant_analysis.LinearDiscriminantAnalysis()
linear_discriminant.fit(X_train, y_train)
prediction = linear_discriminant.predict(X_test)
linear_discriminant_score = get_accuracy(prediction)

xgboost = XGBClassifier(
    random_state=1,
    objective="binary:logistic",
    n_estimators=10,
    eval_metric="mlogloss",
    use_label_encoder=False,
)
xgboost.fit(X_train, y_train)
prediction = xgboost.predict(X_test)
xgboost_score = get_accuracy(prediction)

keras = wrappers.scikit_learn.KerasClassifier(
    build_fn=create_model, epochs=600, batch_size=32, verbose=0
)
keras.fit(X_train, y_train)
prediction = keras.predict(X_test)
keras_score = get_accuracy(prediction)

model_performance = pd.DataFrame(
    {
        "Model": [
            "Ada Boost",
            "Bagging",
            "Keras",
            "XGBClassifier",
            "Linear Discriminant Analysis",
            "Extra Tree",
            "Decision Tree",
            "SVM Nu",
            "SVM Linear",
            "SVM",
            "kNN",
            "Bernoulli Naive Bayes",
            "Gaussian Naive Bayes",
            "SDG",
            "Passive Aggressive",
            "Perceptron",
            "Ridge",
            "Logistic Regression",
            "Logistic Regression CV",
            "Gaussian Process",
            "Random Forest",
            "Gradient Boosting",
        ],
        "Accuracy": [
            ada_boost_score,
            bagging_score,
            keras_score,
            xgboost_score,
            linear_discriminant_score,
            extra_trees_score,
            decision_tree_score,
            svc_nu_score,
            svc_linear_score,
            svc_score,
            knn_score,
            bernoulli_nb_score,
            gaussian_nb_score,
            sdg_score,
            passive_aggressive_score,
            perceptron_score,
            ridge_score,
            logistic_regression_score,
            logistic_regression_cv_score,
            gaussian_pr_score,
            random_forest_score,
            gradient_boosting_score,
        ],
    }
)

model_performance = model_performance.sort_values(by="Accuracy", ascending=False)

# Stack
estimators = [
    ("Gaussian Process", gaussian_pr),
    ("Linear Discriminant", linear_discriminant),
    ("kNN", knn),
]
stack = ensemble.StackingClassifier(estimators=estimators)
stack.fit(X_train, y_train)
prediction = stack.predict(X_test)
stack_score = get_accuracy(prediction)

# Voting
voting = ensemble.VotingClassifier(
    estimators=[
        ("Gaussian Process", gaussian_pr),
        ("Linear Discriminant Analysis", linear_discriminant),
        ("SVM Nu", svc_nu),
        ("Knn", knn),
    ],
    voting="hard",
)
voting.fit(X_train, y_train)
prediction = voting.predict(X_test)
voting_score = get_accuracy(prediction)

# Tunning
rf_clf = linear_model.LogisticRegression(random_state=1)
parameters = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": np.logspace(1, -1),
    "solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
}
grid_cv = model_selection.GridSearchCV(
    rf_clf, parameters, scoring=make_scorer(accuracy_score)
)
grid_cv = grid_cv.fit(X_train, y_train)
best_estimator = grid_cv.best_estimator_
best_score = grid_cv.best_score_
best_params = grid_cv.best_params_

# Submission
all_passenger = passenger_train + passenger_test
df_train_final = full_df[full_df["PassengerId"].isin(all_passenger)]
X_train = df_train_final.drop(["PassengerId", "Survived"], axis=1)
y_train = df_train_final["Survived"]
df_test_final = full_df[~full_df["PassengerId"].isin(all_passenger)]
X_test = df_test_final.drop(["PassengerId", "Survived"], axis=1)

# Keras
keras = wrappers.scikit_learn.KerasClassifier(
    build_fn=create_model, epochs=600, batch_size=32, verbose=0
)
keras.fit(X_train, y_train)
prediction = keras.predict(X_test)
y_pred = []
for y in prediction:
    y_pred.append(y[0])

# Stack
gaussian_pr = gaussian_process.GaussianProcessClassifier()
gaussian_pr.fit(X_train, y_train)

linear_discriminant = discriminant_analysis.LinearDiscriminantAnalysis()
linear_discriminant.fit(X_train, y_train)

knn = neighbors.KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)

estimators = [
    ("Gaussian Process", gaussian_pr),
    ("Linear Discriminant", linear_discriminant),
    ("kNN", knn),
]

stack = ensemble.StackingClassifier(estimators=estimators)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

# Voting
voting = ensemble.VotingClassifier(
    estimators=[
        ("Gaussian Process", gaussian_pr),
        ("Linear Discriminant Analysis", linear_discriminant),
        ("SVM Nu", svc_nu),
        ("Knn", knn),
    ],
    voting="hard",
)

voting.fit(X_train, y_train)
prediction = voting.predict(X_test)
y_pred = stack.predict(X_test)

# Final Submission
submission = pd.DataFrame(
    {"PassengerId": df_test_final["PassengerId"], "Survived": y_pred}
)
submission.Survived = submission.Survived.astype(int)
submission.to_csv(r"../data/submission.csv", index=False)
