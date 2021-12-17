import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount("/content/drive")
fishing = pd.read_csv("drive/kongkea/Dataset/phish.csv")
fishing.head(5)
fishing = fishing.drop(["id"], axis=1)
fishing.shape
fishing.isnull().sum()
fishing.describe()
fishing["CLASS_LABEL"].value_counts()
fishing_class = fishing.groupby("CLASS_LABEL")
fishing_class["NoHttps"].value_counts()
fishing_class["UrlLength"].mean()
fishing_class["NumPercent"].mean()
fishing_class["NumAmpersand"].mean()
fishing_class["NumHash"].mean()
fishing_class["IpAddress"].value_counts()
plt.figure(figsize=(30, 30))
sns.heatmap(fishing.corr(), annot=True, cmap="viridis", linewidths=0.5)
fishing.columns
subset_fishing = fishing[
    [
        "NumDots",
        "PathLevel",
        "NumDash",
        "NumSensitiveWords",
        "PctExtHyperlinks",
        "PctExtResourceUrls",
        "InsecureForms",
        "PctNullSelfRedirectHyperlinks",
        "FrequentDomainNameMismatch",
        "SubmitInfoToEmail",
        "IframeOrFrame",
        "CLASS_LABEL",
    ]
]
subset_fishing.head()
subset_fishing.shape
y = subset_fishing["CLASS_LABEL"]
X = subset_fishing.drop(["CLASS_LABEL"], axis=1)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)
random_model = RandomForestClassifier(n_estimators=250, n_jobs=-1)
random_model.fit(Xtrain, ytrain)
y_pred = random_model.predict(Xtest)
random_model_accuracy = round(random_model.score(Xtrain, ytrain) * 100, 2)
print(round(random_model_accuracy, 2), "%")
random_model_accuracy1 = round(random_model.score(Xtest, ytest) * 100, 2)
print(round(random_model_accuracy1, 2), "%")

saved_model = pickle.dump(
    random_model, open("drive/kongkea/Dataset/Models/Phishing.pickle", "wb")
)
