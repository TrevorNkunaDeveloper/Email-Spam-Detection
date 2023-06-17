import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle

dataset = pd.read_csv("emails.csv")

dataset = dataset.drop("Email No.", axis=1)

x = dataset.drop("Prediction", axis=1)
y = dataset["Prediction"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()

model.fit(x_train, y_train)

model.score(x_test, y_test)

model = RandomForestRegressor()

model.fit(x_train, y_train)

print("Accuracy: ", model.score(x_test, y_test))

with open('emailSpam_classifier.pkl', 'wb') as file:
    pickle.dump(model, file)
