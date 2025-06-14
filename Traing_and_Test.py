import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

#загружаем тренировочный датасет
train_df = pd.read_csv('Training.csv')
train_df = train_df.drop('Unnamed: 133', axis = 1)
train_df.head()

train_df.info()
train_df.describe()
#загружаем тестовый датасет
test_df = pd.read_csv("Testing.csv")
test_df.head(3)

#о
X_train, y_train = train_df.drop('prognosis', axis = 1), train_df['prognosis']

X_test, y_test = test_df.drop('prognosis', axis = 1), test_df['prognosis']

Model = LogisticRegression(multi_class = 'multinomial')
Model.fit(X_train, y_train)

y_pred_train = Model.predict(X_train)
y_pred_test = Model.predict(X_test)

#Вывод аккуратность
print(f'Accuracy Train: {accuracy_score(y_train, y_pred_train):.4f}')
print(f'Accuracy Test: {accuracy_score(y_test, y_pred_test):.4f}')