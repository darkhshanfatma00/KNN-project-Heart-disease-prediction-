import pandas as pd
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

data_set = pd.DataFrame({
    "age": [25, 32, 45, 51, 60, 29, 40, 55, 67, 36,
            48, 58, 31, 72, 43, 52, 64, 27, 47, 69],

    "resting_bp": [118, 125, 135, 145, 155, 120, 130, 150, 165, 128,
                   140, 148, 122, 170, 138, 142, 158, 115, 136, 162],

    "cholesterol": [180, 195, 220, 245, 260, 185, 210, 250, 280, 205,
                    230, 255, 190, 290, 215, 240, 270, 175, 225, 275],

    "max_heart_rate": [185, 175, 160, 145, 130, 182, 168, 140, 120, 172,
                       155, 135, 178, 115, 165, 150, 125, 188, 158, 118],

    "oldpeak": [0.2, 0.5, 1.0, 1.8, 2.5, 0.3, 0.8, 2.0, 3.1, 0.6,
                1.4, 2.2, 0.4, 3.5, 1.1, 1.7, 2.8, 0.1, 1.3, 3.0],

    "heart_disease": [0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
                      1, 1, 0, 1, 0, 1, 1, 0, 1, 1],

    "medical_cost": [1200, 1500, 2800, 4200, 5500, 1300, 2200, 4800, 6500, 1900,
                     3500, 5200, 1600, 7200, 2400, 3900, 6000, 1100, 3200, 6800]
})

print(data_set)

# Data insertion 

first_data=data_set.head(10)
print(first_data)
last_data=data_set.tail(10)
print(last_data)
shape_of_data=data_set.shape
print(shape_of_data)
missing_values=data_set.isna()
print(missing_values)
count_missing_values=data_set.isna().sum()
print(count_missing_values)
statistics_information=data_set.describe()
print(statistics_information)
data_information=data_set.info()
print(data_information)


# KNN model

# KNN regressor 

pipeline1=Pipeline([
    ("scaler",StandardScaler()),
    ("model1",KNeighborsRegressor(n_neighbors=5))
])
X1=data_set[["age","resting_bp","cholesterol", "max_heart_rate", "oldpeak"]]
Y1=data_set["medical_cost"]

X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=5)
pipeline1.fit(X1_train,Y1_train)
prediction_regression=pipeline1.predict(X1_test)
print(prediction_regression)
mse=mean_squared_error(Y1_test,prediction_regression)
print(mse)
r_sqaure=r2_score(Y1_test,prediction_regression)
print(r_sqaure)

# KNN classifier 

pipeline2=Pipeline([
    ("scaler",StandardScaler()),
    ("model2",KNeighborsClassifier(n_neighbors=5))
])
X2=data_set[["age","resting_bp","cholesterol", "max_heart_rate", "oldpeak"]]
Y2=data_set["heart_disease"]

X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2,Y2,test_size=0.2,random_state=5)
pipeline2.fit(X2_train,Y2_train)
prediction_classifier=pipeline2.predict(X2_test)
print(prediction_classifier)
accuracy=accuracy_score(Y2_test,prediction_classifier)
print(accuracy)
precision=precision_score(Y2_test,prediction_classifier)
print(precision)


