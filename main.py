import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from flask import session

def train(input):
    df = pd.read_csv('cleaned_data.csv')

    df.loc[-1] = input
    
    # Specify which columns to one-hot encode
    categorical_columns = ['Platform', 'Watch Reason', 'DeviceType']
    target_column = 'ProductivityLoss'

    # Separate the categorical columns and the rest of the DataFrame
    df_categorical = df[categorical_columns]
    df_numerical = df.drop(columns=categorical_columns)

    # Apply one-hot encoding to the categorical columns
    df_categorical_encoded = pd.get_dummies(df_categorical, drop_first=True)
    print(df_categorical_encoded)

    # Concatenate the encoded categorical columns with the numerical columns
    df_preprocessed = pd.concat([df_numerical, df_categorical_encoded], axis=1)

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_columns = df_numerical.columns
    df_preprocessed[numerical_columns] = scaler.fit_transform(df_numerical)

    # Separate features and target
    X = df_preprocessed
    Y = df[target_column]

    input = df_preprocessed.iloc[-1]
    
    X = X[:-1]    
    Y = Y[:-1].apply(str)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=0.8, shuffle=True)


    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"Y_train shape: {Y_train.shape}")
    # print(f"Y_test shape: {Y_test.shape}")

    ## Model #######################################################
    logreg = LogisticRegression(random_state=42, max_iter=1000000)

    # fit the model with data
    logreg.fit(X_train, Y_train)


    y_pred = logreg.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    conf_matrix = confusion_matrix(Y_test, y_pred)
    class_report = classification_report(Y_test, y_pred)


    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return logreg, input


# # model,input = train(np.array(['TikTok','Smartphone',9000,'Procrastination',4, 0]))
# model, input = train(np.array([session.get('platform'), session.get('device_type'), session.get('time_spent'), 
#                                session.get('watch_reason'), session.get('addiction_level')]))

# final = input.to_frame()

# print("~~~~~~~~~~~~~~~ PREDICTED LOSS IN PRODUCTIVITY ~~~~~~~~~~~~~~~~~~~")
# print(model.predict(final.T))
