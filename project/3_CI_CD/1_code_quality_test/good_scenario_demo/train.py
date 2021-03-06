# Import Libraries #

# Mongo Library

import pymongo

# ML libraries

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import sklearn.model_selection as model_selection
from sklearn.ensemble import RandomForestClassifier

# Tool libraries
import joblib

# Connect to mongodb container #

conn = pymongo.MongoClient('mongo', 27017)

# Read data #

db = conn.training
collection = db.churn_data
df = pd.DataFrame(list(collection.find()))
df.drop('_id', axis=1, inplace=True)
df.drop_duplicates(inplace=True)

# Prepare Data #

cols = df[df.columns.difference(['BAD', 'LOAN', 'REASON', 'JOB'])].columns
df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
df['BAD'] = df['BAD'].astype('category')

# Declare Variables #

# target var
target = df.select_dtypes('category').columns
# categorical (nominal and ordinal) variables
class_inputs = list(df.select_dtypes('object').columns)
# input interval variables
numerical_inputs = list(df.select_dtypes(include=['int64', 'float32']).columns)
inputs = class_inputs + numerical_inputs

# Data engineering #

# Impute missings

categorical_imputer = SimpleImputer(
    missing_values='', strategy='most_frequent')
numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Impute categorical variables

categorical_imputer.fit(df[class_inputs])
categorical_imputed = categorical_imputer.transform(df[class_inputs])
df_categorical_imputed = pd.DataFrame(
    data=categorical_imputed, columns=class_inputs)

# Impute numerical variables

numerical_imputer.fit(df[numerical_inputs])
numerical_imputed = numerical_imputer.transform(df[numerical_inputs])
df_numerical_imputed = pd.DataFrame(
    data=numerical_imputed, columns=numerical_inputs)

# One-hot encoding

encoder = OneHotEncoder()
encoder.fit(categorical_imputed)
categorical_encoded = encoder.transform(categorical_imputed)

categories = list(np.hstack(encoder.categories_, ))
df_categorical_encoded = pd.DataFrame(
    data=categorical_encoded.toarray(), columns=categories)

# ABT

abt = pd.concat([df[target], df_numerical_imputed,
                 df_categorical_encoded], axis=1)


# Model Training #

X_train, X_test, y_train, y_test = model_selection.train_test_split(abt[abt.columns.difference(target)],
                                                                    abt[target],
                                                                    test_size=0.33, random_state=27513)

# Build Sklearn Random Forest
rfor = RandomForestClassifier()
rfor.fit(X_train, y_train)

output = open('./experiment_rfor/rfor.pkl', 'wb')
joblib.dump(rfor, output)
output.close()
