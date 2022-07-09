# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 00:48:40 2022

@author: bindu
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt

data_sample = pd.read_excel("C:/Users/bindu/OneDrive/Desktop/Data_Science_Course/0. Project/estimating_delivery_of sample_collection/model2/final_data.xlsx")

data_sample.describe()
data_sample.info()

data = data_sample.drop(["Unnamed: 0","Agent_ID",
                "Latitudes_and_Longitudes_Patient",
                  "Latitudes_and_Longitudes_Agent",
                  "Latitudes_and_Longitudes_DiagnosticCenter",
                  "shortest_distance_Patient_Pathlab(m)",
                  "shortest_distance_Agent_Pathlab(m)",
                  "Age",
                  "Gender","Test_Booking_Date","Test_Booking_Time_HH:MM",
                  " Time_For_Sample_Collection_MM",
                  "Time_Agent_Pathlab_sec","pincode",
                  "Agent_Arrival_Time_range_HH:MM"
                  ],axis =1)

labelencoder = LabelEncoder()
data['patient_location']= labelencoder.fit_transform(data['patient_location'])
data['Diagnostic_Centers']= labelencoder.fit_transform(data['Diagnostic_Centers'])
data['TimeSlot']= labelencoder.fit_transform(data['TimeSlot'])
data['Availabilty_time_Patient']= labelencoder.fit_transform(data['Availabilty_time_Patient'])

data1 = data[["patient_location","Diagnostic_Centers",
              "Availabilty_time_Patient",
              "shortest_distance_Patient_Agent(m)","TimeSlot",
              "Exact_Arrival_Time_MM"]]

data1 = data1.iloc[:,[5,0,1,2,3,4]]


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(data1, test_size = 0.20)

train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

# predict on train

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

# predict on train
pred_train_rbf = model_rbf.predict(train_X)
np.mean(pred_train_rbf==train_y)


np.mean(pred_test_rbf==test_y)

#####################################################################
# saving the model
# importing pickle
import pickle
pickle.dump(model_rbf, open('model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(data1.iloc[0:1,1:])
list_value

print(model.predict(list_value))




