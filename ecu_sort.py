import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Car_Info.csv")
df = df.drop(df.columns[[2,3,5,11,15,16,18,20,21,24,25,27,28,29,30,31,32]], axis=1)
df.dropna(axis=0, how='all', inplace = True)
print(df.head())
print(df.shape)
df['MAF'] = df['MAF'].str.replace(",",".")
df['ENGINE_LOAD'] = df['ENGINE_LOAD'].str.replace(",",".")
df['FUEL_LEVEL'] = df['FUEL_LEVEL'].str.replace(",",".")
df['ENGINE_POWER'] = df['ENGINE_POWER'].str.replace(",",".")
df['TIMING_ADVANCE'] = df['TIMING_ADVANCE'].str.replace(",",".")

df['MAF'] = df['MAF'].str.replace("%"," ")
df['ENGINE_LOAD'] = df['ENGINE_LOAD'].str.replace("%"," ")
df['FUEL_LEVEL'] = df['FUEL_LEVEL'].str.replace("%"," ")
df['THROTTLE_POS'] = df['THROTTLE_POS'].str.replace("%"," ")
df['TIMING_ADVANCE'] = df['TIMING_ADVANCE'].str.replace("%"," ")
df['VEHICLE_ID'] = df['VEHICLE_ID'].str.replace("car"," ")

df['ENGINE_RUNTIME'] = pd.to_timedelta(df['ENGINE_RUNTIME'])
df['ENGINE_RUNTIME'] = df['ENGINE_RUNTIME'].dt.total_seconds()

df["MAF"] = pd.to_numeric(df["MAF"])
df["ENGINE_LOAD"] = pd.to_numeric(df["ENGINE_LOAD"])
df["FUEL_LEVEL"] = pd.to_numeric(df["FUEL_LEVEL"])
df["ENGINE_POWER"] = pd.to_numeric(df["ENGINE_POWER"])
df["VEHICLE_ID"] = pd.to_numeric(df["VEHICLE_ID"])
df["THROTTLE_POS"] = pd.to_numeric(df["THROTTLE_POS"])
df["ENGINE_RUNTIME"] = pd.to_numeric(df["ENGINE_RUNTIME"])
df["TIMING_ADVANCE"] = pd.to_numeric(df["TIMING_ADVANCE"])

lb_make = LabelEncoder()
df["MARK"] = df["MARK"].astype('category')
df["MARK"] = df["MARK"].cat.codes

a = df['MAF'].astype('float64').mean(axis=0)
df['MAF'].replace(np.nan, a, inplace= True)

b = df['ENGINE_LOAD'].astype('float64').mean(axis=0)
df['ENGINE_LOAD'].replace(np.nan, b, inplace= True)

c = df['FUEL_LEVEL'].astype('float64').mean(axis=0)
df['FUEL_LEVEL'].replace(np.nan, c, inplace= True)

d = df['ENGINE_POWER'].astype('float64').mean(axis=0)
df['ENGINE_POWER'].replace(np.nan, d, inplace= True)

e = df['VEHICLE_ID'].astype('int64').mean(axis=0)
df['VEHICLE_ID'].replace(np.nan, e, inplace= True)

f = df['THROTTLE_POS'].astype('float64').mean(axis=0)
df['THROTTLE_POS'].replace(np.nan, f, inplace= True)

g = df['ENGINE_RUNTIME'].astype('float64').mean(axis=0)
df['ENGINE_RUNTIME'].replace(np.nan, g, inplace= True)

h = df['TIMING_ADVANCE'].astype('float64').mean(axis=0)
df['TIMING_ADVANCE'].replace(np.nan, h, inplace= True)

i = df['TIMESTAMP'].astype('float64').mean(axis=0)
df['TIMESTAMP'].replace(np.nan, i, inplace= True)

j = df['BAROMETRIC_PRESSURE(KPA)'].astype('float64').mean(axis=0)
df['BAROMETRIC_PRESSURE(KPA)'].replace(np.nan, j, inplace= True)

k = df['ENGINE_COOLANT_TEMP'].astype('float64').mean(axis=0)
df['ENGINE_COOLANT_TEMP'].replace(np.nan, k, inplace= True)

l = df['ENGINE_RPM'].astype('float64').mean(axis=0)
df['ENGINE_RPM'].replace(np.nan, l, inplace= True)

m = df['AIR_INTAKE_TEMP'].astype('float64').mean(axis=0)
df['AIR_INTAKE_TEMP'].replace(np.nan, m, inplace= True)

n = df['SPEED'].astype('float64').mean(axis=0)
df['SPEED'].replace(np.nan, n, inplace= True)

o = df['VEHICLE_ID'].astype('float64').mean(axis=0)
df['VEHICLE_ID'].replace(np.nan, o, inplace= True)

p = df['INTAKE_MANIFOLD_PRESSURE'].astype('float64').mean(axis=0)
df['INTAKE_MANIFOLD_PRESSURE'].replace(np.nan, p, inplace= True)

def Car_Status(c):
  if c['FUEL_LEVEL'] < 30:
    return 0
  elif c['ENGINE_RPM'] > 1700 and c['FUEL_LEVEL'] > 30:
    return 1
  elif c['AIR_INTAKE_TEMP'] > 45 and c['ENGINE_RPM'] < 1700 and c['FUEL_LEVEL'] > 30:
    return 2
  elif c['SPEED'] > 50 and c['AIR_INTAKE_TEMP'] < 45 and c['ENGINE_RPM'] < 1700 and c['FUEL_LEVEL'] > 30:
    return 3
  elif c['ENGINE_LOAD'] > 45 and c['SPEED'] < 50 and c['AIR_INTAKE_TEMP'] < 45 and c['ENGINE_RPM'] < 1700 and c['FUEL_LEVEL'] > 30:
    return 4
  else:
    return 5

df['Car_Status'] = df.apply(Car_Status, axis=1)
print(df.dtypes)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("Modified_Car_Info.csv")
print("File Saved")

y = df.pop("Car_Status").values
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print("Train features shape")
print(X_train.shape)
print("Test features shape")
print(X_test.shape)
print("Test features shape")
print(y_train.shape)
print("Test labels shape")
print(y_test.shape)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
print("Training dataset accuracy:")
print(clf.score(X_train,y_train))
predict_percent = 0
prediction = clf.predict(X_test)
print("Accuracy on testing dataset:")
print(accuracy_score(y_test, prediction))
predict1 = clf.predict(X_test[0])
print(np.array(predict1))
#print(np.array(y_test))


