import numpy as np
import matplotlib.pyplot as plt  # To visualize

import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.DataFrame()

#df["Name"]=['Bob', 'Joe', 'Mike', 'Shri', "Andy", 'Kami', 'Walter', 'Kate','Ross', 'Phobe']
df['X']=[1,2,3,4,5,6,7,8,9,10]
df['Y']=[10,2,3,6,3,6,7,8,20,30]

feature = df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
target = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(feature, target)  # perform linear regression
target_pred = linear_regressor.predict(feature)  # make predictions

plt.scatter(feature, target)
plt.plot(feature, target_pred, color='red')
plt.show()

