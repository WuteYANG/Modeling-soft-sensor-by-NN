import numpy as np
np.random.seed(0)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.utils import shuffle

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# Import data from csv file
filepath = "C:\\Users\\WU-TE\\Desktop\\Manager\\Python\\Neural Network\\Processed_09142020.csv"
data = pd.read_csv(filepath, header=None, skipfooter=1, engine='python')
data = data.values
#print(data)
X = data[0][:]
Y = data[1][:]
# Shuffle data
X, Y = shuffle(X, Y, random_state=0)
# Partition the data into training and testing set
X_train, Y_train = X[:80], Y[:80]  # Training data is the first 80
X_test, Y_test = X[80:], Y[80:]    # Testing data is the last 20


"""Closed form solution"""
n = len(X_train)
A = np.zeros((n,2))
for i in range(n):
    A[i][0] = 1
    A[i][1] = X_train[i]
weights = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.transpose(A)), Y_train)



"""Neural Network solution"""
# Building a neural network
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# Choosing loss function and optimizing method
model.compile(loss='mse', optimizer='adam')

# Training
print("Training----------------------------------------")
for step in range(801):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 30 == 0:
        print('Training cost: ', cost)

# Testing
print("\nTesting--------------------------------------")
cost = model.evaluate(X_test, Y_test, batch_size=20)
print('Testing cost:', cost)
W, b = model.layers[0].get_weights()
print()
print("Closed form solution:", "\nWeight:", weights[1], "\nbias:", weights[0])
print()
print("Neural Network solution: ","\nWeights: ", float(W), "\nbias: ", float(b))

# Plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X, Y)
plt.plot(X_test, Y_pred, 'm')
plt.show()
