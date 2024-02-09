#Neural Network
from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in range(1000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

#Main Code
print("Matrix Truth Table: ")
print("# ---> The first value in the row is the output")
print("[0, 0, 1] ----------------------- 0 ")
print("[1, 1, 1] ----------------------- 1 ")
print("[1, 0, 1] ----------------------- 1 ")
print("[0, 1, 1] ----------------------- 0 ")
print("")
print("--> In this case, lets see what the neural network predicts using the sigmoid function (1/1+e^-x) ?")
prediction = (1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
print("[1, 0, 0] ----------------------- ? ", prediction)
print("")
acc = (prediction/1) * 100
print("The percentage accuracy of the neural network is: ",acc,"%")