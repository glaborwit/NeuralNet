# Tutorial from https://www.youtube.com/watch?v=h3l4qz76JhQ

import numpy as np

# sigmoid function
def nonLin(x, deriv=False):
    if (deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))     # else


# the training data: 4 examples
inputs = np.array([ [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1] ])
outputs = np.array([ [0], [1], [1], [0] ])

np.random.seed(1)   # start "random" numbers seed at 1 so that every time the program runs it gives same numbers
                    # good for debugging

# synapses between neurons: 2 synapses because 3 layers to network
syn0 = (2 * np.random.random((3,4)) - 1)
syn1 = (2 * np.random.random((4,1)) - 1)

# weights assigned to synapses
for i in xrange(60000):
    layer0 = inputs

    # matrix mult on each layer and it's synapse; runs the sigmoid function on first layer, layer0, to create next layer1
    layer1 = nonLin(np.dot(layer0, syn0))

    # layer2 predicts output data
    layer2 = nonLin(np.dot(layer1, syn1))

    # calculate error in outputs
    layer2_error = outputs - layer2
    # printing out error rate to make sure it's decreasing
    if ((i % 10000) == 0):
        print("Error: " + str( np.mean( np.abs(layer2_error) ) ))

    # mult. error rate by sigmoid function; gets derivative of our output prediction function from layer 2
    layer2_delta = layer2_error * nonLin(layer2, deriv=True)


    # BACKPROPAGATION: how much did layer1 contribute to the error in layer2?
    layer1_error = layer2_delta.dot(syn1.T)

    # layer 1's delta =  layer 1's error * sigmoid function; ends up being derivative of layer 1 (?)
    layer1_delta = layer1_error * nonLin(layer1, deriv=True)


    # Now we use our deltas to update each synapse's weight to be more accurate
    # uses algorithm called "gradient descent"
    syn1 += layer1.T.dot(layer2_delta)
    syn0 += layer0.T.dot(layer1_delta)

print("\n")
print("Post-training output")
print(layer2)