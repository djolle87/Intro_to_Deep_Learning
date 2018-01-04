import numpy as np

# Step 1 Collect Data
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print(x)

print(y)

# Step 2 build model

num_epochs = 60000

# initialize weights
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

print(syn0)
print(syn1)


def nonlin(x, deriv=False):
    if (deriv == True):
        sigm = 1 / (1 + np.exp(-x))
        return sigm * (1 - sigm)

    return 1 / (1 + np.exp(-x))


# Step 3 Train Model

for j in range(num_epochs):
    # feed forward through layers 0,1, and 2
    k0 = x
    k1 = nonlin(np.dot(k0, syn0))
    k2 = nonlin(np.dot(k1, syn1))

    # how much did we miss the target value?
    k2_error = y - k2

    if (j % 10000) == 0:
        print("Error: {}".format(str(np.mean(np.abs(k2_error)))))

    # in what direction is the target value?
    k2_delta = k2_error * nonlin(k2, deriv=True)

    # how much did each k1 value contribute to k2 error
    k1_error = k2_delta.dot(syn1.T)

    k1_delta = k1_error * nonlin(k1, deriv=True)

    syn1 += k1.T.dot(k2_delta)
    syn0 += k0.T.dot(k1_delta)
