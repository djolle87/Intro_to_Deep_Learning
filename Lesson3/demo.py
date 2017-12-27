from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection
        # we assign random weights to a 3 x 1 matrix, with values on the range -1 to 1
        # and mean 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The sigmoid function, which describes an s shaped curve
    # we pass the weighted sum of the inputes through this function
    # to normalize the between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # # gradient of the sigmoid curve
    # def __sigmoid_derivative(self, x):
    #     return x * (1 - x)

    # gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        sigm = self.__sigmoid(x)
        return sigm * (1 - sigm)

    def train(self, training_set_inputes, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.predict(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        # Pass inputs through our neural network.
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    # Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.predict(array([1, 0, 0])))
