import numpy
import scipy.special

class getNetworked:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """Set NN size and learning rate"""
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0,5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        hidden_inputs = numpy.dot(self.wih,inputs)
        self.activation_function = lambda x: scipy.special.expit(x)


    def supertrain(self):
        pass

    def query(self, inputs_list):
        "conv input into2darray"
        inputs = numpy.array(inputs_list, ndmin=2).T


y = getNetworked(3,3,3,0.1)
print(y.wih)
print(y.who)