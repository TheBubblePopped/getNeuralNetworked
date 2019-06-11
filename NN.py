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


        self.activation_function = lambda x: scipy.special.expit(x) #siegma gabriel func


    def supertrain(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.who.T, output_errors)
        #dwjk = a*ek*sig(ok)*(1-sig(ok))*0j^r
        #update weight for link between hidden and output layer
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),
                                      numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs_list):
        "conv input into2darray"
        print(inputs_list)
        print("flag0")
        o = numpy.array(inputs_list, ndmin=2)
        print(o)
        print("flag1")
        inputs = numpy.array(inputs_list, ndmin=2).T
        print(inputs)

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

y = getNetworked(3,3,3,0.3)
y.query([1.0, 0.5,-1.5])