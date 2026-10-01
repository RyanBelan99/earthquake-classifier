from LinearClassifier import LinearClassifier
from math import e

class LogisticClassifier(LinearClassifier):
    def learningRule(self, x, y):
        h = self.threshold(self.dotProd(x))
        temp = (y - h)*h*(1 - h)
        return temp

    def threshold(self, x):
        return 1.0/(1.0 + e**(-x))

    def trainingReport(self, examples, stepnum, nsteps, file):
        print(str(stepnum) + "\t\t" + str(1-self.squaredErrorPerSample(examples)))
        file.write(str(stepnum) + "\t\t" + str(1-self.squaredErrorPerSample(examples))+ "\n")

    def transfer_derivative(self, output):
        return output*(1.0 - output)