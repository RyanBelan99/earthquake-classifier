from random import choice
#import matplotlib.pyplot as plt

class LinearClassifier:
    def __init__(self, weights):
        #weights is an array of doubles
        if isinstance(weights, list):
            self.weights = weights
        else:
            self.weights = [0.0 for i in range(weights+1)]

    # 	 Update the weights of this LinearClassifer using the given
    # 	 inputs/output example and learning rate alpha.
    def update(self, x, y, alpha):
        a = [1] + x
        newWeights = []
        for i in range(len(a)):
            newWeights.append(self.weights[i] + alpha* self.learningRule(x, y)* a[i])
        for i in range(len(a)):
            self.weights[i] = newWeights[i]


    #    Evaluates the dot product of input x and the weights
    def dotProd(self, x):
        a = [1] + x
        if len(a) != len(self.weights):
            return None
        return sum(i[0] * i[1] for i in zip(a, self.weights))


    # 	 Evaluate the given input vector using this LinearClassifier
    # 	 and return the output value.
    def eval(self, x):
        return self.threshold(self.dotProd(x))

    class Schedule:
        def alpha(self, t):
            return 1000.0/(1000.0+t)


# 	Train this LinearClassifier on the given Examples for the
# 	given number of steps, using given learning rate schedule.

# CHANGE SCHEDULE BROOOO
    def train(self, examples, nsteps, schedule, file):
        for i in range(1, nsteps):
            n = choice(examples)
            if isinstance(schedule, float):
                self.update(n[0], n[1], schedule)
            else:
                self.update(n[0], n[1], schedule.alpha(i))
            self.trainingReport(examples, i, nsteps, file)

    def squaredErrorPerSample(self, examples):
        sum = 0.0
        for ex in examples:
            error = ex[1] - self.eval(ex[0])
            sum += error * error
        return sum / len(examples)

    def accuracy(self, examples):
        ncorrect = 0.0
        for ex in examples:
            if self.eval(ex[0]) == ex[1]:
                ncorrect = ncorrect + 1
        return ncorrect / len(examples)