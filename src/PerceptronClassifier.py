from LinearClassifier import LinearClassifier

class PerceptronClassifier(LinearClassifier):
    def learningRule(self, x, y):
        return y - self.threshold(self.dotProd(x))

    def threshold(self, x):
        return 1.0 if x >= 0.0 else 0.0

    def trainingReport(self, examples, stepnum, nsteps, file):
        print(str(stepnum) + "\t\t" + str(self.accuracy(examples)))
        file.write(str(stepnum) + "\t\t" + str(self.accuracy(examples)) + "\n")