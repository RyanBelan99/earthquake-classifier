from PerceptronClassifier import PerceptronClassifier
from LogisticClassifier import LogisticClassifier
from random import choice, uniform
class NeuralNetwork:
    class Connection:
        def __init__(self, weight, dst):
            self.weight = weight
            self.src = None
            self.dst = dst
            
    class Neuron:
        def __init__(self, weight, classType):
            self.weight = weight
            self.value = None
            self.classType = classType
            self.outgoingConnections = []
            self.incomingConnections = []
            self.error = 0.0
            
        def eval(self):
            inputValue = self.weight
            for conn in self.incomingConnections:
                inputValue += conn.src.value*conn.weight
            self.value = self.classType.threshold(None, inputValue)
            
        def update(self, alpha):
            self.weight = self.weight + alpha * self.error 
            for conn in self.incomingConnections:
                conn.weight = conn.weight + alpha * self.error * conn.src.value
                
        
            
    def __init__(self, width, depth, inputSize, outputSize, classType):
        self.inputs = [self.Neuron(None, None) for i in range(inputSize)]
        self.hiddenNeurons = []
        pastNeurons = self.inputs
        for i in range(depth+1):
            newNeurons = []
            for j in range(width if i != depth else outputSize):
                # Small random bias/weights: zero-initialising every weight makes
                # all neurons in a layer identical (symmetry), so the network can
                # never learn distinct features.
                newNeuron = (self.Neuron(uniform(-0.5, 0.5), classType))
                newNeuron.incomingConnections = [self.Connection(uniform(-0.5, 0.5), newNeuron) for pastNeuron in pastNeurons]
                newNeurons.append(newNeuron)
            for j in range(len(pastNeurons)):
                connections  = []
                for newNeuron in newNeurons:
                    conn = newNeuron.incomingConnections[j]
                    conn.src = pastNeurons[j]
                    connections.append(conn)
                pastNeurons[j].outgoingConnections = connections
            pastNeurons = newNeurons
            self.hiddenNeurons.append(newNeurons)
        self.outputNeurons = pastNeurons
            
            

#         Runs Feed Foward calculation on neural network
    def feedFoward(self, example):
#         Assigns the input neurons their values
        curLevel = self.inputs
        for i in range(len(example)):
            curLevel[i].value = example[i]
        
#          Runs the network for its length
        startNeuron = curLevel[0]
        while True:   
            for conn in startNeuron.outgoingConnections:
                conn.dst.eval()
                
            if startNeuron.outgoingConnections == []:
                break
            startNeuron = startNeuron.outgoingConnections[0].dst
            
#         Finds the solution to the calc
        output = []
        for neuron in self.outputNeurons:
            output.append(neuron.value)
        return max(range(len(output)), key=output.__getitem__)
    
            
    def backProp(self, y, alpha):
        # Hidden errors are accumulated with += below, so they must be cleared
        # before each example or they would compound across the whole run.
        for layer in self.hiddenNeurons:
            for neuron in layer:
                neuron.error = 0.0
        for index, neuron in enumerate(self.outputNeurons):
            # Keep the SIGN of (target - output): abs() would make every error
            # positive, so weights could only ever move in one direction.
            target = 1 if index == y else 0
            neuron.error = (target - neuron.value) * neuron.classType.transfer_derivative(None, neuron.value)
        curLevel = self.outputNeurons
#         Progagate errors back
        while True:
            newLevel = []
            for i in range(len(curLevel)):
                for conn in curLevel[i].incomingConnections:
                    if i == 0:
                        newLevel.append(conn.src)
                    conn.src.error += conn.weight*conn.dst.error * neuron.classType.transfer_derivative(None, conn.src.value)
            curLevel = newLevel
            if curLevel == []:
                break
        
#         Recalculate weights
        startNeuron = self.inputs[0]
        while True:
            for conn in startNeuron.outgoingConnections:
                conn.dst.update(alpha)
                
            if startNeuron.outgoingConnections == []:
                break
            startNeuron = startNeuron.outgoingConnections[0].dst
        
            
    def update(self, x, y, alpha):
        index_max = self.feedFoward(x)
        self.backProp(y, alpha)
        
    def train(self, examples, nsteps, schedule):
        for i in range(nsteps):
            n = choice(examples)
            if isinstance(schedule, float):
                self.update(n[0], n[1], schedule)
            else:
                self.update(n[0], n[1], schedule.alpha(i))
            
    def test(self, examples, nsteps):
        correct = 0
        for i in range(nsteps):
            n = choice(examples)

            if n[1] == self.feedFoward(n[0]):
                correct += 1
        return  float(correct)/nsteps

    # Accuracy over the full example set (deterministic, every example once).
    def accuracy(self, examples):
        correct = sum(1 for x, y in examples if self.feedFoward(x) == y)
        return correct / len(examples)


def load(path):
    examples = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            label = int(float(parts.pop()))
            features = [float(i) for i in parts]
            examples.append([features, label])
    return examples


def main():
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    examples = load(os.path.join(data_dir, "earthquake-clean.data.txt"))

    # One hidden layer of 6 units; 2 seismic features in, 2 classes out
    # (0 = earthquake, 1 = nuclear explosion).
    net = NeuralNetwork(6, 1, 2, 2, LogisticClassifier)
    print("Accuracy before training: {:.1%}".format(net.accuracy(examples)))
    net.train(examples, 20000, 0.1)
    print("Accuracy after training:  {:.1%}".format(net.accuracy(examples)))


if __name__ == '__main__':
    main()