from PerceptronClassifier import PerceptronClassifier
from LogisticClassifier import LogisticClassifier
from LinearClassifier import LinearClassifier

#Perceptron Training Commands (Earthquake/Explosion)
def percept_Earth(examples, nsteps, schedule, file):
    percClass = PerceptronClassifier(len(examples[0][0]))
    print("Number of Weight Updates" + "\t" + "Proportion Correct")
    earth_fig = open(file, "w")
    earth_fig.write("Number of Weight Updates" + "\t" + "Proportion Correct\n")
    percClass.train(examples, nsteps, schedule, earth_fig)
    earth_fig.close()

# #Perceptron Training Commands (House Votes)
def percept_House(examples, nsteps, schedule, file):
    percClass_haus = PerceptronClassifier(len(examples[0][0]))
    print("Number of Weight Updates" + "\t" + "Proportion Correct")
    house_fig = open(file, "w")
    house_fig.write("Number of Weight Updates" + "\t" + "Proportion Correct\n")
    percClass_haus.train(examples, nsteps, schedule, house_fig)
    house_fig.close()

# #Logistic Training Commands (Earthquake/Explosion)
def logistic_Earth(examples, nsteps, schedule, file):
    logClass = LogisticClassifier(len(examples[0][0]))
    print("Iteration" + "\t" + "Training-Set Squared Error")
    earth_fig = open(file, "w")
    earth_fig.write("Iteration" + "\t" + "Training-Set Squared Error\n")
    logClass.train(examples, nsteps, schedule, earth_fig)
    earth_fig.close()

#Logistic Training Commands (House Votes)
def logistic_House(examples, nsteps, schedule, file):
    logClass_haus = LogisticClassifier(len(examples[0][0]))
    print("Number of Weight Updates" + "\t" + "Proportion Correct")
    house_fig = open(file, "w")
    house_fig.write("Iteration" + "\t" + "Training-Set Squared Error\n")
    logClass_haus.train(examples, nsteps, schedule, house_fig)
    house_fig.close()

def load(filename):
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    examples = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = [float(i) for i in line.split(',')]
            label = parts.pop()
            examples.append([parts, label])
    return examples


def main():
    examples = load("earthquake-clean.data.txt")
    examples_noisy = load("earthquake-noisy.data.txt")
    house_votes = load("house-votes-84.data.num.txt")

    s = LinearClassifier.Schedule()

    training_time = True
    while training_time:
        func = input("Which training run would you like to observe? (1-10, or q to quit) ")
        if func == "1":
            print("\nFigure 18.16(a) info for earthquake/explosion data: ")
            percept_Earth(examples, 10000, 0.5, "earthquake_figure1.txt")
        elif func == "2":
            print("\nFigure 18.16(b) info for earthquake/explosion data: ")
            percept_Earth(examples_noisy, 100000, 0.5, "earthquake_figure2.txt")
        elif func == "3":
            print("\nFigure 18.16(c) info for earthquake/explosion data: ")
            percept_Earth(examples_noisy, 100000, s, "earthquake_figure3.txt")
        elif func == "4":
            print("\nFigure 18.18(a) info for earthquake/explosion data: ")
            logistic_Earth(examples, 5000, 0.5, "earthquake_figure4.txt")
        elif func == "5":
            print("\nFigure 18.18(b) info for earthquake/explosion data: ")
            logistic_Earth(examples_noisy, 100000, 0.5, "earthquake_figure5.txt")
        elif func == "6":
            print("\nFigure 18.18(c) info for earthquake/explosion data: ")
            logistic_Earth(examples_noisy, 100000, s, "earthquake_figure6.txt")
        elif func == "7":
            print("\nFigure 18.16(b) info for house votes: ")
            percept_House(house_votes, 10000, 0.5, "house_figure1.txt")
        elif func == "8":
            print("\nFigure 18.16(c) info for house votes: ")
            percept_House(house_votes, 10000, s, "house_figure2.txt")
        elif func == "9":
            print("\nFigure 18.18(b) info for house votes: ")
            logistic_House(house_votes, 10000, 0.5, "house_figure3.txt")
        elif func == "10":
            print("\nFigure 18.18(c) info for house votes: ")
            logistic_House(house_votes, 10000, s, "house_figure4.txt")
        elif func == "q":
            training_time = False
        else:
            print("Pick a valid function value (an integer between 1 and 10)")


if __name__ == '__main__':
    main()