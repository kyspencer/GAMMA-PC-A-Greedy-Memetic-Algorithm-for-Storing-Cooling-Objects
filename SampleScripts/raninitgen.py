# raninitgen.py
#    Function to generate a random generation based on imported objects
#    Each member of the generation is a random combination, returns a
#    list of genotypes.
#    Author: Kristina Spencer
#    Date: March 11, 2016

def main():
    print("Caution: You may need to modify this program if using it by itself.")
    file = input("Please enter the location/name of the file w/ input data: ")
    infile = open(file, "r")
    popsize = input("Please enter the number of members in a generation: ")
    n = input("Please enter the number of items to be sorted: ")
    objects = []
    for line in infile:
        objects.append(float(line))
    infile.close()
    genone = initialp(popsize, objects)


def initialp(popsize, objects):
    # This module takes a sequence of objects and generates random
    # combinations to make an initial random population.
    # Note: objects should have indices associated with them.
    # i = object index
    # j = generation member index
    import random
    random.seed(52)
    solutions = []
    for i in range(popsize):
        x = random.sample(objects, len(objects))
        solutions.append(x)
    return solutions


if __name__ == '__main__':
    main()
