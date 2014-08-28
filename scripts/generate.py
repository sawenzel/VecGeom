#!/usr/bin/env python3
import sys

function = "Contains"
shapes = ["Box", "Orb", "Trapezoid", "Trd1", "Parallelepiped", "Tube - no rmin no phi", "Tube - rmin and phi > PI"]
factors = ["ROOT", "Specialized", "Vectorized", "USolids", "Geant4"]

def fetch(source, factor, shape, func):
    activated = False
    for line in source:
        line = line.split(",")
        if line[0] == shape:
            activated = True
            header = line
        if line[0] == func and activated:
            pos = header.index(factor)
            try:
                val = float(line[pos])
                return val
            except:
                return 0
    return 0

def print_factor(factor, source):
    values = []
    for shape in shapes:
        if factor == "Geant4" and function == "Contains":
            values += [fetch(source, factor, shape, "Inside")]
        else:
            values += [fetch(source, factor, shape, function)]
    print(",".join([str(x) for x in values]))


def main():
    with open(sys.argv[1]) as f:
        source = f.readlines()
        source = [x.strip() for x in source]

    print(function)
    print("{0},{1}".format(len(shapes), len(factors)))
    print(",".join(shapes))
    print(",".join(factors))

    for factor in factors:
        print_factor(factor, source)

    # print(source)

if __name__ == "__main__":
    main()
