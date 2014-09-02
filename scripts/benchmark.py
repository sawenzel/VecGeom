#!/usr/bin/env python3
import subprocess, sys, re, math

# -----
npoints = 1024
nrep= 1000
target = "table.txt"
# ----

def sh(cmd):
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8")

def run(cmd):
    return subprocess.call(cmd, shell=True)

def toBranch(br):
    sh("git checkout {0}".format(br))

def fetch(output, source, function):
    allmatches = []
    for line in str(output).split("\n"):
        if line.startswith(source):
            match = re.search("{0}: (\d+.\d+)".format(function), line)
            if match:
                allmatches += [match.group(1)]
    if len(allmatches) == 0:
        return None
    if len(allmatches) > 1:
        raise Exception("More than one matches: " + " ".join(allmatches))
    return allmatches[0]


def simpleTable(shape, out, f):
    # f.write(shape + "\n")
    functions = ["Inside", "Contains", "DistanceToIn", "DistanceToOut", "SafetyToIn", "SafetyToOut"]
    sources = ["ROOT", "USolids", "Unspecialized", "Specialized", "Vectorized", "Geant4"]

    f.write(shape + "," + ",".join(sources) + "\n")
    for function in functions:
        f.write(function + ",")
        for source in sources:
            val = fetch(out, source, function)
            if val is not None:
                f.write(val)
            f.write(",")
        f.write("\n")
    f.write("\n")

def runall(processor, f):
    sh("cp compile.sh compile-safe.sh")
    toBranch("master")
    run("./compile-safe.sh")

    # ===== Box
    out = sh("./build/BoxBenchmark -npoints {0} -nrep {1} -dx 5 -dy 10 -dz 15".format(npoints, nrep))
    processor("Box", out, f)

    # ===== Tube
    phi = 2*math.pi
    out = sh("./build/TubeBenchmark -npoints {0} -nrep {1} -rmin 0 -rmax 20 -dz 40 -sphi 0 -dphi {2}".format(npoints, nrep, phi))
    processor("Tube - no rmin no phi", out, f)

    phi = 2*math.pi
    out = sh("./build/TubeBenchmark -npoints {0} -nrep {1} -rmin 10 -rmax 20 -dz 40 -sphi 0 -dphi {2}".format(npoints, nrep, phi))
    processor("Tube - rmin no phi", out, f)

    phi = 3*math.pi/2
    out = sh("./build/TubeBenchmark -npoints {0} -nrep {1} -rmin 0 -rmax 20 -dz 40 -sphi 0 -dphi {2}".format(npoints, nrep, phi))
    processor("Tube - no rmin and phi > PI", out, f)

    phi = 3*math.pi/2
    out = sh("./build/TubeBenchmark -npoints {0} -nrep {1} -rmin 10 -rmax 20 -dz 40 -sphi 0 -dphi {2}".format(npoints, nrep, phi))
    processor("Tube - rmin and phi > PI", out, f)

    # ===== Parallelepiped
    run("./compileWithoutUSolids.sh")
    out = sh("./build/ParallelepipedBenchmark -npoints {0} -nrep {1} -dx 3 -dy 3 -dz 3 -alpha 14.9 -theta 39 -phi 3.22".format(npoints, nrep))
    processor("Parallelepiped", out, f)

    # ===== Trd
    toBranch("trd-development")
    run("./compile-safe.sh")
    out = sh("./build/TrdBenchmark -npoints {0} -nrep {1} -dx1 10 -dx2 20 -dy1 30 -dy2 30 -dz 10".format(npoints, nrep))
    processor("Trd1", out, f)

    # ===== Orb
    toBranch("raman/Orb")
    run("./compile-safe.sh")
    out = sh("./build/OrbBenchmark -npoints {0} -nrep {1} -r 3".format(npoints, nrep))
    processor("Orb", out, f)

    # ===== Trapezoid
    toBranch("trap-SOAPlanes")
    run("./compile-safe.sh")
    out = sh("./build/TrapezoidBenchmarkScript -npoints {0} -nrep {1} -dz 15 -p1x -2 -p2x 2 -p3x -3 -p4x 3 -p5x -4 -p6x 4 -p7x -6 -p8x 6 -p1y -5 -p2y -5 -p3y 5 -p4y 5 -p5y -10 -p6y -10 -p7y 10 -p8y 10".format(npoints, nrep))
    processor("Trapezoid", out, f)

    # Back to master
    toBranch("master")
    
def main():
    f = open(target, "w")
    root = sh("git rev-parse --show-toplevel")
    root = root.strip()
    
    runall(simpleTable, f)

    f.close()
    sys.exit(0)

if __name__ == '__main__':
	main()

