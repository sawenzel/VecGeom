import matplotlib.pyplot as plt, csv, sys, numpy as np

if len(sys.argv) < 2:
  print("Please provide path to input data file.")
  sys.exit()

read = {}
labels = {}
cols = 0
rows = 0
with open(sys.argv[1]) as datafile:
  reader = csv.reader(datafile)
  headers = reader.__next__()
  cols = len(headers)
  for i in range(cols):
    read[i] = []
    labels[i] = headers[i]
  for row in reader:
    rows += 1
    for i in range(cols):
      read[i].append(row[i])
data = {}
for i in range(cols):
  data[labels[i]] = read[i]

def plot_point_count():
  for method in ["Inside", "DistanceToIn", "SafetyToIn"]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    point_count = []
    elapsed_scalar = []
    elapsed_vector = []
    elapsed_cuda = []
    elapsed_cudamem = []
    for i in range(rows):
      if data["method"][i] == method:
        if data["library"][i] == "Specialized":
          point_count.append(int(data["points"][i]))
          elapsed_scalar.append(float(data["elapsed"][i])/point_count[-1])
        if data["library"][i] == "Vectorized":
          elapsed_vector.append(elapsed_scalar[-1]/(float(data["elapsed"][i])
                                /point_count[-1]))
        if data["library"][i] == "CUDA":
          elapsed_cuda.append(elapsed_scalar[-1]/(float(data["elapsed"][i])
                              /point_count[-1]))
        if data["library"][i] == "CUDAMemory":
          elapsed_cudamem.append(elapsed_scalar[-1]/(float(data["elapsed"][i])
                                 /point_count[-1]))
    ax.plot(point_count, np.ones(len(point_count), dtype=np.float), "--b",
            label="Scalar")
    ax.plot(point_count, elapsed_vector, "-xr", ms=5, label="AVX")
    ax.plot(point_count, elapsed_cuda, "-xg", ms=5, label="CUDA")
    ax.plot(point_count, elapsed_cudamem, "--", color=[0, 1, 0], ms=5,
            label="CUDA with overhead")
    ax.set_xlim(point_count[0], point_count[-1])
    plt.xticks(point_count, [str(x) for x in point_count])
    ybegin, yend = ax.get_ylim()
    ax.set_yticks(np.arange(0, yend, 1.0))
    ax.set_title("Performance of distance to tube algorithm")
    ax.set_xlabel("Input length")
    ax.set_ylabel("Speedup")
    ax.legend(loc="upper left")
    fig.savefig("%s.pdf" % method)

plot_point_count()