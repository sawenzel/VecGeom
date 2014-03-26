import subprocess as sp, re, pylab as pl, sys

save = False
for arg in sys.argv:
  if arg.lower() == "save":
    save = True

pattern = "Points located on {:s} in ([^s]+)s{:s}."

n_list = []
cpu = []
gpu = []
gpu_m = []

n = 1<<5;
while n < 1<<17:
  p = sp.Popen(['../build/cpugpu', '%i' % n], stdout=sp.PIPE, stderr=sp.PIPE)
  out, err = p.communicate()
  out = str(out)
  if len(err) > 3:
    print("Error reported at %i particles: %s" % (n, err))
    break
  n_list.append(n)
  cpu.append(float(re.search(pattern.format("CPU", ""), out).group(1)))
  gpu.append(float(re.search(pattern.format("GPU", ""), out).group(1)))
  gpu_m.append(float(re.search(pattern.format("GPU", "[^\.+]"),
                               out).group(1)))
  n = n<<1

fig, ax = pl.subplots()
ax.plot(n_list, cpu, "-x", label="CPU time")
ax.plot(n_list, gpu, "-o", label="GPU time")
ax.plot(n_list, gpu_m, "-o", label="GPU time with overhead")
ax.set_xscale("log")
ax.set_yscale("log")
legend = ax.legend(loc="upper left", frameon=None)
pl.title("Particle location in box geometry")
pl.xlabel("Number of particles located")
pl.ylabel("Time [s]")
if save:
  location = "../figures/location_gpugpu.eps"
  print("Saving file to \"%s\"..." % location)
  fig.savefig(location)
else:
  fig.show()
  input("Press enter to continue...")