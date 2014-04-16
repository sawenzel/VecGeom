import subprocess as sp, re, pylab as pl, sys, numpy as np, math

save = False
for arg in sys.argv:
  if arg.lower() == "save":
    save = True

pattern = "Points located on {:s} in ([^s]+)s{:s}."

n_particles = np.array([2**x for x in range(5,17)], dtype=np.int)
cpu = np.zeros(len(n_particles), dtype=np.double)
gpu = np.zeros(len(n_particles), dtype=np.double)
gpu_m = np.zeros(len(n_particles), dtype=np.double)

i = 0
for n in n_particles:
  print("Running for %i particles..." % n)
  p = sp.Popen(['../build/cpugpu', '%i' % n], stdout=sp.PIPE, stderr=sp.PIPE)
  out, err = p.communicate()
  out = str(out)
  # if len(err) > 3:
  #   print("Error reported at %i particles: %s" % (n, err))
  #   break
  cpu[i] = (float(re.search(pattern.format("CPU", ""), out).group(1)))
  gpu[i] = (float(re.search(pattern.format("GPU", ""), out).group(1)))
  gpu_m[i] = (float(re.search(pattern.format("GPU", "[^\.+]"), out).group(1)))
  i += 1

gpu_frac = cpu/gpu
gpu_m_frac = cpu/gpu_m

fig, ax = pl.subplots()
ax.plot(n_particles, gpu_frac, "-og", label="GPU")
ax.plot(n_particles, gpu_m_frac, "--xg", label="GPU with overhead")
ax.plot(n_particles, cpu/cpu, "--b", label="CPU")
ax.set_xscale("log")
# ax.set_yscale("log")

ax.set_xlim([n_particles[0], n_particles[-1]])
# all_speeds = np.concatenate((gpu_frac, gpu_m_frac))
# y_min = math.floor(math.log10(min(all_speeds)))
# y_max = math.ceil(math.log10(max(all_speeds)))
# y_lim = max(abs(y_min), abs(y_max))
# ax.set_ylim([10**(-y_lim), 10**(y_lim)])

legend = ax.legend(loc="upper left", frameon=None)
pl.title("Particle location in box geometry")
pl.xlabel("Number of particles located")
pl.ylabel("Speedup")
if save:
  location = "../figures/location_cpugpu.eps"
  print("Saving file to \"%s\"..." % location)
  fig.savefig(location)
else:
  fig.show()
  input("Press enter to continue...")