#include "Frontend.h" // VecGeom/gdml/Frontend.h

#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/BVHNavigator.h"
#include "VecGeom/navigation/LoopNavigator.h"
#include "VecGeom/navigation/GlobalLocator.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iomanip>
#include <err.h>
#include <getopt.h>
#include <libgen.h>

using namespace vecgeom;

static std::random_device rd;
static std::default_random_engine rng;
static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

double uniform(double a, double b) { return a + (b - a) * dist(rng); }

Vector3D<Precision> random_unit_vector()
{
  Precision z = uniform(-1.0f, 1.0f);
  Precision r = sqrt(1.0f - z * z);
  Precision t = uniform(0.0f, 6.2831853f);
  return {r * vecCore::math::Cos<Precision>(t), r * vecCore::math::Sin<Precision>(t), z};
}

bool nearly_equal(double x, double y)
{
  return vecCore::Abs(x - y) < kRelTolerance(std::max(x, y), kToleranceStrict<double>);
}

bool navigate(Vector3D<Precision> p, Vector3D<Precision> dir, const BVHNavigator *navigator,
              const LoopNavigator *ref_navigator, Precision &checksum_add, bool verbose = true)
{
  auto &geoManager      = GeoManager::Instance();
  NavigationState *curr = new NavigationState();
  NavigationState *next = new NavigationState();

  GlobalLocator::LocateGlobalPoint(geoManager.GetWorld(), p, *curr, true);

  LogicalVolume const *curr_volume = curr->Top()->GetLogicalVolume();

  if (verbose) {
    printf("initial conditions:\n\n\t   volume: %s"
           "\n\t position: [ % .8f, % .8f, % .8f ]\n\tdirection: [ % .8f, % .8f, % .8f ]\n\n",
           curr_volume->GetLabel().c_str(), p.x(), p.y(), p.z(), dir.x(), dir.y(), dir.z());
    printf("%6s%25s%-25s%15s%15s%8s\n\n", "step", "", "position", "step length", "reference", "volume");
  }

  size_t steps = 0;
  while (!curr->IsOutside()) {
    curr_volume = curr->Top()->GetLogicalVolume();

    Precision ref_step = ref_navigator->ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);
    Precision step     = navigator->ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);

    if (!nearly_equal(step, ref_step)) {
      // Print info for the step that failed
      printf("Navigation error while testing:\n");
      printf("% 14.20f\n% 14.20f\n", step, ref_step);
      printf("%6zu [ % 14.8f, % 14.8f, % 14.8f ] % 14.8f % 14.8f %s\n", steps, p.x(), p.y(), p.z(), step, ref_step,
             curr_volume->GetLabel().c_str());
      // Call both navigators again with the same parameters, so we can easily reproduce the error in the debugger
      ref_navigator->ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);
      navigator->ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);

      return false;
    }
    step = vecCore::math::Max(step, kTolerance);
    checksum_add += steps * step;

    p = p + step * dir;

    std::swap(curr, next);

    if (verbose)
      printf("%6zu [ % 14.8f, % 14.8f, % 14.8f ] % 14.8f % 14.8f %s\n", steps, p.x(), p.y(), p.z(), step, ref_step,
             curr_volume->GetLabel().c_str());
    steps++;
  }

  if (verbose) printf("\n");

  return true;
}

int main(int argc, char **argv)
{
  bool verbose                       = false;
  bool validate                      = false;
  double mm_unit                     = 0.1;
  double checksum                    = 0.;
  unsigned long seed                 = 0;
  unsigned long iterations           = 1;
  BVHNavigator const *navigator      = new BVHNavigator();
  LoopNavigator const *ref_navigator = new LoopNavigator();

  for (;;) {
    int opt = getopt(argc, argv, "hi:n:s:v");

    if (opt == -1) break;

    switch (opt) {
    case 'i':
      errno      = 0;
      iterations = strtoul(optarg, nullptr, 10);
      if (errno) errx(errno, "%s: %lu", strerror(errno), iterations);
      if (iterations == 0) iterations = 1;
      break;

    case 's':
      errno = 0;
      seed  = atoi(optarg);
      if (errno) errx(errno, "%s: %lu", strerror(errno), seed);
      break;

    case 'v':
      verbose = 1;
      break;

    case 'h':
    default:
      fprintf(stderr, "Usage: %s [-i iterations] [-n navigator] [-s seed] [-v] file.gdml\n", basename(argv[0]));
      return EXIT_FAILURE;
    }
  }

  if (optind == argc) errx(ENOENT, "No input GDML file");

  const char *filename = argv[optind];

  if (!filename || !vgdml::Frontend::Load(filename, validate, mm_unit, /* verbose */ false))
    errx(EBADF, "Cannot open file '%s'", filename);

  auto &geoManager = GeoManager::Instance();

  if (!geoManager.IsClosed()) errx(1, "Geometry not closed");

  BVHManager::Init();

  rng.seed(seed ? seed : seed = rd());

  Stopwatch timer;
  timer.Start();
  for (unsigned long i = 0; i < iterations; ++i) {
    Vector3D<Precision> p(0.0, 0.0, 0.0);
    Vector3D<Precision> dir = random_unit_vector();

    auto success = navigate(p, dir, navigator, ref_navigator, checksum, verbose);
    if (success) continue;

    // Call navigate with verbose=1
    Precision dummy_checksum = 0;
    navigate(p, dir, navigator, ref_navigator, dummy_checksum, 1);
    std::cout << "\nNavigation test for BVHNavigator failed! seed = " << seed << " iteration = " << i << std::endl;
    delete navigator;
    return EXIT_FAILURE;
  }
  auto real_time = timer.Stop();

  delete navigator;
  std::cout << "\nNavigation test for BVHNavigator passed! seed = " << seed << " checksum = " << std::setprecision(15)
            << checksum << " real_time = " << std::setprecision(6) << 1.e6 * real_time / iterations
            << " [us] per iteration\n";
  return EXIT_SUCCESS;
}
