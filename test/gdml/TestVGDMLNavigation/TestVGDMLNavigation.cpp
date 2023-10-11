#include "Frontend.h" // VecGeom/gdml/Frontend.h

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/BVHNavigatorV.h"
#include "VecGeom/navigation/HybridNavigator2.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"

#include <algorithm>
#include <sstream>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <err.h>
#include <getopt.h>
#include <libgen.h>

using namespace vecgeom;

static std::random_device rd;
static std::default_random_engine rng;
static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

double uniform(double a, double b)
{
  return a + (b - a) * dist(rng);
}

Vector3D<Precision> random_unit_vector()
{
  Precision z = uniform(-1.0f, 1.0f);
  Precision r = sqrt(1.0f - z * z);
  Precision t = uniform(0.0f, 6.2831853f);
  return {r * vecCore::math::Cos<Precision>(t), r * vecCore::math::Sin<Precision>(t), z};
}

bool nearly_equal(double x, double y)
{
  using std::abs;

  if (x == y)
    return true;
  else if (x * y == 0.0)
    return abs(x - y) < DBL_EPSILON * DBL_EPSILON;
  else
    return abs(x - y) < (abs(x) + abs(y)) * DBL_EPSILON;
}

VNavigator const *get_navigator(const char *name)
{
  static VNavigator const *navigators[] = {
      NewSimpleNavigator<>::Instance(),
      SimpleABBoxNavigator<>::Instance(),
      HybridNavigator<>::Instance(),
      BVHNavigatorV<>::Instance(),
  };

  for (auto navigator : navigators)
    if (strcmp(name, navigator->GetName()) == 0) return navigator;

  return nullptr;
}

bool navigate(Vector3D<Precision> p, Vector3D<Precision> dir, bool verbose = true)
{
  auto &geoManager      = GeoManager::Instance();
  NavigationState *curr = NavigationState::MakeInstance(geoManager.getMaxDepth());
  NavigationState *next = NavigationState::MakeInstance(geoManager.getMaxDepth());

  VNavigator const &ref_navigator = *NewSimpleNavigator<>::Instance();

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

    // Set the last exited volume to be the current one, to avoid wrongly skipping a valid volume entrance
    curr->SetLastExited();
    Precision ref_step = ref_navigator.ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);
    Precision step     = curr_volume->GetNavigator()->ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);

    if (verbose)
      printf("%6zu [ % 14.8f, % 14.8f, % 14.8f ] % 14.8f % 14.8f %s\n", ++steps, p.x(), p.y(), p.z(), step, ref_step,
             curr_volume->GetLabel().c_str());

    if (!nearly_equal(step, ref_step)) {
      if (verbose)
        printf("FAILED: step = %14.8f ref_step = %14.8f  step / ref_step - 1 = %14.8f\n", step, ref_step,
               step / ref_step - 1);
      // Replay last distance queries (for debugger sessions)
      ref_navigator.ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);
      curr_volume->GetNavigator()->ComputeStepAndPropagatedState(p, dir, kInfLength, *curr, *next);
      return false;
    }
    step = vecCore::math::Max(step, kTolerance);

    p = p + step * dir;

    std::swap(curr, next);
  }

  if (verbose) printf("\n");

  return true;
}

int main(int argc, char **argv)
{
  bool verbose                = false;
  bool validate               = false;
  double mm_unit              = 0.1;
  unsigned long seed          = 0;
  unsigned long iterations    = 1;
  VNavigator const *navigator = SimpleABBoxNavigator<>::Instance();

  for (;;) {
    int opt = getopt(argc, argv, "hi:n:s:v");

    if (opt == -1) break;

    switch (opt) {
    case 'i':
      errno      = 0;
      iterations = strtoul(optarg, nullptr, 10);
      if (errno) errx(errno, "%s: %lu", strerror(errno), iterations);
      break;

    case 'n':
      navigator = get_navigator(optarg);
      if (!navigator) errx(EINVAL, "Invalid navigator: %s", optarg);
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

  for (auto &item : geoManager.GetLogicalVolumesMap()) {
    auto &volume   = *item.second;
    auto nchildren = volume.GetDaughters().size();
    volume.SetNavigator(nchildren > 0 ? navigator : NewSimpleNavigator<>::Instance());

    HybridManager2::Instance().InitStructure(item.second);
  }

  BVHManager::Init();

  auto getSeed = [](std::default_random_engine &rng) {
    std::stringstream ss;
    ss << rng;
    return std::stoul(ss.str());
  };

  rng.seed(seed ? seed : seed = rd());

  for (unsigned long i = 0; i < iterations; ++i) {
    // backup last used seed
    seed = getSeed(rng);
    Vector3D<Precision> p(0.0, 0.0, 0.0);
    Vector3D<Precision> dir = random_unit_vector();

    if (navigate(p, dir, verbose)) continue;

    navigate(p, dir);
    printf("\nNavigation test for %s failed! seed = %lu, iteration = %lu\n", navigator->GetName(), seed, i);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
