#include "base/Vector3D.h"
#include "base/Stopwatch.h"
#include "volumes/Planes.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

int main() {
  Planes planes(6);
  planes.Set(0, Vector3D<Precision>(0, 0, 1), Vector3D<Precision>(0, 0, 1));
  planes.Set(1, Vector3D<Precision>(0, 0, -1), Vector3D<Precision>(0, 0, -1));
  planes.Set(2, Vector3D<Precision>(1, 0, 0), Vector3D<Precision>(1, 0, 0));
  planes.Set(3, Vector3D<Precision>(-1, 0, 0), Vector3D<Precision>(-1, 0, 0));
  planes.Set(4, Vector3D<Precision>(0, 1, 0), Vector3D<Precision>(0, 1, 0));
  planes.Set(5, Vector3D<Precision>(0, -1, 0), Vector3D<Precision>(0, -1, 0));
  std::cout << planes;
  assert(planes.Inside(Vector3D<Precision>(-0.1, 0., 0.6)) == EInside::kInside);
  assert(planes.Inside(Vector3D<Precision>(0.1, 0., -0.6)) == EInside::kInside);
  assert(planes.Inside(Vector3D<Precision>(-1, 0.1, 0.3)) == EInside::kSurface);
  assert(planes.Inside(Vector3D<Precision>(-1, 3, 0)) == EInside::kOutside);
  assert(planes.Inside(Vector3D<Precision>(-1.1, 0.3, 0)) == EInside::kOutside);
  Stopwatch timer;
  Vector3D<Precision> const sampleBounds(2., 2., 2.);
  constexpr int iMax = 1<<20;
  std::cout << "Running " << iMax << " iterations...\n";
  int inside = 0, outside = 0, surface = 0;
  timer.Start();
  for (int i = 0; i < iMax; ++i) {
    Inside_t result = planes.Inside(volumeUtilities::SamplePoint(sampleBounds));
    if (result == EInside::kInside) {
      ++inside;
    } else if (result == EInside::kOutside) {
      ++outside;
    } else {
      ++surface;
    }
  }
  std::cout << "Elapsed time: " << timer.Stop() << "s.\n";
  std::cout << "Inside: " << inside << ", Outside: " << outside
            << " Surface: " << surface << "\n";
  return 0;
}