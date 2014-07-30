#include "base/Vector3D.h"
#include "base/Stopwatch.h"
#include "volumes/Planes.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

int main() {
  constexpr int multiplier = 1;
  Planes planes(6*multiplier);
  for (int i = 0; i < multiplier; ++i) {
    planes.Set(6*i, Vector3D<Precision>(0,0,1), Vector3D<Precision>(0,0,1));
    planes.Set(6*i+1, Vector3D<Precision>(0,0,-1), Vector3D<Precision>(0,0,-1));
    planes.Set(6*i+2, Vector3D<Precision>(1,0,0), Vector3D<Precision>(1,0,0));
    planes.Set(6*i+3, Vector3D<Precision>(-1,0,0), Vector3D<Precision>(-1,0,0));
    planes.Set(6*i+4, Vector3D<Precision>(0,1,0), Vector3D<Precision>(0,1,0));
    planes.Set(6*i+5, Vector3D<Precision>(0,-1,0), Vector3D<Precision>(0,-1,0));
  }
  std::cout << planes;
  Stopwatch timer;
  Vector3D<Precision> const sampleBounds(2., 2., 2.);
  constexpr int iMax = 1<<22;
  std::cout << "Running " << iMax << " iterations...\n";
  Inside_t output[6*multiplier];
  timer.Start();
  for (int i = 0; i < iMax; ++i) {
    planes.Inside(volumeUtilities::SamplePoint(sampleBounds), output);
  }
  std::cout << "Elapsed time: " << timer.Stop() << "s.\n";
  return 0;
}