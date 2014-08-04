#include "base/Vector3D.h"
#include "base/Stopwatch.h"
#include "volumes/Box.h"
#include "volumes/Planes.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

constexpr int kIterations = 1<<22;

int DynamicPlanes() {
  constexpr int multiplier = 1;
  Planes<0> planes(6*multiplier);
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
  const Vector3D<Precision> sampleBounds(2., 2., 2.);
  Inside_t output[6*multiplier];
  std::cout << "Running " << kIterations << " iterations...\n";
  timer.Start();
  for (int i = 0; i < kIterations; ++i) {
    planes.Inside(volumeUtilities::SamplePoint(sampleBounds), output);
  }
  std::cout << "Elapsed time: " << timer.Stop() << "s.\n";
  return 0;
}

int StaticPlanes() {
  Planes<6> planes;
  planes.Set(0, Vector3D<Precision>(0,0,1), Vector3D<Precision>(0,0,1));
  planes.Set(1, Vector3D<Precision>(0,0,-1), Vector3D<Precision>(0,0,-1));
  planes.Set(2, Vector3D<Precision>(1,0,0), Vector3D<Precision>(1,0,0));
  planes.Set(3, Vector3D<Precision>(-1,0,0), Vector3D<Precision>(-1,0,0));
  planes.Set(4, Vector3D<Precision>(0,1,0), Vector3D<Precision>(0,1,0));
  planes.Set(5, Vector3D<Precision>(0,-1,0), Vector3D<Precision>(0,-1,0));
  SimpleBox die("die", 1, 1, 1);
  std::cout << planes;
  const Vector3D<Precision> sampleBounds(2., 2., 2.);
  constexpr int nPoints = 1024;
  int mismatches = 0;
  for (int i = 0; i < nPoints; ++i) {
    Vector3D<Precision> point = volumeUtilities::SamplePoint(sampleBounds);
    Vector3D<Precision> direction = volumeUtilities::SampleDirection();
    Inside_t insidePlanes = planes.Inside<kScalar>(point);
    Inside_t insideDie = die.Inside(point);
    if (insidePlanes != insideDie) {
      ++mismatches;
      std::cout << "Inside mismatch for " << point << ": " << insidePlanes
                << " / " << insideDie << "\n";
    } else {
      if (insidePlanes == EInside::kInside) {
        Precision distancePlanes =
            planes.DistanceToOut<kScalar>(point, direction);
        Precision distanceDie = die.DistanceToOut(point, direction);
        if (Abs(distancePlanes - distanceDie) > kTolerance) {
          ++mismatches;
          std::cout << "DistanceToOut mismatch for " << point << "--"
                    << direction << ": " << distancePlanes << " / "
                    << distanceDie << "\n";
        }
      }
    }
  }
  std::cout << "Mismatches: " << mismatches << " / " << nPoints << "\n";
  return 0;
}

int main() {
  bool success = true;
  success &= !StaticPlanes();
  // success &= !DynamicPlanes();
  return success;
}