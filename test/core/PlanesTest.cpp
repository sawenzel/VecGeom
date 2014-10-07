#include "base/Vector3D.h"
#include "base/Stopwatch.h"
#include "volumes/Box.h"
#include "volumes/Planes.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

constexpr int kIterations = 1<<22;

int StaticPlanes() {
  Planes planes(6);
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
        Precision distancePlanes = planes.Distance<kScalar>(point, direction);
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
  return mismatches > 0;
}

int main() {
  bool error = false;
  error |= StaticPlanes();
  return error;
}