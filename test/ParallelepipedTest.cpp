#include "volumes/Parallelepiped.h"
#include "volumes/Paraboloid.h"
#include "volumes/Trapezoid.h"
#include "volumes/kernel/ParallelepipedImplementation.h"

#include <stdio.h>

using namespace vecgeom;

int main() {
  const Vector3D<Precision> point(3, -3, 0.1);
  const Vector3D<Precision> direction(0.3, 5.9, -1.01);
  const Transformation3D trans(1, -2, 3, 10.03, 13.77, 67.8);
  const UnplacedParallelepiped unplaced = UnplacedParallelepiped(
      8, -0.8, 1.111, 70.3, 3.88, 39.17);

  Vector3D<Precision> pointTransFirst = trans.Transform(point);
  Vector3D<Precision> dirTransFirst = trans.TransformDirection(direction);

  Vector3D<Precision> pointSkewed = point;
  Vector3D<Precision> dirSkewed = direction;
  ParallelepipedImplementation<-1, -1>::Transform<kScalar>(unplaced,
                                                           pointSkewed);
  ParallelepipedImplementation<-1, -1>::Transform<kScalar>(unplaced,
                                                           dirSkewed);
  ParallelepipedImplementation<-1, -1>::Transform<kScalar>(unplaced,
                                                           pointTransFirst);
  ParallelepipedImplementation<-1, -1>::Transform<kScalar>(unplaced,
                                                           dirTransFirst);

  Vector3D<Precision> pointSkewedFirst = trans.Transform(pointSkewed);
  Vector3D<Precision> dirSkewedFirst = trans.TransformDirection(dirSkewed);

  printf("(%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)\n"
         "(%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)\n",
         pointTransFirst[0], pointTransFirst[1], pointTransFirst[2],
         dirTransFirst[0], dirTransFirst[1], dirTransFirst[2],
         pointSkewedFirst[0], pointSkewedFirst[1], pointSkewedFirst[2],
         dirSkewedFirst[0], dirSkewedFirst[1], dirSkewedFirst[2]);

  return 0;
}