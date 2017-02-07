#include "utilities/ShapeDebugger.h"
#include "volumes/Sphere.h"

using namespace vecgeom;

int main() {
  SimpleSphere sphere("Debugger Sphere", 0 , 5 , 0.0 , 2*kPi , kPi/6 , kPi/6);
  ShapeDebugger debugger(&sphere);
  debugger.ShowCorrectResults(false);
  //debugger.CompareDistanceToInToROOT(Vector3D<Precision>(4, 4, 4));
  debugger.CompareDistanceToOutToROOT(Vector3D<Precision>(20, 20, 20));
  return 0;
}