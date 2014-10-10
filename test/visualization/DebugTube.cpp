#include "utilities/ShapeDebugger.h"
#include "volumes/Tube.h"

using namespace vecgeom;

int main() {
  SimpleTube tube("Debugger Tube", 1, 4, 4, kPi/4, 5/4*kPi);
  ShapeDebugger debugger(&tube);
  debugger.CompareDistanceToOutROOT(Vector3D<Precision>(4, 4, 4));
  return 0;
}