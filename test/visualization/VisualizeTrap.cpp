#include "utilities/ShapeDebugger.h"
#include "volumes/Trapezoid.h"

using namespace vecgeom;

int main() {

  SimpleTrapezoid trap("Trapezoid Visualizer", 2, 0, 0, 4, 3., 5., 0, 6, 4.5, 7.5, 0);
  ShapeDebugger debugger(&trap);

  debugger.ShowCorrectResults(true);
  debugger.CompareDistanceToOutToROOT( Vector3D<Precision>(10,8,4) );

  return 0;
}
