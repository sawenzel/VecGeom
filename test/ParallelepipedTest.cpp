#include "volumes/Parallelepiped.h"

using namespace vecgeom;

int main() {
  UnplacedParallelepiped unplaced = UnplacedParallelepiped(5, 5, 5, 5, 5, 5);
  LogicalVolume logical = LogicalVolume(&unplaced);
  return 0;
}