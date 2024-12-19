#include "VecGeom/volumes/GenericPolycone.h"

int main(int argc, char** argv) {
  double r[5] {0, 2, 20, 30.149999999999999, 0};
  double z[5] {2, 0, 0, 5.5199999999999998, 48.399999999999999};
  auto poly = new vecgeom::UnplacedGenericPolycone(0, 2*M_PI, 5, r, z);
  return 0;
}
