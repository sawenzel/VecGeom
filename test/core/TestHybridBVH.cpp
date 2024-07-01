#include "VecGeom/management/HybridManager2.h"
#include "VecGeom/navigation/HybridNavigator2.h"
#include "VecGeom/management/ABBoxManager.h"
#include <memory>
#undef NDEBUG
#include <cassert>

using namespace vecgeom;

using Boxes_t     = ABBoxManager<Precision>::ABBoxContainer_t;
using BoxCorner_t = ABBoxManager<Precision>::ABBox_s;

// make a vector of aligned bounding boxes
// boxes are just arranged linearly
Boxes_t MakeABBoxes_linear(int N)
{
  // N boxes ... so 2*N corners
  double boxhalflength{4}; // half length of an aligned box
  BoxCorner_t *boxcorners = new BoxCorner_t[2 * N];
  for (int i = 0; i < N; ++i) {
    double xoffset            = i * (2.5 * boxhalflength);
    boxcorners[2 * i].x()     = xoffset - boxhalflength;
    boxcorners[2 * i].y()     = -boxhalflength;
    boxcorners[2 * i].z()     = -boxhalflength;
    boxcorners[2 * i + 1].x() = xoffset + boxhalflength;
    boxcorners[2 * i + 1].y() = boxhalflength;
    boxcorners[2 * i + 1].z() = boxhalflength;
  }
  return &boxcorners[0];
}

using BVHStructure = HybridManager2::HybridBoxAccelerationStructure;
BVHStructure const *CreateHybridStructure(Boxes_t boxes, size_t N)
{
  return HybridManager2::Instance().BuildStructure(boxes, N);
}

void QueryStructure(BVHStructure const &s)
{
  // testing the looper + correct intersection by calculating
  // the sum of all box ids intersected
  int checkhitsum = 0;
  auto userhook   = [&](HybridManager2::BoxIdDistancePair_t hitbox) {
    checkhitsum += (hitbox.first + 1); // +1 just to detect also hit with the first box (whose id is 0)
    return false;                      // we are never done early here
  };

  int c = 0;
  for (int i = 0; i < (int)s.fNumberOfOriginalBoxes; ++i) {
    c += (i + 1);
  }

  HybridNavigator<> *instance = (HybridNavigator<> *)HybridNavigator<>::Instance();

  {
    // for a ray passing all boxes left to right
    Vector3D<Precision> pos(-1000, 0, 0);
    Vector3D<Precision> dir(1., 0., 0);

    checkhitsum = 0;
    // intersect ray with the BVH structure and use hook
    instance->BVHSortedIntersectionsLooper(s, pos, dir, 1E20, userhook);
    assert(c == checkhitsum); // checks that all boxes have been hit
  }

  { // for a ray passing all boxes right to left
    // define the ray
    Vector3D<Precision> pos(10000, 0, 0);
    Vector3D<Precision> dir(-1., 0., 0);

    checkhitsum = 0;
    // intersect ray with the BVH structure and use hook
    instance->BVHSortedIntersectionsLooper(s, pos, dir, 1E20, userhook);
    assert(c == checkhitsum); // checks that all boxes have been hit
  }

  {
    // for a ray passing none of the boxes
    // define the ray
    Vector3D<Precision> pos(1000., 1000., 0.);
    Vector3D<Precision> dir(-1., 0., 0);

    checkhitsum = 0;
    // intersect ray with the BVH structure and use hook
    instance->BVHSortedIntersectionsLooper(s, pos, dir, 1E20, userhook);
    assert(0 == checkhitsum);
  }

  {
    // for a ray passing exactly one of the boxes
    // define the ray
    Vector3D<Precision> pos(0., -100., 0.);
    Vector3D<Precision> dir(0., 1., 0.);

    // intersect ray with the BVH structure and use hook
    instance->BVHSortedIntersectionsLooper(s, pos, dir, 1E20, userhook);
    assert(1 == checkhitsum); // should hit exactly the first box (index 0 + 1)
  }
}

int main()
{
  // test for various numbers of aligned boxes
  for (int i = 4; i < 100; i += 6) {
    auto structure = CreateHybridStructure(MakeABBoxes_linear(i), i);
    QueryStructure(*structure);
  }
  std::cout << "test passed\n";
  return 0;
}
