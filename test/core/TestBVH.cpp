#include "VecGeom/base/BVH.h"
#include "VecGeom/base/BVH_V2.h"
#include "VecGeom/base/PriorityQueue.h"
#include "VecGeom/management/ABBoxManager.h"
#include <memory>
#undef NDEBUG
#include "VecGeom/base/Assert.h"
#include <iostream>

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

BVH<float> const *CreateBVHStructure(Boxes_t boxes, size_t N) { return new BVH<float>(0, boxes, N); }
BVH_V2<float> const *CreateBVH2Structure(Boxes_t boxes, size_t N) { return new BVH_V2<float>(0, boxes, N); }

void QueryStructure(BVH<float> const &bvh, BVH_V2<float> const &bvh2)
{
  // testing the looper + correct intersection by calculating
  // the sum of all box ids intersected
  int checkhitsum     = 0;
  int checkhitsum2    = 0;
  float min_distance  = 1.E20;
  float min_distance2 = 1.E20;
  auto userhook       = [&](BVHIntersectContext<float> &ctx) {
    checkhitsum += (ctx.primID + 1); // +1 just to detect also hit with the first box (whose id is 0)
    if (ctx.tnear < min_distance) {
      min_distance = ctx.tnear;
    }
    return false; // do not stop after here
  };
  auto userhook2 = [&](BVHIntersectContext<float> &ctx) {
    checkhitsum2 += (ctx.primID + 1);
    if (ctx.tnear < min_distance2) {
      min_distance2 = ctx.tnear;
    }
    return false;
  };

  const auto nChildren = bvh.GetRootNChild();

  int c = 0;
  for (int i = 0; i < (int)nChildren; ++i) {
    c += (i + 1);
  }

  {
    // for a ray passing all boxes left to right
    Vector3D<Precision> pos(-1000, 0, 0);
    Vector3D<Precision> dir(1., 0., 0);

    checkhitsum = 0;
    // intersect ray with the BVH structure and use hook
    bvh.Intersect(pos, dir, 1E20, userhook);
    bvh2.Intersect<false>(pos, dir, 1E20, userhook2);
    VECGEOM_ASSERT(c == checkhitsum); // checks that all boxes have been hit
    VECGEOM_ASSERT(c == checkhitsum2);
  }

  { // for a ray passing all boxes right to left
    // define the ray
    Vector3D<Precision> pos(10000, 0, 0);
    Vector3D<Precision> dir(-1., 0., 0);

    checkhitsum  = 0;
    checkhitsum2 = 0;
    min_distance = 1.E20;
    // intersect ray with the BVH structure and use hook
    bvh.Intersect(pos, dir, 1E20, userhook);
    bvh2.Intersect<false>(pos, dir, 1E20, userhook2);

    VECGEOM_ASSERT(c == checkhitsum); // checks that all boxes have been hit
    VECGEOM_ASSERT(c == checkhitsum2);
  }

  {
    // for a ray passing none of the boxes
    // define the ray
    Vector3D<Precision> pos(1000., 1000., 0.);
    Vector3D<Precision> dir(-1., 0., 0);

    float check   = 1.E20;
    checkhitsum   = 0;
    min_distance  = check;
    min_distance2 = check;
    checkhitsum2  = 0;
    // intersect ray with the BVH structure and use hook
    bvh.Intersect(pos, dir, 1E20, userhook);
    bvh2.Intersect<false>(pos, dir, 1E20, userhook2);

    VECGEOM_ASSERT(0 == checkhitsum);
    VECGEOM_ASSERT(std::abs(check - min_distance) < 1.E-4);

    VECGEOM_ASSERT(0 == checkhitsum2);
    VECGEOM_ASSERT(std::abs(check - min_distance2) < 1E-4);
  }

  {
    // for a ray passing exactly one of the boxes
    // define the ray
    Vector3D<Precision> pos(0., -100., 0.);
    Vector3D<Precision> dir(0., 1., 0.);

    checkhitsum          = 0;
    min_distance         = 1.E20;
    float check_distance = 100.f - 4.f; // 4 is the half length of a box
    // intersect ray with the BVH structure and use hook
    bvh.Intersect<false>(pos, dir, 1E20, userhook);
    VECGEOM_ASSERT(1 == checkhitsum); // should hit exactly the first box (index 0 + 1)
    VECGEOM_ASSERT(std::abs(check_distance - min_distance) < 1E-2);

    checkhitsum2  = 0;
    min_distance2 = 1.E20;
    bvh2.Intersect<false>(pos, dir, 1E20, userhook2);
    VECGEOM_ASSERT(1 == checkhitsum2); // should hit exactly the first box (index 0 + 1)
    VECGEOM_ASSERT(std::abs(check_distance - min_distance2) < 1E-2);
  }

  {
    // special test to check early pruning
    // after the first hit, we modify the step_max to 0 ... which should prune away
    // tree traversal

    // for a ray passing all boxes right to left
    // define the ray
    Vector3D<Precision> pos(10000, 0, 0);
    Vector3D<Precision> dir(-1., 0., 0);

    auto pruning_userhook = [&](BVHIntersectContext<float> &ctx) {
      checkhitsum += (ctx.primID + 1); // +1 just to detect also hit with the first box (whose id is 0)
      ctx.step_max = 0.;               // enable pruning after this hit
      return false;                    // do not stop after here
    };
    auto pruning_userhook2 = [&](BVHIntersectContext<float> &ctx) {
      checkhitsum2 += (ctx.primID + 1);
      ctx.step_max = 0.;
      return false;
    };

    checkhitsum  = 0;
    checkhitsum2 = 0;
    bvh2.Intersect(pos, dir, 1E20, pruning_userhook2);
    bvh.Intersect<false>(pos, dir, 1E20, pruning_userhook);
    VECGEOM_ASSERT(checkhitsum <= c); // should hit less leafes
    VECGEOM_ASSERT(checkhitsum2 <= c);
  }

  {
    // special step_max test

    // for a ray passing all boxes right to left
    // define the ray
    Vector3D<Precision> pos(10000, 0, 0);
    Vector3D<Precision> dir(-1., 0., 0);

    auto pruning_userhook = [&](BVHIntersectContext<float> &ctx) {
      checkhitsum += (ctx.primID + 1); // +1 just to detect also hit with the first box (whose id is 0)
      ctx.step_max = 0.;               // enable pruning after this hit
      return false;                    // do not stop after here
    };
    auto pruning_userhook2 = [&](BVHIntersectContext<float> &ctx) {
      checkhitsum2 += (ctx.primID + 1); // +1 just to detect also hit with the first box (whose id is 0)
      ctx.step_max = 0.;                // enable pruning after this hit
      return false;                     // do not stop after here
    };

    checkhitsum  = 0;
    checkhitsum2 = 0;
    bvh.Intersect(pos, dir, 1., pruning_userhook);
    bvh2.Intersect<false>(pos, dir, 1., pruning_userhook2);
    VECGEOM_ASSERT(checkhitsum == 0); // should not have hit anything because step_max bound
    VECGEOM_ASSERT(checkhitsum2 == 0);
  }
}

void TestPriorityQueue()
{
  vecgeom::PriorityQueue<int, float> pq;
  pq.push(20, 0.5f);
  pq.push(30, 0.7f);
  pq.push(40, 0.4f);
  VECGEOM_ASSERT(pq.empty() == false);
  VECGEOM_ASSERT(pq.peek_priority() == 0.4f);
  VECGEOM_ASSERT(pq.peek() == 40);
  auto i = pq.pop();
  VECGEOM_ASSERT(i == 40);
  VECGEOM_ASSERT(pq.peek_priority() == 0.5f);
  VECGEOM_ASSERT(pq.peek() == 20);

  i = pq.pop();
  VECGEOM_ASSERT(i == 20);
  VECGEOM_ASSERT(pq.peek_priority() == 0.7f);
  VECGEOM_ASSERT(pq.peek() == 30);

  pq.pop();
  VECGEOM_ASSERT(pq.empty() == true);
}

int main()
{
  // test for various numbers of aligned boxes
  for (int i = 4; i < 1000; i += 6) {
    auto structure  = CreateBVHStructure(MakeABBoxes_linear(i), i);
    auto structure2 = CreateBVH2Structure(MakeABBoxes_linear(i), i);
    QueryStructure(*structure, *structure2);
    delete structure;
  }

  TestPriorityQueue();

  std::cout << "test passed\n";
  return 0;
}
