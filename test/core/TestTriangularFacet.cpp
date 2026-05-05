#include "VecGeom/volumes/Tile.h"
#undef NDEBUG
#include "VecGeom/base/Assert.h"
#include "test/unit_tests/ApproxEqual.h"
#include <iostream>

using namespace vecgeom;

void test_suite()
{
  using TT    = TriangularTile<double>;
  using Vec3  = Vector3D<double>;
  using Vec3f = Vector3D<float>;

  // Triangle 1: right triangle in z = 0 plane
  TT tri1(Vec3{0.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0});

  // Triangle 2: shifted triangle in z = 1 plane
  TT tri2(Vec3{1.0, 1.0, 1.0}, Vec3{3.0, 1.0, 1.0}, Vec3{1.0, 3.0, 1.0});

  // check ray-triangle intersection ----------
  {
    // Hits inside tri1 at (0.25, 0.25, 0)
    Vec3 origin{0.25, 0.25, 1.0};
    Vec3 dir{0.0, 0.0, -1.0};

    double t = tri1.Distance(origin, dir);
    VECGEOM_ASSERT(ApproxEqual<Precision>(t, 1.0));
  }

  {
    // Misses tri1
    Vec3 origin{1.2, 1.2, 1.0};
    Vec3 dir{0.0, 0.0, -1.0};

    double t = tri1.Distance(origin, dir);
    VECGEOM_ASSERT(ApproxEqual<double>(t, std::numeric_limits<double>::infinity()));
  }

  {
    // Hits tri2 at (1.5, 1.5, 1)
    Vec3 origin{1.5, 1.5, 3.0};
    Vec3 dir{0.0, 0.0, -1.0};

    double t = tri2.Distance(origin, dir);
    VECGEOM_ASSERT(ApproxEqual<double>(t, 2.0));
  }

  {
    // Parallel ray to tri2 plane -> no hit
    Vec3 origin{1.5, 1.5, 2.0};
    Vec3 dir{1.0, 0.0, 0.0};

    double t = tri2.Distance(origin, dir);
    VECGEOM_ASSERT(ApproxEqual<double>(t, std::numeric_limits<double>::infinity()));
  }

  // ---------- Point-to-triangle min squared distance ----------
  {
    // Point directly above interior of tri1
    Vec3 p{0.25, 0.25, 2.0};
    float d2 = tri1.SafetySq<float>(p);
    VECGEOM_ASSERT(ApproxEqual<float>(d2, 4.0f)); // vertical distance = 2
  }

  {
    // Point on tri1
    Vec3f p{0.2, 0.2, 0.0};
    float d2 = tri1.SafetySq(p);
    VECGEOM_ASSERT(ApproxEqual<float>(d2, 0.0));
  }

  {
    // Closest point is vertex (0,0,0) of tri1
    Vec3 p{-1.0, -1.0, 0.0};
    double d2 = tri1.SafetySq(p);
    VECGEOM_ASSERT(ApproxEqual<float>(d2, 2.0)); // 1^2 + 1^2
  }

  {
    // Closest point is on edge from (1,1,1)? no, tri2 lies in z=1 plane.
    // For p = (2,2,1), closest point is on hypotenuse x+y=4? actually inside? no, outside.
    // Projection onto edge from (3,1,1) to (1,3,1) is exactly (2,2,1), so distance is 0.
    Vec3 p{2.0, 2.0, 1.0};
    double d2 = tri2.SafetySq(p);
    VECGEOM_ASSERT(ApproxEqual<double>(d2, 0.0)); // point lies on tri2 edge
  }

  {
    // Point above tri2 interior
    Vec3 p{1.25, 1.25, 4.0};
    double d2 = tri2.SafetySq(p);
    VECGEOM_ASSERT(ApproxEqual<double>(d2, 9.0)); // z distance = 3
  }

  {
    // Closest point is tri2 vertex (1,1,1)
    Vec3 p{0.0, 0.0, 1.0};
    double d2 = tri2.SafetySq(p);
    VECGEOM_ASSERT(ApproxEqual(d2, 2.0));
  }

  {
    TT tri(Vec3{20, -30, 40}, Vec3{20, 30, -40}, Vec3{20, 30, 40});

    Vec3 p{20, 0, 0};
    Vec3 dir{1, 0, 0};
    double s = tri.SafetySq(p);
    VECGEOM_ASSERT(ApproxEqual(s, 0.0));
    double d = tri.Distance(p, dir, -kHalfTolerance);
    VECGEOM_ASSERT(ApproxEqual(d, 0.0)); // we are on the surface / edge : should be zero
  }
}

int main()
{
  test_suite();
  return 0;
}
