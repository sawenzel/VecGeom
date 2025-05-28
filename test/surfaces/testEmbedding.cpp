#include <VecGeom/surfaces/mask/FrameMasks.h>

using namespace vgbrep;
using namespace vecgeom;

template <typename Real_t, typename FrameType1, typename FrameType2>
void CheckFrames(FrameType1 const &f1, FrameType2 const &f2, TransformationMP<Real_t> const &trans,
                 FrameIntersect expected, FrameIntersect expected_inv)
{
  auto result = FrameChecker<Real_t, FrameType1, FrameType2>::CheckFrames(f1, f2, trans);
  VECGEOM_ASSERT(result == expected);
  // Check that the opposite stands
  result = FrameChecker<Real_t, FrameType2, FrameType1>::CheckFrames(f2, f1, trans.Inverse());
  VECGEOM_ASSERT(result == expected_inv);
}

int main(int argc, char *argv[])
{
  // create some frames
  using Real_t = vecgeom::Precision;

  vecgeom::Transformation3DMP<Real_t> identity;

  // Check AngleInterval
  AngleInterval<Real_t> ang1(0., vecgeom::kPi / 4);                  // [0, 45]
  AngleInterval<Real_t> ang2(vecgeom::kPi / 4, vecgeom::kPi / 2);    // [45, 90]
  AngleInterval<Real_t> ang3(vecgeom::kPi / 12, kPi);                // [15, 180]
  AngleInterval<Real_t> ang4(-vecgeom::kPi / 12, -vecgeom::kPi / 4); // [-15, -45] -> [345, 675]
  AngleInterval<Real_t> ang5(-vecgeom::kPi / 4, vecgeom::kPi / 12);  // [-45, 15] -> [315, 375]

  VECGEOM_ASSERT(ang1 == ang1);
  VECGEOM_ASSERT(ang4 > ang3);
  VECGEOM_ASSERT(ang4 != ang5);
  VECGEOM_ASSERT(ang1.Intersect(ang2).IsNull());
  VECGEOM_ASSERT(ang1.Intersect(ang3) == AngleInterval<Real_t>(vecgeom::kPi / 12, vecgeom::kPi / 4));
  VECGEOM_ASSERT(ang4.Intersect(ang5) == AngleInterval<Real_t>(-vecgeom::kPi / 12, vecgeom::kPi / 12));
  VECGEOM_ASSERT(ang4.Intersect(ang5) > ang3);

  // Validate segment intersection
  auto CheckIntersect = [](Segment2D<Real_t> const &seg1, Segment2D<Real_t> const &seg2, SegmentIntersect expected1,
                           SegmentIntersect expected2) {
    VECGEOM_ASSERT(seg1.Intersect(seg2) == expected1);
    VECGEOM_ASSERT(seg2.Intersect(seg1) == expected2);
  };

  Segment2D<Real_t> seg1({0., 0.}, {10., 0.});
  Segment2D<Real_t> seg2({0., -0.5 * vecgeom::kTolerance}, {0., 10.});
  Segment2D<Real_t> seg3({vecgeom::kTolerance, -vecgeom::kTolerance}, {1., 1.});
  Segment2D<Real_t> seg4({5., -0.5 * vecgeom::kTolerance}, {5., 1.});
  Segment2D<Real_t> seg5({5., -2 * vecgeom::kTolerance}, {5., 1.});
  Segment2D<Real_t> seg6({5., 2 * vecgeom::kTolerance}, {5., 1.});
  Segment2D<Real_t> seg7({0., 0.5 * vecgeom::kTolerance}, {10., 0.5 * vecgeom::kTolerance});
  Segment2D<Real_t> seg8({0.5 * vecgeom::kTolerance, 0.5 * vecgeom::kTolerance},
                         {10. - 10 * vecgeom::kTolerance, vecgeom::kTolerance});
  Segment2D<Real_t> seg9({2., 0.}, {8., 0.});
  Segment2D<Real_t> seg10({-2., 0.}, {8., 0.});
  // Distance to intersection point < tolerance to both segment edges -> no intersect
  CheckIntersect(seg1, seg2, SegmentIntersect::kNoIntersect, SegmentIntersect::kNoIntersect);
  // Distance to intersection to seg1=2*tol, seg2=sqrt(2)*tol -> intersect
  CheckIntersect(seg1, seg3, SegmentIntersect::kIntersect, SegmentIntersect::kIntersect);
  // Distance to intersection to seg1=5, seg2=-0.5*tol -> no intersect
  CheckIntersect(seg1, seg4, SegmentIntersect::kNoIntersect, SegmentIntersect::kNoIntersect);
  // Distance to intersection to seg1=5, seg2=2*tol -> intersect
  CheckIntersect(seg1, seg5, SegmentIntersect::kIntersect, SegmentIntersect::kIntersect);
  // Bounding box disjoint
  CheckIntersect(seg1, seg6, SegmentIntersect::kNoIntersect, SegmentIntersect::kNoIntersect);
  // Equal segments
  CheckIntersect(seg1, seg1, SegmentIntersect::kEqual, SegmentIntersect::kEqual);
  // Equal segments matching ends within tolerance
  CheckIntersect(seg1, seg7, SegmentIntersect::kEqual, SegmentIntersect::kEqual);
  // Embedded/embedding segment
  CheckIntersect(seg1, seg8, SegmentIntersect::kEmbedding, SegmentIntersect::kEmbedded);
  CheckIntersect(seg1, seg9, SegmentIntersect::kEmbedding, SegmentIntersect::kEmbedded);
  // Overlapping segments
  CheckIntersect(seg1, seg10, SegmentIntersect::kOverlap, SegmentIntersect::kOverlap);

  WindowMask<Real_t> w1(20., 30.);
  WindowMask<Real_t> w2(10., 20.);
  WindowMask<Real_t> w3(1., 1.);
  WindowMask<Real_t> w4(1., 15.);
  WindowMask<Real_t> w5(12., 12.);
  WindowMask<Real_t> w6(10., 10.);
  WindowMask<Real_t> w7(-10., 0., -5., 5.);
  WindowMask<Real_t> w8(-10., 0.0001, -5., 5.);

  RingMask<Real_t> r0(2., 5., true, 0., kTwoPi);                           // full ring
  RingMask<Real_t> r00(2., 10., true, 0., kTwoPi);                         // full ring
  RingMask<Real_t> r1(10., 20., true, 0., kTwoPi);                         // full ring
  RingMask<Real_t> r2(0., 20., true, 0., kTwoPi);                          // full disk
  RingMask<Real_t> r3(0., 20., false, 0., kHalfPi);                        // convex pie
  RingMask<Real_t> r4(0., 20., false, 0., 1.5 * kPi);                      // concave pie
  RingMask<Real_t> r5(10., 20., false, 0., 1.5 * kPi);                     // concave cut ring
  RingMask<Real_t> r6(0., 1., true, 0., kTwoPi);                           // smaller full disk
  RingMask<Real_t> r7(2., 10. * std::cos(std::atan(2)), true, 0., kTwoPi); // full ring
  RingMask<Real_t> r8(10., 40., true, 0., kTwoPi);                         // full ring
  RingMask<Real_t> r9(12., 20., false, 0., kPi);                           // half ring
  RingMask<Real_t> r10(20., 30., false, -kPi / 6, kPi / 6);                // half ring

  TriangleMask<Real_t> tr1(10., 0., -10., -10., -10., 10.); // simple triangle
  TriangleMask<Real_t> tr2(-10., 0., 0., -5., 0.,
                           5.); // a triangle within tr1 touching it at the center of each segment
  TriangleMask<Real_t> tr3(-10.000001, 0., 0., -5., 0., 5.); // same as above but slightly extruding t1

  QuadrilateralMask<Real_t> q1(-20., 0., -10., -10., 10., 0., -10., 10.); // simple quadrilateral

  ZPhiMask<Real_t> z1(-10., 10., true, 5., 5.);
  ZPhiMask<Real_t> z2(-10., 10., false, 5., 5., 0., kPi / 4);
  ZPhiMask<Real_t> z3(-10., 10., false, 5., 5., kPi / 4, kTwoPi);
  ZPhiMask<Real_t> z4(-5575., 5575., true, 5., 5.);
  ZPhiMask<Real_t> z5(-21200., 21200., true, 5., 5.);

  // a few identity tests
  CheckFrames(w1, w1, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(r1, r1, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(r2, r2, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(r3, r3, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(r4, r4, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(r5, r5, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(r6, r6, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(tr1, tr1, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(tr2, tr2, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(tr3, tr3, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(q1, q1, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);
  CheckFrames(z1, z1, identity, FrameIntersect::kEqual, FrameIntersect::kEqual);

  // RingMask - RingMask
  CheckFrames(r2, r1, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(r1, r5, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(r5, r6, Transformation3DMP<Real_t>(11., 1., 0.), FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(r5, r6, Transformation3DMP<Real_t>(-19.000001, 0., 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);
  CheckFrames(r1, r8, Transformation3DMP<Real_t>(0., 0., 0., 0., 180., 0.), FrameIntersect::kEmbedded,
              FrameIntersect::kEmbedding);
  CheckFrames(r5, r9, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(r5, r9, Transformation3DMP<Real_t>(0., 0., 0., -90., 180., 0.), FrameIntersect::kEmbedding,
              FrameIntersect::kEmbedded);
  CheckFrames(r10, r10, Transformation3DMP<Real_t>(0., 0., 0., 0., 180., 0.), FrameIntersect::kEqual,
              FrameIntersect::kEqual);
  CheckFrames(r10, r10, Transformation3DMP<Real_t>(0., 0., 0., 180., 0., 180.), FrameIntersect::kEqual,
              FrameIntersect::kEqual);

  // WindowMask-WindowMask
  CheckFrames(w1, w2, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(w1, w2, Transformation3DMP<Real_t>(10., 10., 0.), FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(w1, w2, Transformation3DMP<Real_t>(-10., 10.000001, 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);
  CheckFrames(
      w1, w2,
      Transformation3DMP<Real_t>(0., 0., 0., kRadToDeg * (std::atan(2.) - std::acos(2. / std::sqrt(5.))), 0., 0.),
      FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(w1, w2, Transformation3DMP<Real_t>(30., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(w1, w2, Transformation3DMP<Real_t>(30. - 1.5 * vecgeom::kTolerance, 0., 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);
  CheckFrames(w1, w1, Transformation3DMP<Real_t>(40. - 1.5 * vecgeom::kTolerance, 0., 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);

  // WindowMask - Ringmask
  // Ring fully contained in the window
  CheckFrames(w6, r0, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  // Ring contained in the window but touching its sides
  CheckFrames(w6, r00, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  // Ring fully outside of window
  CheckFrames(w6, r0, Transformation3DMP<Real_t>(-15., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  // Ring overlapping outside of window
  CheckFrames(w6, r00, Transformation3DMP<Real_t>(-15., 0., 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);

  // RingMask - WindowMask
  // crossing only rmin
  CheckFrames(r5, w4, Transformation3DMP<Real_t>(0., -3., 0.), FrameIntersect::kIntersect, FrameIntersect::kIntersect);
  // crossing rmax
  CheckFrames(r5, w3, Transformation3DMP<Real_t>(0., -20., 0.), FrameIntersect::kIntersect, FrameIntersect::kIntersect);
  // embedded in phi1
  CheckFrames(r5, w3, Transformation3DMP<Real_t>(15., 1., 0.), FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  // rmin embedded in frame
  CheckFrames(r1, w5, identity, FrameIntersect::kIntersect, FrameIntersect::kIntersect);
  // Corner of the window in the center of the ring
  CheckFrames(r4, w3, Transformation3DMP<Real_t>(-1., 1., 0.), FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);

  // TriangleMask - WindowMask
  CheckFrames(tr1, w7, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(tr1, w8, identity, FrameIntersect::kIntersect, FrameIntersect::kIntersect);

  // TriangleMask - TriangleMask
  CheckFrames(tr1, tr2, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  CheckFrames(tr1, tr3, identity, FrameIntersect::kIntersect, FrameIntersect::kIntersect);

  // TriangleMask - RingMask
  // Ring within triangle, touching the sides
  CheckFrames(tr1, r7, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  // Ring outside of triangle
  CheckFrames(tr1, r0, identity, FrameIntersect::kIntersect, FrameIntersect::kIntersect);

  // QuadrilateralMask - TriangleMask
  // the triangle should be fully inside the quadrilateral, as they have 3 common vertices
  CheckFrames(q1, tr1, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  // ring within quadrilateral
  CheckFrames(q1, r7, identity, FrameIntersect::kEmbedding, FrameIntersect::kEmbedded);
  // Ring outside of quadrilateral
  CheckFrames(q1, r0, identity, FrameIntersect::kIntersect, FrameIntersect::kIntersect);

  // ZPhiMask - ZPhiMask
  CheckFrames(z1, z1, Transformation3DMP<Real_t>(0., 0., 1.), FrameIntersect::kIntersect, FrameIntersect::kIntersect);
  CheckFrames(z2, z3, identity, FrameIntersect::kNoIntersect, FrameIntersect::kNoIntersect);
  CheckFrames(z4, z5, Transformation3DMP<Real_t>(0., 0., 21125.) * Transformation3DMP<Real_t>(0., 0., 5650.).Inverse(),
              FrameIntersect::kEmbedded, FrameIntersect::kEmbedding);

  // Tests for IsDisjoint
  // ring-ring
  CheckFrames(r0, r1, Transformation3DMP<Real_t>(0., 0., 5.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(r0, r1, Transformation3DMP<Real_t>(0., 0., 5.000001), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);
  CheckFrames(r0, r1, Transformation3DMP<Real_t>(25., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(r0, r1, Transformation3DMP<Real_t>(25. - 0.000001, 0., 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);

  // ring-window
  CheckFrames(r0, w3, Transformation3DMP<Real_t>(6., 0.5, 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(r1, w3, Transformation3DMP<Real_t>(9., 0., 0.), FrameIntersect::kIntersect, FrameIntersect::kIntersect);
  CheckFrames(r1, w3, Transformation3DMP<Real_t>(8.5857, 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  // ring-triangle
  CheckFrames(r1, tr2, Transformation3DMP<Real_t>(30., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  // ring-quadrilateral
  CheckFrames(r2, q1, Transformation3DMP<Real_t>(-30., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  // zphi-zphi
  CheckFrames(z2, z3, identity, FrameIntersect::kNoIntersect, FrameIntersect::kNoIntersect);
  // window-window
  CheckFrames(w3, w3, Transformation3DMP<Real_t>(2., 2., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  // window-triangle
  CheckFrames(w3, tr1, Transformation3DMP<Real_t>(11. - 2 * vecgeom::kTolerance, 0., 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);
  CheckFrames(w3, tr1, Transformation3DMP<Real_t>(11., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(w3, tr1, Transformation3DMP<Real_t>(-11., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(tr1, w3, Transformation3DMP<Real_t>(-10., -11., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(tr1, w3, Transformation3DMP<Real_t>(-10., -11. + 2 * vecgeom::kTolerance, 0.), FrameIntersect::kIntersect,
              FrameIntersect::kIntersect);
  // window-quadrilateral
  CheckFrames(w3, q1, Transformation3DMP<Real_t>(-10., 0., 0.), FrameIntersect::kIntersect, FrameIntersect::kIntersect);
  CheckFrames(w3, q1, Transformation3DMP<Real_t>(-11., 0., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  CheckFrames(w3, q1, Transformation3DMP<Real_t>(0., -7., 0.), FrameIntersect::kNoIntersect,
              FrameIntersect::kNoIntersect);
  // triangle-quadrilateral
  CheckFrames(tr1, q1, Transformation3DMP<Real_t>(19., 0., 0.), FrameIntersect::kIntersect, FrameIntersect::kIntersect);
  // quadrilateral-quadrilateral
  CheckFrames(q1, q1, Transformation3DMP<Real_t>(29., 0., 0.), FrameIntersect::kIntersect, FrameIntersect::kIntersect);

  // == more to add...
  return 0;
}
