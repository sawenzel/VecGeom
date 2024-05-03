#include <VecGeom/surfaces/mask/FrameMasks.h>

using namespace vgbrep;
using namespace vecgeom;

int main(int argc, char *argv[])
{
  // create some frames
  using Real_t = vecgeom::Precision;

  vecgeom::Transformation3DMP<Real_t> identity;
  vecgeom::Transformation3DMP<Real_t> t1(10., 10., 0.);
  vecgeom::Transformation3DMP<Real_t> t2(-10., 10.000001, 0.);
  vecgeom::Transformation3DMP<Real_t> t3(0., 0., 0., kRadToDeg * (std::atan(2.) - std::acos(2. / std::sqrt(5.))), 0.,
                                         0.);
  vecgeom::Transformation3DMP<Real_t> t4(11., 1., 0.);
  vecgeom::Transformation3DMP<Real_t> t5(-19.000001, 0., 0.);
  vecgeom::Transformation3DMP<Real_t> t6(0., -3., 0.);
  vecgeom::Transformation3DMP<Real_t> t7(0., -20., 0.);
  vecgeom::Transformation3DMP<Real_t> t8(15., 1., 0.);
  vecgeom::Transformation3DMP<Real_t> t9(-15., 0., 0.);
  vecgeom::Transformation3DMP<Real_t> t10(0., 0., 1.);

  // WindowMask-WindowMask
  WindowMask<Real_t> w1(20., 30.);
  WindowMask<Real_t> w2(10., 20.);
  WindowMask<Real_t> w3(1., 1.);
  WindowMask<Real_t> w4(1., 15.);
  WindowMask<Real_t> w5(12., 12.);
  bool embedded_w1_w2_id = FrameChecker<Real_t, WindowMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(w1, w2, identity);
  assert(embedded_w1_w2_id);
  bool embedded_w1_w2_t1 = FrameChecker<Real_t, WindowMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(w1, w2, t1);
  assert(embedded_w1_w2_t1);
  bool embedded_w1_w2_t2 = FrameChecker<Real_t, WindowMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(w1, w2, t2);
  assert(!embedded_w1_w2_t2);
  bool embedded_w1_w2_t3 = FrameChecker<Real_t, WindowMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(w1, w2, t3);
  assert(embedded_w1_w2_t3);

  // WindowMask - Ringmask
  WindowMask<Real_t> w6(10., 10.);
  RingMask<Real_t> r0(2., 5., true, 0., kTwoPi);   // full ring
  RingMask<Real_t> r00(2., 10., true, 0., kTwoPi); // full ring
  // Ring fully contained in the window
  bool embedded_w6_r0_id = FrameChecker<Real_t, WindowMask<Real_t>, RingMask<Real_t>>::IsEmbedding(w6, r0, identity);
  assert(embedded_w6_r0_id);
  // Ring contained in the window but touching its sides
  bool embedded_w6_r00_id = FrameChecker<Real_t, WindowMask<Real_t>, RingMask<Real_t>>::IsEmbedding(w6, r00, identity);
  assert(embedded_w6_r00_id);
  // Ring fully outside of window
  bool embedded_w6_r0_t9 = FrameChecker<Real_t, WindowMask<Real_t>, RingMask<Real_t>>::IsEmbedding(w6, r0, t9);
  assert(!embedded_w6_r0_t9);
  // Ring fully outside of window
  bool embedded_w6_r00_t9 = FrameChecker<Real_t, WindowMask<Real_t>, RingMask<Real_t>>::IsEmbedding(w6, r00, t9);
  assert(!embedded_w6_r00_t9);

  // RingMask - RingMask
  RingMask<Real_t> r1(10., 20., true, 0., kTwoPi);     // full ring
  RingMask<Real_t> r2(0., 20., true, 0., kTwoPi);      // full disk
  RingMask<Real_t> r3(0., 20., false, 0., kHalfPi);    // convex pie
  RingMask<Real_t> r4(0., 20., false, 0., 1.5 * kPi);  // concave pie
  RingMask<Real_t> r5(10., 20., false, 0., 1.5 * kPi); // concave cut ring
  RingMask<Real_t> r6(0., 1., true, 0., kTwoPi);       // smaller full disk
  bool embedded_r1_r2_id = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r1, r2, identity);
  assert(!embedded_r1_r2_id);
  bool embedded_r2_r1_id = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r2, r1, identity);
  assert(embedded_r2_r1_id);
  bool embedded_r1_r5_id = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r1, r5, identity);
  assert(embedded_r1_r5_id);
  bool embedded_r5_r6_t4 = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r5, r6, t4);
  assert(embedded_r5_r6_t4);
  bool embedded_r5_r6_t5 = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r5, r6, t5);
  assert(!embedded_r5_r6_t5);
  // a few identity tests
  bool embedded_r1_r1_id = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r1, r1, identity);
  assert(embedded_r1_r1_id);
  bool embedded_r3_r3_id = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r3, r3, identity);
  assert(embedded_r3_r3_id);
  bool embedded_r5_r5_id = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r5, r5, identity);
  assert(embedded_r5_r5_id);
  bool embedded_r6_r6_id = FrameChecker<Real_t, RingMask<Real_t>, RingMask<Real_t>>::IsEmbedding(r6, r6, identity);
  assert(embedded_r6_r6_id);

  // RingMask - WindowMask
  // crossing only rmin
  bool embedded_r5_w4_t6 = FrameChecker<Real_t, RingMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(r5, w4, t6);
  assert(!embedded_r5_w4_t6);
  // crossing rmax
  bool embedded_r5_w3_t7 = FrameChecker<Real_t, RingMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(r5, w3, t7);
  assert(!embedded_r5_w3_t7);
  // embedded in phi1
  bool embedded_r5_w3_t8 = FrameChecker<Real_t, RingMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(r5, w3, t8);
  assert(embedded_r5_w3_t8);
  // rmin embedded in frame
  bool embedded_r1_w5_id = FrameChecker<Real_t, RingMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(r1, w5, identity);
  assert(!embedded_r1_w5_id);

  // TriangleMask - WindowMask
  //  TriangleMask<Real_t> tr1(-10., 10., -10., -10., 10., 0.);     // simple triangle
  TriangleMask<Real_t> tr1(10., 0., -10., -10., -10., 10.); // simple triangle

  TriangleMask<Real_t> tr2(-10., 0., 0., -5., 0.,
                           5.); // a triangle within tr1 touching it at the center of each segment
  TriangleMask<Real_t> tr3(-10.000001, 0., 0., -5., 0., 5.); // same as above but slightly extruding t1

  WindowMask<Real_t> w7(-10., 0., -5., 5.);
  WindowMask<Real_t> w8(-10., 0.0001, -5., 5.);

  bool embedded_tr1_w7_id =
      FrameChecker<Real_t, TriangleMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(tr1, w7, identity);
  assert(embedded_tr1_w7_id);
  bool embedded_tr1_w8_id =
      FrameChecker<Real_t, TriangleMask<Real_t>, WindowMask<Real_t>>::IsEmbedding(tr1, w8, identity);
  assert(!embedded_tr1_w8_id);

  // TriangleMask - TriangleMask
  bool embedded_tr1_tr1_id =
      FrameChecker<Real_t, TriangleMask<Real_t>, TriangleMask<Real_t>>::IsEmbedding(tr1, tr1, identity);
  assert(embedded_tr1_tr1_id);
  bool embedded_tr1_tr2_id =
      FrameChecker<Real_t, TriangleMask<Real_t>, TriangleMask<Real_t>>::IsEmbedding(tr1, tr2, identity);
  assert(embedded_tr1_tr2_id);
  bool embedded_tr1_tr3_id =
      FrameChecker<Real_t, TriangleMask<Real_t>, TriangleMask<Real_t>>::IsEmbedding(tr1, tr3, identity);
  assert(!embedded_tr1_tr3_id);

  // TriangleMask - RingMask
  // Ring within triangle, touching the sides
  RingMask<Real_t> r7(2., 10. * std::cos(std::atan(2)), true, 0., kTwoPi); // full ring
  bool embedded_tr1_r7_id =
      FrameChecker<Real_t, TriangleMask<Real_t>, RingMask<Real_t>>::IsEmbedding(tr1, r7, identity);
  assert(embedded_tr1_r7_id);
  // Ring outside of triangle
  bool embedded_tr1_r0_t9 = FrameChecker<Real_t, TriangleMask<Real_t>, RingMask<Real_t>>::IsEmbedding(tr1, r0, t9);
  assert(!embedded_tr1_r0_t9);

  // QuadrilateralMask - TriangleMask
  QuadrilateralMask<Real_t> q1(-20., 0., -10., -10., 10., 0., -10., 10.); // simple quadrilateral
  // the triangle should be fully inside the quadrilateral, as they have 3 common vertices
  bool embedded_q1_tr1_id =
      FrameChecker<Real_t, QuadrilateralMask<Real_t>, TriangleMask<Real_t>>::IsEmbedding(q1, tr1, identity);
  assert(embedded_q1_tr1_id);
  // ring within quadrilateral
  bool embedded_q1_r7_id =
      FrameChecker<Real_t, QuadrilateralMask<Real_t>, RingMask<Real_t>>::IsEmbedding(q1, r7, identity);
  assert(embedded_q1_r7_id);
  // Ring outside of quadrilateral
  bool embedded_q1_r0_t9 = FrameChecker<Real_t, QuadrilateralMask<Real_t>, RingMask<Real_t>>::IsEmbedding(q1, r0, t9);
  assert(!embedded_q1_r0_t9);

  // ZPhiMask - ZPhiMask
  ZPhiMask<Real_t> z1(-10., 10., true, 5., 5.);
  bool embedded_z1_z1_id = FrameChecker<Real_t, ZPhiMask<Real_t>, ZPhiMask<Real_t>>::IsEmbedding(z1, z1, identity);
  assert(embedded_z1_z1_id);
  bool embedded_z1_z1_t10 = FrameChecker<Real_t, ZPhiMask<Real_t>, ZPhiMask<Real_t>>::IsEmbedding(z1, z1, t10);
  assert(!embedded_z1_z1_t10);

  // == more to add...
  return 0;
}
