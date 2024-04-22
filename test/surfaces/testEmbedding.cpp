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

  // == more to add...
  return 0;
}
