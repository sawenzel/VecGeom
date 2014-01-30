#ifndef KERNELTUBE_H
#define KERNELTUBE_H

#include "LibraryGeneric.h"
#include "Tube.h"

namespace kernel {

namespace tube {

template <ImplType it>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
void DetermineRHit(
    Vector3D<typename Impl<it>::float_v> const &pos,
    Vector3D<typename Impl<it>::float_v> const &dir,
    typename Impl<it>::float_v           const &distance_r,
    typename Impl<it>::bool_v            &hit) {

  typedef typename Impl<it>::bool_v Bool;

  Bool hit_x = pos[0] + distance_r*dir[0];
  Bool hit_y = pos[1] + distance_r*dir[1];

  return distance_r > 0;

}

template <ImplType it>
inline __attribute__((always_inline))
CUDA_HEADER_BOTH
void DistanceToIn(
    TubeParameters                          const &tube,
    TransMatrix<typename Impl<it>::float_t> const * const matrix,
    Vector3D   <typename Impl<it>::float_v> const &pos,
    Vector3D   <typename Impl<it>::float_v> const &dir,
    typename Impl<it>::float_v              const &step_max,
    typename Impl<it>::float_v              &distance) {

  typedef typename Impl<it>::float_v Float;
  typedef typename Impl<it>::bool_v Bool;

  Bool done = Impl<it>::kFalse;
  distance = kInfinity;
  Vector3D<Float> pos_local;
  Vector3D<Float> dir_local;
  Bool in_z;
  Float safety_z;

  matrix->Transform(pos, pos_local);
  matrix->TransformRotation(dir, dir_local);

  // Check safety for z
  safety_z = tube.z - Abs<it>(pos_local[2]);
  in_z = safety_z > kGTolerance;
  done |= !in_z && (tube.z * dir_local[2] >= 0.0);

  if (done == Impl<it>::kTrue) return;

  // Solve second order equation
  //
  // a * t^2 + b * t + c = 0
  //
  // which must satisfy
  //
  // b^2 -  4 a*c > 0
  //
  // with solutions given by
  //
  // t = (-b +- SQRT(d)) / (2a)
  // for
  // b^2 -  4 a*c > 0
  //
  // a -- independent on shape
  // b -- independent on shape
  // c -- dependent on shape

  Float r2 = pos_local[0]*pos_local[0] + pos_local[1]*pos_local[1];
  Float n2 = 1.0 - dir_local[2]*dir_local[2];
  Float r_dot_n = pos_local[0]*dir_local[0] + pos_local[1]*dir_local[1];
  
  Float a = n2;
  Float b = 2.0*r_dot_n;
  Float c = r2 - tube.r_max*tube.r_max;

  Float d = b*b - 4.0*a*c;

  Bool can_hit_r_max = (d >= 0.0);
  done |= !can_hit_r_max;

  if (done == Impl<it>::kTrue) return;

  // Check outer cylinder

  Float inv_2a = 1.0 / (2.0*a);
  Float distance_r_max(kInfinity);
  MaskedAssign(can_hit_r_max, (-b - Sqrt<it>(d))*inv_2a, distance_r_max);

  Bool done_r(Impl<it>::kFalse);
  DetermineRHit(pos_local, dir_local, distance_r_max, done_r);
  MaskedAssign(!done_r, kInfinity, distance_r_max);

  // Check inner cylinder

  d = d - 4.0*a*(tube.r_max*tube.r_max - tube.r_min*tube.r_min);

  Bool can_hit_r_min = (d >= 0.0);
  Float distance_r_min(kInfinity);
  MaskedAssign(can_hit_r_min, (-b + Sqrt<it>(d))*inv_2a, distance_r_min);

  Bool done_r_min;
  DetermineRHit(pos_local, dir_local, distance_r_min, done_r_min);
  MaskedAssign(!done_r_min, kInfinity, distance_r_min);

  // distance_r_max = Min<it>(distance_r_min, distance_r_max);

}

} // End namespace tube

} // End namespace kernel

#endif /* KERNELTUBE_H */