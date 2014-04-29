#ifndef VECGEOM_GLOBAL_H_
#define VECGEOM_GLOBAL_H_

#include <cmath>

#define VECGEOM_VC

#ifdef VECGEOM_VC
#include <Vc/Vc>
struct kVc {
  typedef Vc::Vector<double> double_v;
  typedef double_v::Mask bool_v;
};
typedef kVc::double_v VcDouble;
typedef kVc::bool_v VcBool;
VcDouble fabs(VcDouble const &in) { return Vc::abs(in); }
#endif

struct kScalar {
  typedef double double_v;
  typedef bool bool_v;
};

template <typename Type> class Vector3D;

#endif // VECGEOM_GLOBAL_H_