/// \file vc/backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_VCBACKEND_H_
#define VECGEOM_BACKEND_VCBACKEND_H_

#include "base/Global.h"

#include "backend/scalar/Backend.h"

#include <Vc/Vc>

namespace VECGEOM_NAMESPACE {

struct kVc {
  typedef Vc::int_v                   int_v;
  typedef Vc::Vector<Precision>       precision_v;
  typedef Vc::Vector<Precision>::Mask bool_v;
  typedef Vc::Vector<int>             inside_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
  // alternative typedefs ( might supercede above typedefs )
  typedef Vc::int_v                   Int_t;
  typedef Vc::Vector<Precision>       Double_t;
  typedef Vc::Vector<Precision>::Mask Bool_t;
};

constexpr int kVectorSize = kVc::precision_v::Size;

typedef kVc::int_v       VcInt;
typedef kVc::precision_v VcPrecision;
typedef kVc::bool_v      VcBool;
typedef kVc::inside_v    VcInside;

template <typename Type>
VECGEOM_INLINE
void CondAssign(typename Vc::Vector<Type>::Mask const &cond,
                Vc::Vector<Type> const &thenval,
                Vc::Vector<Type> const &elseval,
                Vc::Vector<Type> *const output) {
  (*output)(cond) = thenval;
  (*output)(!cond) = elseval;
}

template <typename Type>
VECGEOM_INLINE
void CondAssign(typename Vc::Vector<Type>::Mask const &cond,
                Type const &thenval,
                Type const &elseval,
                Vc::Vector<Type> *const output) {
  (*output)(cond) = thenval;
  (*output)(!cond) = elseval;
}

template <typename Type>
VECGEOM_INLINE
void MaskedAssign(typename Vc::Vector<Type>::Mask const &cond,
                  Vc::Vector<Type> const &thenval,
                  Vc::Vector<Type> *const output) {
  (*output)(cond) = thenval;
}

template <typename Type>
VECGEOM_INLINE
void MaskedAssign(typename Vc::Vector<Type>::Mask const &cond,
                  Type const &thenval,
                  Vc::Vector<Type> *const output) {
  (*output)(cond) = thenval;
}

// VECGEOM_INLINE
// void MaskedAssign(VcBool const &cond,
//                   const kScalar::int_v thenval,
//                   VcInt *const output) {
//   (*output)(VcInt::Mask(cond)) = thenval;
// }

VECGEOM_INLINE
void MaskedAssign(VcBool const &cond,
                  const Inside_t thenval,
                  VcInside *const output) {
  (*output)(VcInside::Mask(cond)) = thenval;
}


VECGEOM_INLINE
bool IsFull(VcBool const &cond){
    return cond.isFull();
}

VECGEOM_INLINE
bool IsEmpty(VcBool const &cond){
    return cond.isEmpty();
}

VECGEOM_INLINE
VcPrecision Abs(VcPrecision const &val) {
  return Vc::abs(val);
}

VECGEOM_INLINE
VcPrecision Sqrt(VcPrecision const &val) {
  return Vc::sqrt(val);
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
VcPrecision ATan2(VcPrecision const &y, VcPrecision const &x) {
  return Vc::atan2(y, x);
}


VECGEOM_INLINE
VcPrecision sin(VcPrecision const &x) {
  return Vc::sin(x);
}

VECGEOM_INLINE
VcPrecision cos(VcPrecision const &x) {
  return Vc::cos(x);
}

VECGEOM_INLINE
VcPrecision tan(VcPrecision const &radians) {
  // apparently Vc does not have a tan function
  //  return Vc::tan(radians);
  // emulating it for the moment
  VcPrecision s,c;
  Vc::sincos(radians,&s,&c);
  return s/c;
}

VECGEOM_INLINE
Precision Pow(Precision const &x, Precision arg) {
   return std::pow(x,arg);
}

VECGEOM_INLINE
VcPrecision Min(VcPrecision const &val1, VcPrecision const &val2) {
  return Vc::min(val1, val2);
}

VECGEOM_INLINE
VcPrecision Max(VcPrecision const &val1, VcPrecision const &val2) {
  return Vc::max(val1, val2);
}

VECGEOM_INLINE
VcInt Min(VcInt const &val1, VcInt const &val2) {
  return Vc::min(val1, val2);
}

VECGEOM_INLINE
VcInt Max(VcInt const &val1, VcInt const &val2) {
  return Vc::max(val1, val2);
}

} // End global namespace

#endif // VECGEOM_BACKEND_VCBACKEND_H_
