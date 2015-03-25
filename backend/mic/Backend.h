/// \file mic/backend.h

#ifndef VECGEOM_BACKEND_MICBACKEND_H_
#define VECGEOM_BACKEND_MICBACKEND_H_

#include "base/Global.h"
#include "backend/scalar/Backend.h"

#include <micvec.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

struct kMic {
  typedef Is32vec16 int_v;
  typedef F64vec8 precision_v;
  typedef VecMask16 bool_v;
  typedef Is32vec16 inside_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
  // alternative typedefs ( might supercede above typedefs )
  typedef Is32vec16 Int_t;
  typedef F64vec8 Double_t;
  typedef Is32vec16 Bool_t;
  typedef F64vec8 Index_t;
};

constexpr int kVectorSize = 8;

typedef kMic::int_v       MicInt;
typedef kMic::precision_v MicPrecision;
typedef kMic::bool_v      MicBool;
typedef kMic::inside_v    MicInside;

/*VECGEOM_INLINE
MicPrecision operator >=(MicPrecision const &val1,
			 MicPrecision const &val2) {
  return ;
}


template <typename Type>
VECGEOM_INLINE
void CondAssign(typename VecMask16 const &cond,
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
bool IsFull(VcBool const &cond) {
  return cond.isFull();
}

VECGEOM_INLINE
bool Any(VcBool const &cond) {
  return !cond.isEmpty();
}

VECGEOM_INLINE
bool IsEmpty(VcBool const &cond) {
  return cond.isEmpty();
}
*/
VECGEOM_INLINE
F64vec8 Abs(F64vec8 const &val) {
  F64vec8 _v(_mm512_set1_pd(-0.0));
  return val&(val^_v);
}
/*
VECGEOM_INLINE
VcPrecision Sqrt(VcPrecision const &val) {
  return Vc::sqrt(val);
}

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


VECGEOM_INLINE
VcPrecision Floor( VcPrecision const &val ){
  return Vc::floor( val );
}

*/

} // End inline namespace

} // End global namespace


#endif // VECGEOM_BACKEND_VCBACKEND_H_
