/// \file mic/backend.h

#ifndef VECGEOM_BACKEND_MICBACKEND_H_
#define VECGEOM_BACKEND_MICBACKEND_H_

#include "base/Global.h"
#include "backend/scalar/Backend.h"

#include <micvec.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class MicVectorRegister;

struct kMic {
  typedef Is32vec16 int_v;
  typedef MicVectorRegister precision_v;
  typedef VecMask16 bool_v;
  typedef Is32vec16 inside_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
  // alternative typedefs ( might supercede above typedefs )
  typedef Is32vec16 Int_t;
  typedef MicVectorRegister Double_t;
  typedef Is32vec16 Bool_t;
  typedef MicVectorRegister Index_t;
};

#ifdef kVectorSize
#undef kVectorSize
#endif
constexpr int kVectorSize = 8;

typedef kMic::int_v       MicInt;
typedef kMic::precision_v MicPrecision;
typedef kMic::bool_v      MicBool;
typedef kMic::inside_v    MicInside;

class MicVectorRegister : public F64vec8 {

public:
  MicVectorRegister() : F64vec8() {}
  MicVectorRegister(__m512d m) : F64vec8(m) {}
  MicVectorRegister(const double d) : F64vec8(d) {}
  MicVectorRegister(const double d[8]) : F64vec8((double*)d) {}
  MicVectorRegister(F64vec8 m) : MicVectorRegister((__m512d)m) {}

  VECGEOM_INLINE
  MicVectorRegister operator = (Precision const &val) {
    return MicVectorRegister(vec = _mm512_set1_pd(val));
  }

  VECGEOM_INLINE
  MicVectorRegister operator = (MicVectorRegister const &val) {
    return MicVectorRegister(vec = val.vec);
  }

  VECGEOM_INLINE
  MicVectorRegister operator + (Precision const &val) const {
    return (*this) * MicVectorRegister(val);
  }

};

VECGEOM_INLINE
VecMask16 operator == (MicPrecision const &val1,
                       Precision const &val2) {
  return cmpeq(val1,MicPrecision(val2));
}

VECGEOM_INLINE
VecMask16 operator != (MicPrecision const &val1,
                       Precision const &val2) {
  return cmpneq(val1,MicPrecision(val2));
}

VECGEOM_INLINE
VecMask16 operator > (MicPrecision const &val1,
                      MicPrecision const &val2) {
  return cmpnle(val1, val2);
}

VECGEOM_INLINE
VecMask16 operator > (MicPrecision const &val1,
                      Precision const &val2) {
  return val1 > MicPrecision(val2);
}

VECGEOM_INLINE
VecMask16 operator >= (MicPrecision const &val1,
                       MicPrecision const &val2) {
  return cmpnlt(val1, val2);
}

VECGEOM_INLINE
VecMask16 operator >= (MicPrecision const &val1,
                       Precision const &val2) {
  return val1 >= MicPrecision(val2);
}

VECGEOM_INLINE
VecMask16 operator < (MicPrecision const &val1,
                      MicPrecision const &val2) {
  return cmplt(val1, val2);
}

VECGEOM_INLINE
VecMask16 operator < (MicPrecision const &val1,
                      int const &val2) {
  return val1 < MicPrecision((double)val2);
}

VECGEOM_INLINE
VecMask16 operator <= (MicPrecision const &val1,
                       MicPrecision const &val2) {
  return cmple(val1, val2);
}

VECGEOM_INLINE
VecMask16 operator <= (MicPrecision const &val1,
                       Precision const &val2) {
  return val1 <= MicPrecision(val2);
}

VECGEOM_INLINE
VecMask16 operator <= (Precision const &val1,
                       MicPrecision const &val2) {
  return MicPrecision(val1) <= val2;
}

VECGEOM_INLINE
MicPrecision operator - (int const &val1,
                         MicPrecision const &val2) {
  return MicPrecision((double)val1) - val2;
}

VECGEOM_INLINE
MicPrecision operator * (Precision const &val1,
                         MicPrecision const &val2) {
  return MicPrecision(val1) * val2;
}

VECGEOM_INLINE
MicPrecision operator / (int const &val1,
                         MicPrecision const &val2) {
  return MicPrecision((double)val1) / val2;
}

VECGEOM_INLINE
void MaskedAssign(MicBool const &cond,
                  MicPrecision const &thenval,
                  MicPrecision * const output) {
  (*output) = _mm512_castsi512_pd(_mm512_mask_or_epi64(_mm512_castpd_si512(*output), cond, _mm512_castpd_si512(thenval), _mm512_castpd_si512(thenval)));
}

VECGEOM_INLINE
MicPrecision Abs(F64vec8 const &val) {
  MicPrecision _v(-0.0);
  return val & (val ^_v);
}

VECGEOM_INLINE
MicPrecision Abs(MicPrecision const &val) {
  MicPrecision _v(-0.0);
  return val & (val ^ _v);
}

VECGEOM_INLINE
MicPrecision Sqrt(MicPrecision const &val) {
  return sqrt(val);
}

VECGEOM_INLINE
MicPrecision Sqrt(F64vec8 const &val) {
  return MicPrecision(sqrt(val));
}

} // End inline namespace

} // End global namespace


#endif // VECGEOM_BACKEND_VCBACKEND_H_
