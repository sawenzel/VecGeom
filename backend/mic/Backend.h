/// \file mic/backend.h

#ifndef VECGEOM_BACKEND_MICBACKEND_H_
#define VECGEOM_BACKEND_MICBACKEND_H_

#include "base/Global.h"
#include "backend/scalar/Backend.h"

#include <micvec.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class MicIntegerVector;
class MicDoubleVector;

struct kMic {
  typedef MicIntegerVector int_v;
  typedef MicDoubleVector precision_v;
  typedef VecMask16 bool_v;
  typedef MicIntegerVector inside_v;
  constexpr static bool early_returns = false;
  const static precision_v kOne;
  const static precision_v kZero;
  const static bool_v kTrue;
  const static bool_v kFalse;
  // alternative typedefs ( might supercede above typedefs )
  typedef MicIntegerVector Int_t;
  typedef MicDoubleVector Double_t;
  typedef VecMask16 Bool_t;
  typedef MicDoubleVector Index_t;
};

#ifdef kVectorSize
#undef kVectorSize
#endif
constexpr int kVectorSize = 8;

typedef kMic::int_v       MicInt;
typedef kMic::precision_v MicPrecision;
typedef kMic::bool_v      MicBool;
typedef kMic::inside_v    MicInside;

// Class, operators and functinos for Integer

class MicIntegerVector : public Is32vec16 {

public:
  MicIntegerVector() : Is32vec16() {}
  MicIntegerVector(__m512i mm) : Is32vec16(mm) {}
  MicIntegerVector(const int i) : Is32vec16(i) {}
  MicIntegerVector(const int i[16]) : Is32vec16((int*)i) {}
  MicIntegerVector(Is32vec16 mm) : MicIntegerVector((__m512i)mm) {}

};

// Class, operators and functions for Double

class MicDoubleVector : public F64vec8 {

public:
  MicDoubleVector() : F64vec8() {}
  MicDoubleVector(__m512d m) : F64vec8(m) {}
  MicDoubleVector(const double d) : F64vec8(d) {}
  MicDoubleVector(const double d[8]) : F64vec8((double*)d) {}
  MicDoubleVector(F64vec8 m) : MicDoubleVector((__m512d)m) {}

  VECGEOM_INLINE
  MicDoubleVector operator = (Precision const &val) {
    return MicDoubleVector(vec = _mm512_set1_pd(val));
  }

  VECGEOM_INLINE
  MicDoubleVector operator = (MicDoubleVector const &val) {
    return MicDoubleVector(vec = val.vec);
  }

  VECGEOM_INLINE
  MicDoubleVector operator + (Precision const &val) const {
    return (*this) + MicDoubleVector(val);
  }

  VECGEOM_INLINE
  MicDoubleVector operator - () const {
    return MicPrecision(0.0) - vec;
  }

  VECGEOM_INLINE
  MicDoubleVector operator - (Precision const &val) const {
    return (*this) - MicDoubleVector(val);
  }

};

// Operators and Functions for MicDoubleVector/MicPrecision

VECGEOM_INLINE
VecMask16 operator == (MicPrecision const &val1,
                       Precision const &val2) {
  return cmpeq(val1,MicPrecision(val2));
}

VECGEOM_INLINE
VecMask16 operator != (MicPrecision const &val1,
                       MicPrecision const &val2) {
  return val1 != val2;
}

VECGEOM_INLINE
VecMask16 operator != (MicPrecision const &val1,
                       Precision const &val2) {
  return val1 != MicPrecision(val2);
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
  return MicPrecision(MicPrecision((double)val1) - val2);
}

VECGEOM_INLINE
MicPrecision operator + (Precision const &val1,
                         MicPrecision const &val2) {
  return MicPrecision(val1) + val2;
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

VECGEOM_INLINE
MicPrecision Max(MicPrecision const &val1, MicPrecision const &val2) {
  return max(val1, val2);
}

VECGEOM_INLINE
MicPrecision Min(MicPrecision const &val1, MicPrecision const &val2) {
  return min(val1, val2);
}

VECGEOM_INLINE
MicPrecision ATan2(MicPrecision const &y, MicPrecision const &x) {
  return atan2(x, y);
}

// Operators and Functions for Is32vec16/Inside_v

VECGEOM_INLINE
void MaskedAssign(MicBool const &cond,
                  Inside_t const &thenval,
                  MicInside * const output) {
  MaskedAssign(cond, thenval, output);
}

VECGEOM_INLINE
void MaskedAssign(MicBool const &cond,
                  MicInside const &thenval,
                  MicInside * const output) {
  (*output) = _mm512_mask_or_epi64(*output, cond, thenval, thenval);
}

} // End inline namespace

} // End global namespace


#endif // VECGEOM_BACKEND_VCBACKEND_H_
