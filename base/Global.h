/// \file Global.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_GLOBAL_H_
#define VECGEOM_BASE_GLOBAL_H_

#include <cassert>
#include <cmath>
#include <cfloat>
#include <limits>
#include <memory>
#include <cstdio>
#include <cstdlib>

#define VECGEOM

#if __cplusplus >= 201103L
  #define VECGEOM_STD_CXX11
#endif

#if (defined(__CUDACC__) || defined(__NVCC__))
  // Compiling with nvcc
  #define VECGEOM_NVCC
  #ifdef __CUDA_ARCH__
    // Compiling device code
    #define VECGEOM_NVCC_DEVICE
  #endif
  #define HAVECUDANAMESPACE
  #define VECGEOM_IMPL_NAMESPACE cuda
  #define VECGEOM_NAMESPACE ::vecgeom
  #define VECGEOM_CUDA_HEADER_HOST __host__
  #define VECGEOM_CUDA_HEADER_DEVICE __device__
  #define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
  #define VECGEOM_CUDA_HEADER_GLOBAL __global__
  #define VECGEOM_ALIGNED __align__((64))
  #define VECGEOM_HOST_FORWARD_DECLARE(X) namespace cxx { X }
  #define VECGEOM_DEVICE_FORWARD_DECLARE(X)
  #define VECGEOM_DEVICE_DECLARE_CONV(X)
  #define VECGEOM_DEVICE_DECLARE_NS_CONV(NS,X,Def)
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(X,ArgType)
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(X,ArgType1,Def1,ArgType2,Def2)
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(X,ArgType1,Def1,ArgType2,Def2,ArgType3)
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3)
  #undef VECGEOM_VC
  #undef VECGEOM_VC_ACCELERATION
  #undef VECGEOM_CILK
  #undef VECGEOM_ROOT
  #undef VECGEOM_USOLIDS
  #undef VECGEOM_GEANT4
  #undef VECGEOM_BENCHMARK
#else
  // Not compiling with NVCC
#define HAVENORMALNAMESPACE
  #define VECGEOM_IMPL_NAMESPACE cxx
  #define VECGEOM_NAMESPACE ::vecgeom
  #define VECGEOM_CUDA_HEADER_HOST
  #define VECGEOM_CUDA_HEADER_DEVICE
  #define VECGEOM_CUDA_HEADER_BOTH
  #define VECGEOM_CUDA_HEADER_GLOBAL
  #ifdef VECGEOM_CUDA
    // CUDA is enabled, but currently compiling regular C++ code.
    // This enables methods that interface between C++ and CUDA environments
    #define VECGEOM_CUDA_INTERFACE
  #endif
  namespace vecgeom {
     template <typename DataType> struct kCudaType;
     template <typename DataType> using CudaType_t = typename kCudaType<DataType>::type_t;
     template <> struct kCudaType<float> { using type_t = float; };
     template <> struct kCudaType<double> { using type_t = double; };
     template <> struct kCudaType<int> { using type_t = int; };
  }
  #define VECGEOM_HOST_FORWARD_DECLARE(X)
  #define VECGEOM_DEVICE_FORWARD_DECLARE(X)  namespace cuda { X }

  #define VECGEOM_DEVICE_DECLARE_CONV(X) \
     namespace cuda { class X; } \
     inline namespace cxx  { class X; } \
     template <> struct kCudaType<cxx::X> { using type_t = cuda::X; };

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(X,ArgType) \
     namespace cuda { template <ArgType Arg> class X; } \
     inline namespace cxx  { template <ArgType Arg> class X; } \
     template <ArgType Arg> struct kCudaType<cxx::X<Arg> > \
     { using type_t = cuda::X<CudaType_t<Arg> >; };

#ifdef VECGEOM_CUDA_VOLUME_SPECIALIZATION

  #define VECGEOM_DEVICE_DECLARE_NS_CONV(NS,X,Def)     \
     namespace cuda { namespace NS { class X; } } \
     inline namespace cxx { namespace NS { class X; } } \
     template <> struct kCudaType<cxx::NS::X> { using type_t = cuda::NS::X; };

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(X,ArgType1,Def1,ArgType2,Def2) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> class X; } \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2> class X; } \
     template <ArgType1 Arg1,ArgType2 Arg2> struct kCudaType<cxx::X<Arg1,Arg2> > \
     { using type_t = cuda::X<Arg1,Arg2 >; };
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(X,ArgType1,Def1,ArgType2,Def2,ArgType3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> > \
     { using type_t = cuda::X<Arg1, Arg2, CudaType_t<Arg3> >; };
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> > \
     { using type_t = cuda::X<Arg1,Arg2,Arg3 >; };

#else // VECGEOM_CUDA_VOLUME_SPECIALIZATION

  #define VECGEOM_DEVICE_DECLARE_NS_CONV(NS,X,Def)     \
     namespace cuda { namespace NS { class Def; } } \
     inline namespace cxx { namespace NS { class X; } } \
     template <> struct kCudaType<cxx::NS::X> { using type_t = cuda::NS::Def; };

  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(X,ArgType1,Def1,ArgType2,Def2) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2> class X; } \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2> class X; } \
     template <ArgType1 Arg1,ArgType2 Arg2> struct kCudaType<cxx::X<Arg1,Arg2> > \
     { using type_t = cuda::X<Def1, Def2 >; };
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(X,ArgType1,Def1,ArgType2,Def2,ArgType3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> > \
     { using type_t = cuda::X<Def2, Def2, CudaType_t<Arg3> >; };
  #define VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_3v(X,ArgType1,Def1,ArgType2,Def2,ArgType3,Def3) \
     namespace cuda { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     inline namespace cxx  { template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> class X; } \
     template <ArgType1 Arg1,ArgType2 Arg2,ArgType3 Arg3> struct kCudaType<cxx::X<Arg1,Arg2,Arg3> > \
     { using type_t = cuda::X<Def1,Def2,Def3 >; };

#endif // VECGEOM_CUDA_VOLUME_SPECIALIZATION

/* Instead of multiple macro, when we have auto expansion of Template pack we could use:
template <typename... Arguments>
struct kCudaType<cxx::BoxImplementation<Arguments...>  >
   { using type_t = typename cuda::BoxImplementation<CudaType_t<Arguments...> >; };
*/
#endif

#ifdef __INTEL_COMPILER
  // Compiling with icc
  #define VECGEOM_INTEL
  #define VECGEOM_INLINE inline
#else
  // Functionality of <mm_malloc.h> is automatically included in icc
  #include <mm_malloc.h>
  #if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__clang__) && !defined(__NO_INLINE__) && !defined( VECGEOM_NOINLINE )
    #define VECGEOM_INLINE inline __attribute__((always_inline))
    #ifndef VECGEOM_NVCC
      #define VECGEOM_ALIGNED __attribute__((aligned(64)))
    #endif
  #else
//#pragma message "forced inlining disabled"
  // Clang or forced inlining is disabled ( by falling back to compiler decision )
    #define VECGEOM_INLINE inline
    #ifndef VECGEOM_NVCC
      #define VECGEOM_ALIGNED
    #endif
  #endif
#endif

#ifndef NULL
  #define NULL nullptr
#endif

// Allow constexpr variables and functions if possible
#ifdef VECGEOM_STD_CXX11
  #define VECGEOM_CONSTEXPR constexpr
  #define VECGEOM_CONSTEXPR_RETURN constexpr
#else
  #define VECGEOM_CONSTEXPR const
  #define VECGEOM_CONSTEXPR_RETURN
#endif

// Qualifier(s) of global constants
#ifdef VECGEOM_NVCC_DEVICE
    // constexpr not supported on device in CUDA 6.5
    #define VECGEOM_GLOBAL static __constant__ const
    #define VECGEOM_CLASS_GLOBAL static const
#else
  #define VECGEOM_GLOBAL static constexpr
  #define VECGEOM_CLASS_GLOBAL static constexpr
#endif

namespace vecgeom {
#ifdef VECGEOM_FLOAT_PRECISION
typedef float Precision;
#else
typedef double Precision;
#endif
// namespace EInside {
// enum EInside {
//   kInside = 0,
//   kSurface = 1,
//   kOutside = 2
// };
// }
// typedef EInside::EInside Inside_t;
typedef int Inside_t;
}

//namespace vecgeom::cuda {
//typedef vecgeom::Precision Precision;
//typedef vecgeom::Inside_t Inside_t;
//}


//namespace std {
//  template< class T, class Deleter=std::default_delete<T> > class unique_ptr;
//}

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC
   using std::unique_ptr;

#else

   template <typename T>
   class unique_ptr {
      T *fValue;
   public:
     VECGEOM_CUDA_HEADER_BOTH
     unique_ptr(T *in) : fValue(in) {}

     VECGEOM_CUDA_HEADER_BOTH
     ~unique_ptr() { delete fValue; }

     VECGEOM_CUDA_HEADER_BOTH
     T* operator->() { return fValue; }
   };

   template <typename T>
   class unique_ptr<T[]> {
      T *fValue;
   public:
     VECGEOM_CUDA_HEADER_BOTH
     unique_ptr(T *in) : fValue(in) {}

     VECGEOM_CUDA_HEADER_BOTH
     ~unique_ptr() { delete [] fValue; }

     VECGEOM_CUDA_HEADER_BOTH
     T &operator[](size_t idx) { return fValue[idx]; }
   };
#endif

#ifdef __MIC__
VECGEOM_GLOBAL int kAlignmentBoundary = 64;
#else
VECGEOM_GLOBAL int kAlignmentBoundary = 32;
#endif
VECGEOM_GLOBAL Precision kPi = 3.14159265358979323846;
VECGEOM_GLOBAL Precision kTwoPi = 2.*kPi;
VECGEOM_GLOBAL Precision kTwoPiInv = 1./kTwoPi;
VECGEOM_GLOBAL Precision kDegToRad = kPi/180.;
VECGEOM_GLOBAL Precision kRadToDeg = 180./kPi;
VECGEOM_GLOBAL Precision kInfinity =
#ifndef VECGEOM_NVCC
    std::numeric_limits<Precision>::infinity();
#else
    INFINITY;
#endif
VECGEOM_GLOBAL Precision kEpsilon =
#ifndef VECGEOM_NVCC
    std::numeric_limits<Precision>::epsilon();
#elif VECGEOM_FLOAT_PRECISION
    FLT_EPSILON;
#else
    DBL_EPSILON;
#endif
VECGEOM_GLOBAL Precision kMinimum =
#ifndef VECGEOM_NVCC
    std::numeric_limits<Precision>::min();
#elif VECGEOM_FLOAT_PRECISION
    FLT_MIN;
#else
    DBL_MIN;
#endif
VECGEOM_GLOBAL Precision kMaximum =
#ifndef VECGEOM_NVCC
    std::numeric_limits<Precision>::max();
#elif VECGEOM_FLOAT_PRECISION
    FLT_MAX;
#else
    DBL_MAX;
#endif
VECGEOM_GLOBAL Precision kTiny = 1e-30;
VECGEOM_GLOBAL Precision kTolerance = 1e-12;
VECGEOM_GLOBAL Precision kRadTolerance = 1e-12;
VECGEOM_GLOBAL Precision kAngTolerance = 1e-12;

VECGEOM_GLOBAL Precision kHalfTolerance = 0.5*kTolerance;
VECGEOM_GLOBAL Precision kToleranceSquared = kTolerance*kTolerance;

namespace EInside {
VECGEOM_GLOBAL vecgeom::Inside_t kInside = 0;
VECGEOM_GLOBAL vecgeom::Inside_t kSurface = 1;
VECGEOM_GLOBAL vecgeom::Inside_t kOutside = 2;
}

// namespace EMatrix3DEntry {
// enum EMatrix3DEntry {
//   k00 = 0x001, k01 = 0x002, k02 = 0x004,
//   k10 = 0x008, k11 = 0x010, k12 = 0x020,
//   k20 = 0x040, k21 = 0x080, k22 = 0x100
// };
// }
// rotation::kGeneric
// translation::kGeneric

typedef int RotationCode;
typedef int TranslationCode;
namespace rotation {
enum RotationId { kGeneric = -1, kDiagonal = 0x111, kIdentity = 0x200 };
}
namespace translation {
enum TranslationId { kGeneric = -1, kIdentity = 0 };
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Assert(const bool condition, char const *const message) {
#ifndef VECGEOM_NVCC
  assert(condition && message);
#else
  if (!condition) printf("Assertion failed: %s", message);
#endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Assert(const bool condition) {
  Assert(condition, "");
}

namespace details {
   template <typename DataType, typename Target> struct UseIfSameType {
      VECGEOM_CUDA_HEADER_BOTH
      static Target const *Get(DataType*) { return nullptr; }
   };
   template <typename DataType> struct UseIfSameType<DataType,DataType> {
      VECGEOM_CUDA_HEADER_BOTH
      static DataType const *Get(DataType *ptr) { return ptr; }
   };
}

} } // End global namespace

#endif // VECGEOM_BASE_GLOBAL_H_
