#ifndef VECGEOM_GLOBAL_H_
#define VECGEOM_GLOBAL_H_

#if (defined(__CUDACC__) || defined(__NVCC__))
  #define VECGEOM_NVCC
  #define VECGEOM_CUDA_HEADER_DEVICE __device__
  #define VECGEOM_CUDA_HEADER_HOST __host__
  #define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
  #define VECGEOM_CUDA_HEADER_GLOBAL __global__
#else // Not compiling with NVCC
  #define VECGEOM_VC
  #define VECGEOM_CUDA_HEADER_DEVICE
  #define VECGEOM_CUDA_HEADER_HOST
  #define VECGEOM_CUDA_HEADER_BOTH
  #define VECGEOM_CUDA_HEADER_GLOBAL
  #include <Vc/Vc>
#endif

#ifdef VECGEOM_VC
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

#endif // VECGEOM_GLOBAL_H_