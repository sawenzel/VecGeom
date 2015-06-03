/// \file ShapeImplementationHelper.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "backend/scalar/Backend.h"
#include "backend/Backend.h"
#include "base/SOA3D.h"
#include "base/AOS3D.h"
#include "volumes/PlacedBox.h"

#include <algorithm>

#ifdef VECGEOM_DISTANCE_DEBUG
#include "volumes/utilities/ResultComparator.h"
#endif

#ifdef VECGEOM_VC
#define VECGEOM_BACKEND_TYPE         kVc
#define VECGEOM_BACKEND_PRECISION    VcPrecision
#define VECGEOM_BACKEND_BOOL         VcBool
#define VECGEOM_BACKEND_INSIDE       kVc::inside_v
#elif MIC_SIDE
#define VECGEOM_BACKEND_TYPE         kMic
#define VECGEOM_BACKEND_PRECISION    MicPrecision
#define VECGEOM_BACKEND_BOOL         MicBool
#define VECGEOM_BACKEND_INSIDE       kMic::inside_v
#elif VECGEOM_SCALAR
#define VECGEOM_BACKEND_TYPE         kScalar
#define VECGEOM_BACKEND_PRECISION(P) (*(P))
#define VECGEOM_BACKEND_BOOL         ScalarBool
#define VECGEOM_BACKEND_INSIDE       kScalar::inside_v
#endif

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(ShapeImplementationHelper,class)

inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Specialization>
class ShapeImplementationHelper : public Specialization::PlacedShape_t {

using PlacedShape_t = typename Specialization::PlacedShape_t;
using UnplacedShape_t = typename Specialization::UnplacedShape_t;
using Helper_t = ShapeImplementationHelper<Specialization>;
using Implementation_t = Specialization;

public:

#ifndef VECGEOM_NVCC

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox) {}

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
     : ShapeImplementationHelper(label, logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this)) {}

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume *const logical_volume,
                            Transformation3D const*const transformation,
                            PlacedBox const*const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox) {}

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume *const logical_volume,
                            Transformation3D const*const transformation)
      : ShapeImplementationHelper(label, logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this)) {}

  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : ShapeImplementationHelper("", logical_volume, transformation,
                                  boundingBox) {}


  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : ShapeImplementationHelper("", logical_volume, transformation) {}

  // this constructor mimics the constructor from the Unplaced solid
  // it ensures that placed volumes can be constructed just like ordinary Geant4/ROOT/USolids solids
  template <typename... ArgTypes>
  ShapeImplementationHelper(char const *const label, ArgTypes... params)
      : ShapeImplementationHelper(label,
                                  new LogicalVolume(new UnplacedShape_t(params...)),
                                  &Transformation3D::kIdentity) {}


#else // Compiling for CUDA

  __device__
  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox,
                            const int id)
      : PlacedShape_t(logical_volume, transformation, boundingBox, id) {}


  __device__
  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            const int id)
      : PlacedShape_t(logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this), id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const { Specialization::PrintType(); }

#ifdef VECGEOM_CUDA_INTERFACE

  virtual size_t DeviceSizeOf() const { return DevicePtr<CudaType_t<Helper_t> >::SizeOf(); }

  DevicePtr<cuda::VPlacedVolume> CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const logical_volume,
    DevicePtr<cuda::Transformation3D> const transform,
    DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const
 {
     DevicePtr<CudaType_t<Helper_t> > gpu_ptr(in_gpu_ptr);
     gpu_ptr.Construct(logical_volume, transform, DevicePtr<cuda::PlacedBox>(), this->id());
     CudaAssertError();
     // Need to go via the void* because the regular c++ compilation
     // does not actually see the declaration for the cuda version
     // (and thus can not determine the inheritance).
     return DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr);
 }

 DevicePtr<cuda::VPlacedVolume> CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const logical_volume,
    DevicePtr<cuda::Transformation3D> const transform) const
 {
     DevicePtr<CudaType_t<Helper_t> > gpu_ptr;
     gpu_ptr.Allocate();
     return CopyToGpu(logical_volume,transform,
                      DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr));
 }

#endif // VECGEOM_CUDA_INTERFACE

  VECGEOM_CUDA_HEADER_BOTH
  virtual EnumInside Inside(Vector3D<Precision> const &point) const {
    Inside_t output = EInside::kOutside;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      output
    );
    // we need to convert the output from int to an enum
    // necessary because Inside kernels operate on ints to be able to vectorize operations
    return (EnumInside) output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point) const {
    bool output = false;
    Vector3D<Precision> localPoint;
    Specialization::template Contains<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      localPoint,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point,
                        Vector3D<Precision> &localPoint) const {
    bool output = false;
    Specialization::template Contains<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      localPoint,
      output
    );


#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareUnplacedContains( this, output, localPoint );
#endif

    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const {
    bool output = false;
    Specialization::template UnplacedContains<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareUnplacedContains( this, output, point );
#endif

    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &point,
                                 Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfinity) const {
#ifndef VECGEOM_NVCC
      assert( direction.IsNormalized() && " direction not normalized in call to  DistanceToIn " );
#endif
      Precision output = kInfinity;
    Specialization::template DistanceToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      direction,
      stepMax,
      output
    );

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToIn( this, output, point, direction, stepMax );
#endif

    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfinity) const {
#ifndef VECGEOM_NVCC
      assert( direction.IsNormalized() && " direction not normalized in call to  DistanceToOut " );
#endif
    Precision output = kInfinity;
    Specialization::template DistanceToOut<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      direction,
      stepMax,
      output
    );

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToOut( this, output, point, direction, stepMax );
#endif


    return output;
  }


  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &point,
                                        Vector3D<Precision> const &direction,
                                        const Precision stepMax = kInfinity) const {
#ifndef VECGEOM_NVCC
      assert( direction.IsNormalized() && " direction not normalized in call to  PlacedDistanceToOut " );
#endif
     Precision output = kInfinity;
     Transformation3D const * t = this->GetTransformation();
     Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(),
        t->Transform< Specialization::transC, Specialization::rotC, Precision>(point),
        t->TransformDirection< Specialization::rotC, Precision>(direction),
        stepMax,
        output
      );

  #ifdef VECGEOM_DISTANCE_DEBUG
      DistanceComparator::CompareDistanceToOut(
              this,
              output,
              this->GetTransformation()->Transform(point),
              this->GetTransformation()->TransformDirection(direction),
              stepMax );
  #endif
      return output;
    }



#ifdef VECGEOM_USOLIDS
  /*
   * WARNING: Trivial implementation for standard USolids interface
   * for DistanceToOut. The value for convex might be wrong
   */
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  Vector3D<Precision> &normal,
                                  bool &convex, Precision step = kInfinity ) const {
      double d = DistanceToOut(point, direction, step );
        Vector3D<double> hitpoint = point + d*direction;
        PlacedShape_t::Normal( hitpoint, normal );
        // we could make this something like
        // convex = PlacedShape_t::IsConvex;
        convex = true;
        return d;
  }
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const {
    Precision output = kInfinity;
    Specialization::template SafetyToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const {
    Precision output = kInfinity;
    Specialization::template SafetyToOut<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );
    return output;
  }

  virtual void Contains(SOA3D<Precision> const &points,
                        bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> localPoint;
      VECGEOM_BACKEND_BOOL result(false);
      Specialization::template Contains<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        localPoint,
        result
      );
#ifdef VECGEOM_VC
      for (unsigned j = 0; j < kVectorSize; ++j) {
        output[j+i] = result[j];
      }
#elif VECGEOM_SCALAR
      output[i] = result;
#endif
    }
  }

  virtual void Inside(SOA3D<Precision> const &points,
                      Inside_t *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_INSIDE result = VECGEOM_BACKEND_INSIDE(EInside::kOutside);
      Specialization::template Inside<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        result
      );
#ifdef VECGEOM_VC
      for (unsigned j = 0; j < kVectorSize; ++j) {
        output[j+i] = result[j];
      }
#elif MIC_SIDE
      for (unsigned j = 0; j < kVectorSize; ++j) {
        output[j+i] = result[j];
      }
#elif VECGEOM_SCALAR
      output[i] = result;
#endif
    }
  }

  virtual void DistanceToIn(SOA3D<Precision> const &points,
                            SOA3D<Precision> const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
#ifdef MIC_SIDE
  #pragma omp parallel for
#endif
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        direction,
        stepMaxBackend,
        result
      );
#ifdef VECGEOM_VC
      result.store(&output[i]);
#elif MIC_SIDE
      _mm512_store_pd(output+i,result);
#elif VECGEOM_SCALAR
      output[i] = result;
#endif
    }
  }

#if !defined(__clang__) && !defined(VECGEOM_INTEL)
  #pragma GCC push_options
  #pragma GCC optimize ("unroll-loops")
#endif
  virtual void DistanceToInMinimize(SOA3D<Precision> const &points,
                                    SOA3D<Precision> const &directions,
                                    int daughterId,
                                    Precision *const currentDistance,
                                    int *const nextDaughterIdList) const {
    for (int i = 0, iMax = points.size(); i < iMax; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      // currentdistance is also estimate for stepmax
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&currentDistance[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        direction,
        stepMaxBackend,
        result
      );
#ifdef VECGEOM_VC
      // now we have distance and we can compare it to old distance step
      // and update it if necessary
      VcBool mask=result>stepMaxBackend;
      result( mask ) = stepMaxBackend;
      result.store(&currentDistance[i]);
      // currently do not know how to do this better (can do it when Vc offers long ints )
#ifdef VECGEOM_INTEL
#pragma unroll
#endif
      for(unsigned int j=0;j<kVectorSize;++j) {
        nextDaughterIdList[i+j]=( ! mask[j] )? daughterId : nextDaughterIdList[i+j];
      }
#elif MIC_SIDE
      assert("Not implemented yet.");
#elif VECGEOM_SCALAR
      if (result < currentDistance[i]) {
        currentDistance[i] = result;
        nextDaughterIdList[i] = daughterId;
      }
#endif
    }
  }
#if !defined(__clang__) && !defined(VECGEOM_INTEL)
#pragma GCC pop_options
#endif

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        direction,
        stepMaxBackend,
        result
      );
#ifdef VECGEOM_VC
      result.store(&output[i]);
#elif MIC_SIDE
      _mm512_store_pd(output+i,result);
#elif VECGEOM_SCALAR
      output[i] = result;
#endif
    }
  }

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output,
                             int *const nextNodeIndex) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        direction,
        stepMaxBackend,
        result
      );
#ifdef VECGEOM_VC
      result.store(&output[i]);
      for (unsigned int j=0;j<kVectorSize;++j) {
        // -1: physics step is longer than geometry
        // -2: particle may stay inside volume
        nextNodeIndex[i+j] = ( result[j] < stepMaxBackend[j] )? -1 : -2;
      }
#elif VECGEOM_SCALAR
      nextNodeIndex[i] = (output[i] < stepMaxBackend) ? -1 : -2;
#endif
    }
  }

  virtual void SafetyToIn(SOA3D<Precision> const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        result
      );
#ifdef VECGEOM_VC
      result.store(&output[i]);
#elif MIC_SIDE
      _mm512_store_pd(output+i,result);
#elif VECGEOM_SCALAR
      output[i] = result;
#endif
    }
  }

  virtual void SafetyToInMinimize(SOA3D<Precision> const &points,
                                  Precision *const safeties) const {
    for (int i = 0, iMax = points.size(); i < iMax; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        result
      );
#ifdef VECGEOM_VC
      VECGEOM_BACKEND_TYPE::precision_v estimate = VECGEOM_BACKEND_PRECISION(&safeties[i]);
      result(estimate < result) = estimate;
      result.store(&safeties[i]);
#elif MIC_SIDE
      assert("Not implemented yet.");
#elif VECGEOM_SCALAR
      safeties[i] = (result < safeties[i]) ? result : safeties[i];
#endif
    }
  }

  virtual void SafetyToOut(SOA3D<Precision> const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        result
      );
#ifdef VECGEOM_VC
      result.store(&output[i]);
#elif MIC_SIDE
      _mm512_store_pd(output+i,result);
#elif VECGEOM_SCALAR
      output[i] = result;
#endif
    }
  }

  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points,
                                   Precision *const safeties) const {
    for (int i = 0, iMax = points.size(); i < iMax; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      // The scalar implementation assiged 0 to result... ?
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        result
      );
#ifdef VECGEOM_VC
      VECGEOM_BACKEND_TYPE::precision_v estimate = VECGEOM_BACKEND_PRECISION(&safeties[i]);
      result(estimate < result) = estimate;
      result.store(&safeties[i]);
#elif MIC_SIDE
      assert("Not implemented yet.");
#elif VECGEOM_SCALAR
      safeties[i] = (result < safeties[i]) ? result : safeties[i];
#endif
    }
  }

}; // End class ShapeImplementationHelper

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
