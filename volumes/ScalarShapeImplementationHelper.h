/// \file ScalarShapeImplementationHelper.h

#ifndef VECGEOM_VOLUMES_SCALARSHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SCALARSHAPEIMPLEMENTATIONHELPER_H_

#include "base/Global.h"
#include "base/SOA3D.h"
#include "base/AOS3D.h"
#include "volumes/PlacedBox.h"

#include <algorithm>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(ScalarShapeImplementationHelper,class)

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A helper class implementing "repetetive" dispatching of high level interfaces to
 * actual implementations
 *
 * In contrast to the ordinary ShapeImplementationHelper,
 * the ScalarShapeImplementatioHelper
 * does not explicitely use vectorization; Hence the multi-particle interfaces
 * are dispatched to loops over scalar implementations
 *
 */
template <typename Specialization>
class ScalarShapeImplementationHelper : public Specialization::PlacedShape_t {

using PlacedShape_t = typename Specialization::PlacedShape_t;
using UnplacedShape_t = typename Specialization::UnplacedShape_t;
using Helper_t = ScalarShapeImplementationHelper<Specialization>;
using Implementation_t = Specialization;

public:

#ifndef VECGEOM_NVCC

  ScalarShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox) {}

  ScalarShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : ScalarShapeImplementationHelper(label, logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this)) {}

  ScalarShapeImplementationHelper(char const *const label,
                            LogicalVolume *const logical_volume,
                            Transformation3D const*const transformation,
                            PlacedBox const*const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox) {}

  ScalarShapeImplementationHelper(char const *const label,
                            LogicalVolume *const logical_volume,
                            Transformation3D const*const transformation)
      : ScalarShapeImplementationHelper(label, logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this)) {}

  ScalarShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : ScalarShapeImplementationHelper("", logical_volume, transformation,
                                  boundingBox) {}

  ScalarShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : ScalarShapeImplementationHelper("", logical_volume, transformation) {}

  template <typename... ArgTypes>
  ScalarShapeImplementationHelper(char const *const label, ArgTypes... params)
      : ScalarShapeImplementationHelper(label, 
                                  new LogicalVolume(new UnplacedShape_t(params...)),
                                  &Transformation3D::kIdentity) {}

#else // Compiling for CUDA

  __device__
  ScalarShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox,
                            const int id)
      : PlacedShape_t(logical_volume, transformation, boundingBox, id) {}


  __device__
  ScalarShapeImplementationHelper(LogicalVolume const *const logical_volume,
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
  virtual Inside_t Inside(Vector3D<Precision> const &point) const {
    Inside_t output = EInside::kOutside;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      output
    );
    return output;
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
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &point,
                                 Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfinity) const {
    Precision output = kInfinity;
    Specialization::template DistanceToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      direction,
      stepMax,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfinity) const {
    Precision output = kInfinity;
    Specialization::template DistanceToOut<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      direction,
      stepMax,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &point,
                                        Vector3D<Precision> const &direction,
                                        const Precision stepMax = kInfinity) const {
     Transformation3D const* t = this->GetTransformation();

     Precision output = kInfinity;
     Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(),
        t->Transform< Specialization::transC, Specialization::rotC, Precision>(point),
        t->TransformDirection< Specialization::rotC, Precision>(direction),
        stepMax,
        output);

    #ifdef VECGEOM_DISTANCE_DEBUG
        DistanceComparator::CompareDistanceToOut( this, output, point, direction, stepMax );
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

  template <class Container_t>
  void ContainsTemplate(Container_t const &points, bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Vector3D<Precision> localPoint;
      Specialization::template Contains<kScalar>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        points[i],
        localPoint,
        output[i]
      );
    }
  }

  template <class Container_t>
  void InsideTemplate(Container_t const &points,
                      Inside_t *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Inside_t result = EInside::kOutside;
      Specialization::template Inside<kScalar>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        points[i],
        result
      );
      output[i] = result;
    }
  }

  template <class Container_t>
  void DistanceToInTemplate(Container_t const &points,
                            Container_t const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template DistanceToIn<kScalar>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        points[i],
        directions[i],
        stepMax[i],
        output[i]
      );
    }
  }

  VECGEOM_INLINE
  void DistanceToInMinimizeTemplate(SOA3D<Precision> const &points,
                                    SOA3D<Precision> const &directions,
                                    int daughterId,
                                    Precision *const currentDistance,
                                    int *const nextDaughterIdList) const {
      for (int i = 0, iMax = points.size(); i < iMax; ++i) {
        Precision stepMax = currentDistance[i];
        Precision result = kInfinity;
        Specialization::template DistanceToIn<kScalar>(
          *this->GetUnplacedVolume(),
          *this->GetTransformation(),
          points[i],
          directions[i],
          stepMax,
          result
        );
        if (result < currentDistance[i]) {
          currentDistance[i] = result;
          nextDaughterIdList[i] = daughterId;
        }
    }
  }

  template <class Container_t>
  void DistanceToOutTemplate(Container_t const &points,
                             Container_t const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(),
        points[i],
        directions[i],
        stepMax[i],
        output[i]
      );
    }
  }

  VECGEOM_INLINE
  void DistanceToOutTemplate(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output,
                             int *const nodeIndex) const {
    for (int i = 0, iMax = points.size(); i < iMax; ++i) {
      Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(),
        points[i],
        directions[i],
        stepMax[i],
        output[i]
      );
      nodeIndex[i] = (output[i] < stepMax[i]) ? -1 : -2;
    }
  }

  template <class Container_t>
  void SafetyToInTemplate(Container_t const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template SafetyToIn<kScalar>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        points[i],
        output[i]
      );
    }
  }

  template <class Container_t>
  void SafetyToInMinimizeTemplate(Container_t const &points,
                                  Precision *const output) const {
    for (int i = 0, iMax = points.size(); i < iMax; ++i) {
      Precision result = 0;
      Specialization::template SafetyToIn<kScalar>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        points[i],
        result
      );
      output[i] = (result < output[i]) ? result : output[i];
    }
  }

  template <class Container_t>
  void SafetyToOutTemplate(Container_t const &points,
                           Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Specialization::template SafetyToOut<kScalar>(
        *this->GetUnplacedVolume(),
        points[i],
        output[i]
      );
    }
  }

  template <class Container_t>
  void SafetyToOutMinimizeTemplate(Container_t const &points,
                                   Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Precision result = 0;
      Specialization::template SafetyToOut<kScalar>(
        *this->GetUnplacedVolume(),
        points[i],
        result
      );
      output[i] = (result < output[i]) ? result : output[i];
    }
  }

  // virtual void Contains(AOS3D<Precision> const &points,
  //                       bool *const output) const {
  //   ContainsTemplate(points, output);
  // }

  virtual void Contains(SOA3D<Precision> const &points,
                        bool *const output) const {
    ContainsTemplate(points, output);
  }

  // virtual void Inside(AOS3D<Precision> const &points,
  //                     Inside_t *const output) const {
  //   InsideTemplate(points, output);
  // }

  virtual void Inside(SOA3D<Precision> const &points,
                      Inside_t *const output) const {
    InsideTemplate(points, output);
  }

  // virtual void DistanceToIn(AOS3D<Precision> const &points,
  //                           AOS3D<Precision> const &directions,
  //                           Precision const *const stepMax,
  //                           Precision *const output) const {
  //   DistanceToInTemplate(points, directions, stepMax, output);
  // }

  virtual void DistanceToIn(SOA3D<Precision> const &points,
                            SOA3D<Precision> const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    DistanceToInTemplate(points, directions, stepMax, output);
  }


  virtual void DistanceToInMinimize(SOA3D<Precision> const &points,
                                    SOA3D<Precision> const &directions,
                                    int daughterindex,
                                    Precision *const output,
                                    int *const nextnodeids) const {
      DistanceToInMinimizeTemplate(points, directions, daughterindex, output, nextnodeids);
  }

  // virtual void DistanceToOut(AOS3D<Precision> const &points,
  //                            AOS3D<Precision> const &directions,
  //                            Precision const *const stepMax,
  //                            Precision *const output) const {
  //   DistanceToOutTemplate(points, directions, stepMax, output);
  // }

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    DistanceToOutTemplate(points, directions, stepMax, output);
  }

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output,
                             int *const nextNodeIndex) const {
    DistanceToOutTemplate(points, directions, stepMax, output, nextNodeIndex);
  }

  virtual void SafetyToIn(SOA3D<Precision> const &points,
                          Precision *const output) const {
    SafetyToInTemplate(points, output);
  }

  // virtual void SafetyToIn(AOS3D<Precision> const &points,
  //                         Precision *const output) const {
  //   SafetyToInTemplate(points, output);
  // }

  virtual void SafetyToInMinimize(SOA3D<Precision> const &points,
                                  Precision *const safeties) const {
    SafetyToInMinimizeTemplate(points, safeties);
  }

  virtual void SafetyToOut(SOA3D<Precision> const &points,
                          Precision *const output) const {
    SafetyToOutTemplate(points, output);
  }

  // virtual void SafetyToOut(AOS3D<Precision> const &points,
  //                         Precision *const output) const {
  //   SafetyToOutTemplate(points, output);
  // }

  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points,
                                   Precision *const safeties) const {
    SafetyToOutMinimizeTemplate(points, safeties);
  }

}; // End class ScalarShapeImplementationHelper

} // End Impl namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_SCALARSHAPEIMPLEMENTATIONHELPER_H_
