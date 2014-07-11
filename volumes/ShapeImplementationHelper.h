/// \file ShapeImplementationHelper.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

#include "base/Global.h"

#include "backend/scalar/Backend.h"
#include "backend/Backend.h"
#include "base/SOA3D.h"
#include "base/AOS3D.h"
#include "volumes/PlacedBox.h"

#include <algorithm>

namespace VECGEOM_NAMESPACE {

template <class Shape, class Specialization>
class ShapeImplementationHelper : public Shape {

public:

#ifndef VECGEOM_NVCC

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : Shape(label, logical_volume, transformation, boundingBox) {}

  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : ShapeImplementationHelper("", logical_volume, transformation,
                                  boundingBox) {}

#else // Compiling for CUDA

  __device__
  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox,
                            const int id)
      : Shape(logical_volume, transformation, boundingBox, id) {}

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual Inside_t Inside(Vector3D<Precision> const &point) const {
    Inside_t output = EInside::kOutside;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
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
      *this->transformation(),
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
      *this->transformation(),
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
      *this->transformation(),
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

  /*
   * WARNING: Dummy method that pretends it supports the USolids interface
   * for DistanceToOut. Normal and convex are completely ignored.
   * VecGeom kernels do not yet support normals, so this was added for
   * USolids interoperability, in particular to aid in testing
   */
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  Vector3D<Precision> const &norm,
                                  bool &convex) {
      return DistanceToOut(point, direction);
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const {
    Precision output = kInfinity;
    Specialization::template SafetyToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->transformation(),
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

#ifdef VECGEOM_VC

  void ContainsTemplate(SOA3D<Precision> const &points,
                        bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      Vector3D<VcPrecision> localPoint;
      VcBool result(false);
      Specialization::template Contains<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        localPoint,
        result
      );
      for (unsigned j = 0; j < VcPrecision::Size; ++j) {
        output[j+i] = result[j];
      }
    }
  }

  void InsideTemplate(SOA3D<Precision> const &points,
                      Inside_t *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcInside result = EInside::kOutside;
      Specialization::template Inside<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      for (unsigned j = 0; j < VcPrecision::Size; ++j) output[j+i] = result[j];
    }
  }

  VECGEOM_INLINE
  void DistanceToInTemplate(SOA3D<Precision> const &points,
                            SOA3D<Precision> const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      Vector3D<VcPrecision> direction(
        VcPrecision(&directions.x(i)),
        VcPrecision(&directions.y(i)),
        VcPrecision(&directions.z(i))
      );
      VcPrecision stepMaxVc = VcPrecision(&stepMax[i]);
      VcPrecision result = kInfinity;
      Specialization::template DistanceToIn<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        direction,
        stepMaxVc,
        result
      );
      result.store(&output[i]);
    }
  }

#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")
  VECGEOM_INLINE
  void DistanceToInMinimizeTemplate(SOA3D<Precision> const &points,
                                    SOA3D<Precision> const &directions,
                                    int daughterid,
                                    Precision *const currentdistance,
                                    int *const nextdaughteridlist) const {
      for (int i = 0, iMax = points.size(); i < iMax; i += VcPrecision::Size) {
            Vector3D<VcPrecision> point(
              VcPrecision(&points.x(i)),
              VcPrecision(&points.y(i)),
              VcPrecision(&points.z(i))
            );
            Vector3D<VcPrecision> direction(
              VcPrecision(&directions.x(i)),
              VcPrecision(&directions.y(i)),
              VcPrecision(&directions.z(i))
            );
            // currentdistance is also estimate for stepmax
            VcPrecision stepMaxVc = VcPrecision(&currentdistance[i]);
            VcPrecision result = kInfinity;
            Specialization::template DistanceToIn<kVc>(
              *this->GetUnplacedVolume(),
              *this->transformation(),
              point,
              direction,
              stepMaxVc,
              result
            );
            // now we have distance and we can compare it to old distance step
            // and update it if necessary
            VcBool mask=result>stepMaxVc;
            result( mask ) = stepMaxVc;
            result.store(&currentdistance[i]);
            // currently do not know how to do this better (can do it when Vc offers long ints )
            for(int j=0;j<VcPrecision::Size;++j)
            {
                nextdaughteridlist[i+j]
                                   =( ! mask[j] )? daughterid : nextdaughteridlist[i+j];
            }
      }
  }
#pragma GCC pop_options
  VECGEOM_INLINE
  void DistanceToOutTemplate(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      Vector3D<VcPrecision> direction(
        VcPrecision(&directions.x(i)),
        VcPrecision(&directions.y(i)),
        VcPrecision(&directions.z(i))
      );
      VcPrecision stepMaxVc = VcPrecision(&stepMax[i]);
      VcPrecision result = kInfinity;
      Specialization::template DistanceToOut<kVc>(
        *this->GetUnplacedVolume(),
        point,
        direction,
        stepMaxVc,
        result
      );
      result.store(&output[i]);
    }
  }

  VECGEOM_INLINE
  void DistanceToOutTemplate(SOA3D<Precision> const &points,
                               SOA3D<Precision> const &directions,
                               Precision const *const stepMax,
                               Precision *const output,
                               int *const nodeindex ) const {
      for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
        Vector3D<VcPrecision> point(
          VcPrecision(&points.x(i)),
          VcPrecision(&points.y(i)),
          VcPrecision(&points.z(i))
        );
        Vector3D<VcPrecision> direction(
          VcPrecision(&directions.x(i)),
          VcPrecision(&directions.y(i)),
          VcPrecision(&directions.z(i))
        );
        VcPrecision stepMaxVc = VcPrecision(&stepMax[i]);
        VcPrecision result = kInfinity;
        Specialization::template DistanceToOut<kVc>(
          *this->GetUnplacedVolume(),
          point,
          direction,
          stepMaxVc,
          result
        );
        result.store(&output[i]);
        for (int j=0;j<VcPrecision::Size;++j) {
            // -1: physics step is longer than geometry
            // -2: particle may stay inside volume
            nodeindex[i+j] = ( result[j] < stepMaxVc[j] )? -1 : -2;
        }
      }
    }

  void SafetyToInTemplate(SOA3D<Precision> const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcPrecision result = kInfinity;
      Specialization::template SafetyToIn<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      result.store(&output[i]);
    }
  }

  void SafetyToInMinimizeTemplate(SOA3D<Precision> const &points,
                                  Precision *const safeties) const {
    for (int i = 0, iMax = points.size(); i < iMax; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcPrecision estimate = VcPrecision(&safeties[i]);
      VcPrecision result = kInfinity;
      Specialization::template SafetyToIn<kVc>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
        point,
        result
      );
      result(estimate < result) = estimate;
      result.store(&safeties[i]);
    }
  }

  void SafetyToOutTemplate(SOA3D<Precision> const &points,
                           Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcPrecision result = kInfinity;
      Specialization::template SafetyToOut<kVc>(
        *this->GetUnplacedVolume(),
        point,
        result
      );
      result.store(&output[i]);
    }
  }

  void SafetyToOutMinimizeTemplate(SOA3D<Precision> const &points,
                                   Precision *const safeties) const {
    for (int i = 0, iMax = points.size(); i < iMax; i += VcPrecision::Size) {
      Vector3D<VcPrecision> point(
        VcPrecision(&points.x(i)),
        VcPrecision(&points.y(i)),
        VcPrecision(&points.z(i))
      );
      VcPrecision estimate = VcPrecision(&safeties[i]);
      VcPrecision result = kInfinity;
      Specialization::template SafetyToOut<kVc>(
        *this->GetUnplacedVolume(),
        point,
        result
      );
      result(estimate < result) = estimate;
      result.store(&safeties[i]);
    }
  }

#else // Scalar default

  template <class Container_t>
  void ContainsTemplate(Container_t const &points, bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; ++i) {
      Vector3D<Precision> localPoint;
      Specialization::template Contains<kScalar>(
        *this->GetUnplacedVolume(),
        *this->transformation(),
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
        *this->transformation(),
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
        *this->transformation(),
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
          *this->transformation(),
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
        *this->transformation(),
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
        *this->transformation(),
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

#endif

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

}; // End class ShapeImplementationHelper

} // End global namespace

#endif // VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
