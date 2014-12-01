/// \file ScalarShapeImplementationHelper.h

#ifndef VECGEOM_VOLUMES_SCALARSHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SCALARSHAPEIMPLEMENTATIONHELPER_H_

#include "base/Global.h"
#include "base/SOA3D.h"
#include "base/AOS3D.h"
#include "volumes/PlacedBox.h"

#include <algorithm>
namespace VECGEOM_NAMESPACE {

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
template <class Shape, typename Specialization>
class ScalarShapeImplementationHelper : public Shape {

public:

#ifndef VECGEOM_NVCC

  ScalarShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : Shape(label, logical_volume, transformation, boundingBox) {}

  ScalarShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : ScalarShapeImplementationHelper("", logical_volume, transformation,
                                  boundingBox) {}

#else // Compiling for CUDA

  __device__
  ScalarShapeImplementationHelper(LogicalVolume const *const logical_volume,
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

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &point,
                                        Vector3D<Precision> const &direction,
                                        const Precision stepMax = kInfinity) const {
     Transformation3D const* t = this->transformation();

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
        Shape::Normal( hitpoint, normal );
        // we could make this something like
        // convex = Shape::IsConvex;
        convex = true;
        return d;
  }
#endif

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

} // End global namespace

#endif // VECGEOM_VOLUMES_SCALARSHAPEIMPLEMENTATIONHELPER_H_
