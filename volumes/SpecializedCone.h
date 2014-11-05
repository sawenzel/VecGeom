/*
 * SpecializedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCONE_H_

#include "base/Global.h"

#include "volumes/kernel/ConeImplementation.h"
#include "volumes/PlacedCone.h"
//#include "volumes/ShapeImplementationHelper.h"
#include "base/SOA3D.h"

#include <stdio.h>

namespace VECGEOM_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT,
          typename ConeType>
class SpecializedCone : public PlacedCone

   // : public ShapeImplementationHelper<PlacedCone,
   //                                    ConeImplementation<
   //                                        transCodeT, rotCodeT, ConeType> >
{

//  typedef ShapeImplementationHelper<PlacedCone,
//                                    ConeImplementation<
//                                        transCodeT, rotCodeT, ConeType> > Helper;
    typedef ConeImplementation<transCodeT, rotCodeT, ConeType> Specialization;

public:

#ifndef VECGEOM_NVCC

  SpecializedCone(char const *const label,
                  LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : PlacedCone(label, logical_volume, transformation, NULL) {}

  SpecializedCone(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation)
      : SpecializedCone("", logical_volume, transformation) {}

  SpecializedCone(char const *const label,
                   const Precision rmin1,
                   const Precision rmax1,
                   const Precision rmin2,
                   const Precision rmax2,
                   const Precision dz,
                   const Precision sphi=0,
                   const Precision dphi=kTwoPi)
        : SpecializedCone(label,
                new LogicalVolume(new UnplacedCone(rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi )),
                &Transformation3D::kIdentity) {}

#else

  __device__
  SpecializedCone(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox, const int id)
      : PlacedCone(logical_volume, transformation, boundingBox, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

  VECGEOM_CUDA_HEADER_BOTH
    virtual ~SpecializedCone() {}


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
     VECGEOM_INLINE
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
     VECGEOM_INLINE
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
     VECGEOM_INLINE
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
     VECGEOM_INLINE
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
     VECGEOM_INLINE
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
        Normal( hitpoint, normal );
        // we could make this something like
        // convex = Shape::IsConvex;
        convex = true;
        return d;
  }
#endif

};

typedef SpecializedCone<translation::kGeneric, rotation::kGeneric, ConeTypes::UniversalCone>
    SimpleCone;
typedef SpecializedCone<translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>
    SimpleUnplacedCone;


template <TranslationCode transCodeT, RotationCode rotCodeT, typename ConeType>
void SpecializedCone<transCodeT, rotCodeT, ConeType>::PrintType() const {
  printf("SpecializedCone<%i, %i>", transCodeT, rotCodeT);
}

} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
