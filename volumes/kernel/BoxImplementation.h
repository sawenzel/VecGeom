/// @file BoxImplementation.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "backend/Backend.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedBox.h"
#include "volumes/kernel/GenericKernels.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(BoxImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)


inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox;
class UnplacedBox;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct BoxImplementation {

#ifdef OFFLOAD_MODE
  VECGEOM_GLOBAL int transC = transCodeT;
  VECGEOM_GLOBAL int rotC   = rotCodeT;
#else
  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;
#endif

  using PlacedShape_t = PlacedBox;
  using UnplacedShape_t = UnplacedBox;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedBox<%i, %i>", transCodeT, rotCodeT);
  }

  template<typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedBox const &box,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedBox const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedBox const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend, bool ForInside>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(Vector3D<Precision> const &,
          Vector3D<typename Backend::precision_v> const &,
          typename Backend::bool_v &,
          typename Backend::bool_v &);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedBox const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedBox const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedBox const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedBox const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      Vector3D<Precision> const &boxDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);



  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      Vector3D<Precision> const &boxDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      Vector3D<Precision> const &dimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

 // template <class Backend>
 // static void Normal( Vector3D<Precision> const &dimensions,
 //         Vector3D<typename Backend::precision_v> const &point,
 //        Vector3D<typename Backend::precision_v> &normal,
 //         Vector3D<typename Backend::precision_v> &valid )

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void NormalKernel(
       UnplacedBox const &,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid );


  // an algorithm to test for intersection ( could be faster than DistanceToIn )
  // actually this also calculated the distance at the same time ( in tmin )
  // template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static bool Intersect( Vector3D<Precision> const * corners,
          Vector3D<Precision> const &point,
          Vector3D<Precision> const &ray,
          Precision t0,
          Precision t1){
    // intersection algorithm 1 ( Amy Williams )

    Precision tmin, tmax, tymin, tymax, tzmin, tzmax;

    // IF THERE IS A STEPMAX; COULD ALSO CHECK SAFETIES

    double inverserayx = 1./ray[0];
    double inverserayy = 1./ray[1];

    // TODO: we should promote this to handle multiple boxes
    int sign[3];
    sign[0] = inverserayx < 0;
    sign[1] = inverserayy < 0;


    tmin =  (corners[sign[0]].x()   -point.x())*inverserayx;
    tmax =  (corners[1-sign[0]].x() -point.x())*inverserayx;
    tymin = (corners[sign[1]].y()   -point.y())*inverserayy;
    tymax = (corners[1-sign[1]].y() -point.y())*inverserayy;

    if((tmin > tymax) || (tymin > tmax))
        return false;

    double inverserayz = 1./ray.z();
    sign[2] = inverserayz < 0;

    if(tymin > tmin)
        tmin = tymin;
    if(tymax < tmax)
        tmax = tymax;

    tzmin = (corners[sign[2]].z()   -point.z())*inverserayz;
    tzmax = (corners[1-sign[2]].z() -point.z())*inverserayz;

    if((tmin > tzmax) || (tzmin > tmax))
        return false;
    if((tzmin > tmin))
        tmin = tzmin;
    if(tzmax < tmax)
        tmax = tzmax;
    //return ((tmin < t1) && (tmax > t0));
   // std::cerr << "tmin " << tmin << " tmax " << tmax << "\n";
    return true;
  }


    // an algorithm to test for intersection ( could be faster than DistanceToIn )
    // actually this also calculated the distance at the same time ( in tmin )
    template <int signx, int signy, int signz>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    //__attribute__((noinline))
    static Precision IntersectCached( Vector3D<Precision> const * corners,
            Vector3D<Precision> const &point,
            Vector3D<Precision> const &inverseray,
            Precision t0,
            Precision t1 ){
      // intersection algorithm 1 ( Amy Williams )

      // NOTE THE FASTEST VERSION IS STILL THE ORIGINAL IMPLEMENTATION

      Precision tmin, tmax, tymin, tymax, tzmin, tzmax;

      // TODO: we should promote this to handle multiple boxes
      // observation: we always compute sign and 1-sign; so we could do the assignment
      // to tmin and tmax in a masked assignment thereafter
      tmin =  (corners[signx].x()   -point.x())*inverseray.x();
      tmax =  (corners[1-signx].x() -point.x())*inverseray.x();
      tymin = (corners[signy].y()   -point.y())*inverseray.y();
      tymax = (corners[1-signy].y() -point.y())*inverseray.y();
      if((tmin > tymax) || (tymin > tmax))
          return vecgeom::kInfinity;

      if(tymin > tmin)
          tmin = tymin;
      if(tymax < tmax)
          tmax = tymax;

      tzmin = (corners[signz].z()   -point.z())*inverseray.z();
      tzmax = (corners[1-signz].z() -point.z())*inverseray.z();

      if((tmin > tzmax) || (tzmin > tmax))
          return vecgeom::kInfinity; // false
      if((tzmin > tmin))
          tmin = tzmin;
      if(tzmax < tmax)
          tmax = tzmax;

      if( ! ((tmin < t1) && (tmax > t0)) )
          return vecgeom::kInfinity;
      return tmin;
    }

    // an algorithm to test for intersection ( could be faster than DistanceToIn )
        // actually this also calculated the distance at the same time ( in tmin )
        template <typename Backend, int signx, int signy, int signz>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::precision_v IntersectCachedKernel(
                Vector3D<typename Backend::precision_v > const * corners,
                Vector3D<Precision> const &point,
                Vector3D<Precision> const &inverseray,
                Precision t0,
                Precision t1 ){

          typedef typename Backend::precision_v Float_t;
          typedef typename Backend::bool_v Bool_t;

          Float_t tmin  = (corners[signx].x()   - point.x())*inverseray.x();
          Float_t tmax  = (corners[1-signx].x() - point.x())*inverseray.x();
          Float_t tymin = (corners[signy].y()   - point.y())*inverseray.y();
          Float_t tymax = (corners[1-signy].y() - point.y())*inverseray.y();

          // do we need this condition ?
          Bool_t done = (tmin > tymax) || (tymin > tmax);
          if( IsFull(done) ) return vecgeom::kInfinity;
          // if((tmin > tymax) || (tymin > tmax))
          //     return vecgeom::kInfinity;

          // Not sure if this has to be maskedassignments
          tmin = Max(tmin, tymin);
          tmax = Min(tmax, tymax);

          Float_t tzmin = (corners[signz].z()   - point.z())*inverseray.z();
          Float_t tzmax = (corners[1-signz].z() - point.z())*inverseray.z();

          done |= (tmin > tzmax) || (tzmin > tmax);
         // if((tmin > tzmax) || (tzmin > tmax))
         //     return vecgeom::kInfinity; // false
          if( IsFull(done) ) return vecgeom::kInfinity;

          // not sure if this has to be maskedassignments
          tmin = Max(tmin, tzmin);
          tmax = Min(tmax, tzmax);

          done |= ! ((tmin < t1) && (tmax > t0));
         // if( ! ((tmin < t1) && (tmax > t0)) )
         //     return vecgeom::kInfinity;
          MaskedAssign(done, vecgeom::kInfinity, &tmin);
          return tmin;
        }


        // an algorithm to test for intersection ( could be faster than DistanceToIn )
        // actually this also calculated the distance at the same time ( in tmin )
  template <typename Backend, typename basep = Precision>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v IntersectCachedKernel2(
          Vector3D<typename Backend::precision_v > const * corners,
          Vector3D<basep> const &point,
          Vector3D<basep> const &inverseray,
          int signx, int signy, int signz,
          basep t0,
          basep t1 ){

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t tmin  = (corners[signx].x()   - point.x())*inverseray.x();
    Float_t tymax = (corners[1-signy].y() - point.y())*inverseray.y();
    Bool_t done = tmin > tymax;
    if( IsFull(done) ) return (basep) vecgeom::kInfinity;

    Float_t tmax  = (corners[1-signx].x() - point.x())*inverseray.x();
    Float_t tymin = (corners[signy].y()   - point.y())*inverseray.y();

    // do we need this condition ?
    done |= (tymin > tmax);
    if( IsFull(done) ) return (basep) vecgeom::kInfinity;

    // if((tmin > tymax) || (tymin > tmax))
    //     return vecgeom::kInfinity;

    // Not sure if this has to be maskedassignments
    tmin = Max(tmin, tymin);
    tmax = Min(tmax, tymax);

    Float_t tzmin = (corners[signz].z()   - point.z())*inverseray.z();
    Float_t tzmax = (corners[1-signz].z() - point.z())*inverseray.z();

    done |= (tmin > tzmax) || (tzmin > tmax);
   // if((tmin > tzmax) || (tzmin > tmax))
   //     return vecgeom::kInfinity; // false
   if( IsFull(done) ) return (basep) vecgeom::kInfinity;

    // not sure if this has to be maskedassignments
    tmin = Max(tmin, tzmin);
    tmax = Min(tmax, tzmax);

    done |= ! ((tmin < t1) && (tmax > t0));
   // if( ! ((tmin < t1) && (tmax > t0)) )
   //     return vecgeom::kInfinity;
    MaskedAssign(done, Float_t((basep) vecgeom::kInfinity), &tmin);
    return tmin;
  }


    // an algorithm to test for intersection against many boxes but just one ray;
    // in this case, the inverse ray is cached outside and directly given here as input
    // we could then further specialize this function to the direction of the ray
    // because also the sign[] variables and hence the branches are predefined

    // one could do: template <class Backend, int sign0, int sign1, int sign2>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static Precision IntersectMultiple(
            Vector3D<typename Backend::precision_v> const lowercorners,
            Vector3D<typename Backend::precision_v> const uppercorners,
            Vector3D<Precision> const &point,
            Vector3D<Precision> const &inverseray,
            Precision t0,
            Precision t1 ){
      // intersection algorithm 1 ( Amy Williams )

      typedef typename Backend::precision_v Float_t;

      Float_t tmin, tmax, tymin, tymax, tzmin, tzmax;
      // IF THERE IS A STEPMAX; COULD ALSO CHECK SAFETIES

      // TODO: we should promote this to handle multiple boxes
      // we might need to have an Index type

      // int sign[3];
      Float_t sign[3]; // this also exists
      sign[0] = inverseray.x() < 0;
      sign[1] = inverseray.y() < 0;

      // observation: we always compute sign and 1-sign; so we could do the assignment
      // to tmin and tmax in a masked assignment thereafter

      //tmin =  (corners[(int)sign[0]].x()   -point.x())*inverserayx;
      //tmax =  (corners[(int)(1-sign[0])].x() -point.x())*inverserayx;
      //tymin = (corners[(int)(sign[1])].y()   -point.y())*inverserayy;
      //tymax = (corners[(int)(1-sign[1])].y() -point.y())*inverserayy;

      double x0 = (lowercorners.x() - point.x())*inverseray.x();
      double x1 = (uppercorners.x() - point.x())*inverseray.x();
      double y0 = (lowercorners.y() - point.y())*inverseray.y();
      double y1 = (uppercorners.y() - point.y())*inverseray.y();
      // could we do this using multiplications?
  //    tmin =   !sign[0] ?  x0 : x1;
  //    tmax =   sign[0] ? x0 : x1;
  //    tymin =  !sign[1] ?  y0 : y1;
  //    tymax =  sign[1] ? y0 : y1;

      // could completely get rid of this ? because the sign is determined by the outside ray

      tmin =   (1-sign[0])*x0 + sign[0]*x1;
      tmax =   sign[0]*x0 + (1-sign[0])*x1;
      tymin =  (1-sign[1])*y0 + sign[1]*y1;
      tymax =  sign[1]*y0 + (1-sign[1])*y1;

      //tmax =  (corners[(int)(1-sign[0])].x() -point.x())*inverserayx;
      //tymin = (corners[(int)(sign[1])].y()   -point.y())*inverserayy;
      //tymax = (corners[(int)(1-sign[1])].y() -point.y())*inverserayy;

      if((tmin > tymax) || (tymin > tmax))
          return vecgeom::kInfinity;

     //  double inverserayz = 1./ray.z();
      sign[2] = inverseray.z() < 0;

      if(tymin > tmin)
          tmin = tymin;
      if(tymax < tmax)
          tmax = tymax;

      //
      //tzmin = (lowercorners[(int) sign[2]].z()   -point.z())*inverseray.z();
      //tzmax = (uppercorners[(int)(1-sign[2])].z() -point.z())*inverseray.z();

      if((tmin > tzmax) || (tzmin > tmax))
          return vecgeom::kInfinity; // false
      if((tzmin > tmin))
          tmin = tzmin;
      if(tzmax < tmax)
          tmax = tzmax;

      if( ! ((tmin < t1) && (tmax > t0)) )
          return vecgeom::kInfinity;
     // std::cerr << "tmin " << tmin << " tmax " << tmax << "\n";
      // return true;
      return tmin;
    }
}; // End struct BoxImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::UnplacedContains(
    UnplacedBox const &box,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  ContainsKernel<Backend>(box.dimensions(), localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::Contains(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> &localPoint,
    typename Backend::bool_v &inside) {

  localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
  UnplacedContains<Backend>(unplaced, localPoint, inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  InsideKernel<Backend>(unplaced.dimensions(),
                        transformation.Transform<transCodeT, rotCodeT>(point),
                        inside);

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToInKernel<Backend>(
    unplaced.dimensions(),
    transformation.Transform<transCodeT, rotCodeT>(point),
    transformation.TransformDirection<rotCodeT>(direction),
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedBox const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToOutKernel<Backend>(
    unplaced.dimensions(),
    point,
    direction,
    stepMax,
    distance
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void BoxImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedBox const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToInKernel<Backend>(
    unplaced.dimensions(),
    transformation.Transform<transCodeT, rotCodeT>(point),
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void BoxImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedBox const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  SafetyToOutKernel<Backend>(
    unplaced.dimensions(),
    point,
    safety
  );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::ContainsKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

  typedef typename Backend::bool_v Boolean_t;
  Boolean_t unused;
  Boolean_t outside;
  GenericKernelForContainsAndInside<Backend, false>(dimensions,
    localPoint, unused, outside);
  inside=!outside;
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend, bool ForInside>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &completelyinside,
    typename Backend::bool_v &completelyoutside) {

//    using vecgeom::GenericKernels;
// here we are explicitely unrolling the loop since  a for statement will likely be a penality
// check if second call to Abs is compiled away
    // and it can anyway not be vectorized
    /* x */
    completelyoutside = Abs(localPoint[0]) > MakePlusTolerant<ForInside>( dimensions[0] );
    if (ForInside)
    {
        completelyinside = Abs(localPoint[0]) < MakeMinusTolerant<ForInside>( dimensions[0] );
    }
    if (Backend::early_returns) {
      if ( IsFull (completelyoutside) ) {
        return;
      }
    }
/* y */
    completelyoutside |= Abs(localPoint[1]) > MakePlusTolerant<ForInside>( dimensions[1] );
    if (ForInside)
    {
      completelyinside &= Abs(localPoint[1]) < MakeMinusTolerant<ForInside>( dimensions[1] );
    }
    if (Backend::early_returns) {
      if ( IsFull (completelyoutside) ) {
        return;
      }
    }
/* z */
    completelyoutside |= Abs(localPoint[2]) > MakePlusTolerant<ForInside>( dimensions[2] );
    if (ForInside)
    {
      completelyinside &= Abs(localPoint[2]) < MakeMinusTolerant<ForInside>( dimensions[2] );
    }
    return;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::InsideKernel(
    Vector3D<Precision> const &boxDimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  typedef typename Backend::bool_v      Boolean_t;
  Boolean_t completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Backend,true>(
      boxDimensions, point, completelyinside, completelyoutside);
  inside=EInside::kSurface;
  MaskedAssign(completelyoutside, EInside::kOutside, &inside);
  MaskedAssign(completelyinside, EInside::kInside, &inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Floating_t;
  typedef typename Backend::bool_v      Boolean_t;

  Vector3D<Floating_t> safety;
  Boolean_t done = Backend::kFalse;

#ifdef VECGEOM_NVCC
  #define surfacetolerant true
#else
  static const bool surfacetolerant=true;
#endif

  safety[0] = Abs(point[0]) - dimensions[0];
  safety[1] = Abs(point[1]) - dimensions[1];
  safety[2] = Abs(point[2]) - dimensions[2];

  done |= (safety[0] >= stepMax ||
           safety[1] >= stepMax ||
           safety[2] >= stepMax);
  if ( IsFull(done) ) return;


  Boolean_t inside = Backend::kFalse;
  inside = safety[0] < 0 && safety[1] < 0 && safety[2] < 0;
  MaskedAssign(!done && inside, -1., &distance);
  done |= inside;
  if ( IsFull(done) ) return;

  Floating_t next, coord1, coord2;
  Boolean_t hit;

  // x
  next = safety[0] / Abs(direction[0] + kMinimum);
  coord1 = point[1] + next * direction[1];
  coord2 = point[2] + next * direction[2];
  hit = safety[0] >= MakeMinusTolerant<surfacetolerant>(0.) &&
        point[0] * direction[0] < 0 &&
        Abs(coord1) <= dimensions[1] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, &distance);
  done |= hit;
  if ( IsFull(done) ) return;

  // y
  next = safety[1] / Abs(direction[1] + kMinimum);
  coord1 = point[0] + next * direction[0];
  coord2 = point[2] + next * direction[2];
  hit = safety[1] >= MakeMinusTolerant<surfacetolerant>(0.) &&
        point[1] * direction[1] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[2];
  MaskedAssign(!done && hit, next, &distance);
  done |= hit;
  if ( IsFull(done) ) return;

  // z
  next = safety[2] / Abs(direction[2] + kMinimum);
  coord1 = point[0] + next * direction[0];
  coord2 = point[1] + next * direction[1];
  hit = safety[2] >= MakeMinusTolerant<surfacetolerant>(0.) &&
        point[2] * direction[2] < 0 &&
        Abs(coord1) <= dimensions[0] &&
        Abs(coord2) <= dimensions[1];
  MaskedAssign(!done && hit, next, &distance);

#ifdef VECGEOM_NVCC
  #undef surfacetolerant
#endif
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Floating_t;
    // typedef typename Backend::bool_v Boolean_t;

    Vector3D<Floating_t> safety;
    // Boolean_t inside;

    distance = kInfinity;

    //safety[0] = Abs(point[0]) - dimensions[0];
    //safety[1] = Abs(point[1]) - dimensions[1];
    //safety[2] = Abs(point[2]) - dimensions[2];

    //inside = safety[0] < stepMax &&
    //         safety[1] < stepMax &&
    //         safety[2] < stepMax;
    //if (inside == Backend::kFalse) return;

    Vector3D<Floating_t> inverseDirection = Vector3D<Floating_t>(
      1. / (direction[0] + kMinimum),
      1. / (direction[1] + kMinimum),
      1. / (direction[2] + kMinimum)
    );
    Vector3D<Floating_t> distances = Vector3D<Floating_t>(
      (dimensions[0] - point[0]) * inverseDirection[0],
      (dimensions[1] - point[1]) * inverseDirection[1],
      (dimensions[2] - point[2]) * inverseDirection[2]
    );

    MaskedAssign(direction[0] < 0,
                 (-dimensions[0] - point[0]) * inverseDirection[0],
                 &distances[0]);
    MaskedAssign(direction[1] < 0,
                 (-dimensions[1] - point[1]) * inverseDirection[1],
                 &distances[1]);
    MaskedAssign(direction[2] < 0,
                 (-dimensions[2] - point[2]) * inverseDirection[2],
                 &distances[2]);

    distance = distances[0];
    MaskedAssign(distances[1] < distance, distances[1], &distance);
    MaskedAssign(distances[2] < distance, distances[2], &distance);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Floating_t;

  safety = -dimensions[0] + Abs(point[0]);
  Floating_t safetyY = -dimensions[1] + Abs(point[1]);
  Floating_t safetyZ = -dimensions[2] + Abs(point[2]);

  // TODO: check if we should use MIN/MAX here instead
  MaskedAssign(safetyY > safety, safetyY, &safety);
  MaskedAssign(safetyZ > safety, safetyZ, &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    Vector3D<Precision> const &dimensions,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

   typedef typename Backend::precision_v Floating_t;

   safety = dimensions[0] - Abs(point[0]);
   Floating_t safetyY = dimensions[1] - Abs(point[1]);
   Floating_t safetyZ = dimensions[2] - Abs(point[2]);

   // TODO: check if we should use MIN here instead
   MaskedAssign(safetyY < safety, safetyY, &safety);
   MaskedAssign(safetyZ < safety, safetyZ, &safety);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void BoxImplementation<transCodeT, rotCodeT>::NormalKernel(
     UnplacedBox const &box,
     Vector3D<typename Backend::precision_v> const &point,
     Vector3D<typename Backend::precision_v> &normal,
     typename Backend::bool_v &valid
    ) {

        typedef typename Backend::precision_v Floating_t;
        typedef typename Backend::bool_v Boolean_t;

         // Computes the normal on a surface and returns it as a unit vector
         //   In case a point is further than tolerance_normal from a surface, set validNormal=false
         //   Must return a valid vector. (even if the point is not on the surface.)
         //
         //   On an edge or corner, provide an average normal of all facets within tolerance
         // NOTE: the tolerance value used in here is not yet the global surface
         //     tolerance - we will have to revise this value - TODO
         // this version does not yet consider the case when we are not on the surface

         Vector3D<Precision> dimensions= box.dimensions();

         constexpr double delta = 100.*kTolerance;
         constexpr double kInvSqrt2 = 0.7071067811865475; // = 1. / Sqrt(2.);
         constexpr double kInvSqrt3 = 0.5773502691896258; // = 1. / Sqrt(3.);
         normal.Set(0.);
         Floating_t nsurf = 0;
         Floating_t safmin(kInfinity);

         // do a loop here over dimensions
         for( int dim = 0; dim < 3; ++dim )
         {
             Floating_t currentsafe = Abs(Abs(point[dim]) - dimensions[dim]);
             MaskedAssign( currentsafe < safmin, currentsafe, &safmin );

             // close to this surface
             Boolean_t closetoplane = currentsafe < delta;
             if( ! IsEmpty( closetoplane ) )
             {
                Floating_t nsurftmp = nsurf + 1.;

                Floating_t sign(1.);
                MaskedAssign( point[dim] < 0, -1., &sign);
                Floating_t tmpnormalcomponent = normal[dim] + sign;

                MaskedAssign( closetoplane, nsurftmp, &nsurf );
                MaskedAssign( closetoplane, tmpnormalcomponent, &normal[dim] );
             }
         }

         valid = Backend::kTrue;
         valid &= nsurf>0;
         MaskedAssign( nsurf == 3., normal*kInvSqrt3, &normal );
         MaskedAssign( nsurf == 2., normal*kInvSqrt2, &normal );

        // TODO: return normal in case of nonvalid case;
        // need to keep track of minimum safety direction
    }

  struct ABBoxImplementation {

  // a contains kernel to be used with aligned bounding boxes
  // scalar and vector modes (aka backend) for boxes but only single points
  // should be useful to test one point against many bounding boxes
  // TODO: check if this can be unified with the normal generic box kernel
  // this might be possible with 2 backend template parameters
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ABBoxContainsKernel(
        Vector3D<typename Backend::precision_v> const &lowercorner,
        Vector3D<typename Backend::precision_v> const &uppercorner,
        Vector3D<Precision> const &point,
        typename Backend::bool_v &inside) {

        inside =  lowercorner.x() < point.x();
        inside &= uppercorner.x() > point.x();
        if( IsEmpty(inside) ) return;

        inside &= lowercorner.y() < point.y();
        inside &= uppercorner.y() > point.y();
        if( IsEmpty(inside) ) return;

        inside &= lowercorner.z() < point.z();
        inside &= uppercorner.z() > point.z();
  }

  }; // end aligned bounding box struct

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif // VECGEOM_VOLUMES_KERNEL_BOXIMPLEMENTATION_H_
