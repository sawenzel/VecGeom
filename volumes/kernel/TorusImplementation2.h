/// @file TorusImplementation.h

#ifndef VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION2_H_
#define VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION2_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/kernel/TubeImplementation.h"
#include "volumes/UnplacedTorus2.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(TorusImplementation2, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {

    //_____________________________________________________________________________
    template <typename T>
    VECGEOM_CUDA_HEADER_BOTH
    unsigned int SolveCubic(T a, T b, T c, T *x)
    {
    // Find real solutions of the cubic equation : x^3 + a*x^2 + b*x + c = 0
    // Input: a,b,c
    // Output: x[3] real solutions
    // Returns number of real solutions (1 or 3)
       const T ott = 1./3.;
       const T sq3 = Sqrt(3.);
       const T inv6sq3 = 1./(6.*sq3);
       unsigned int ireal = 1;
       T p = b - a * a * ott;
       T q = c - a * b * ott + 2. * a * a * a * ott * ott * ott;
       T delta = 4 * p * p * p + 27. * q * q;
       T t,u;
       if (delta>=0) {
          delta = Sqrt(delta);
          t = (-3*q*sq3+delta)*inv6sq3;
          u = (3*q*sq3+delta)*inv6sq3;
          x[0] = CopySign(1.,t)*Cbrt(Abs(t))-
                 CopySign(1.,u)*Cbrt(Abs(u))-a*ott;
       } else {
          delta = Sqrt(-delta);
          t = -0.5*q;
          u = delta*inv6sq3;
          x[0] = 2.*Pow(t*t+u*u,0.5*ott) * cos(ott*ATan2(u,t));
          x[0] -= a*ott;
       }

       t = x[0]*x[0]+a*x[0]+b;
       u = a+x[0];
       delta = u*u-4.*t;
       if (delta>=0) {
          ireal = 3;
          delta = Sqrt(delta);
          x[1] = 0.5*(-u-delta);
          x[2] = 0.5*(-u+delta);
       }
       return ireal;
    }

    template <typename T, unsigned int i, unsigned int j>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    void CmpAndSwap(T *array){
           if( array[i] > array[j] ){
           T c = array[j];
           array[j] = array[i];
           array[i] = c;
       }}

       // a special function to sort a 4 element array
       // sorting is done inplace and in increasing order
       // implementation comes from a sorting network
       template<typename T>
       VECGEOM_INLINE
       VECGEOM_CUDA_HEADER_BOTH
       void Sort4(T *array){
            CmpAndSwap<T,0,2>( array );
            CmpAndSwap<T,1,3>( array );
            CmpAndSwap<T,0,1>( array );
            CmpAndSwap<T,2,3>( array );
            CmpAndSwap<T,1,2>( array );
       }

    // solve quartic taken from ROOT/TGeo and adapted
    //_____________________________________________________________________________

       template <typename T>
       VECGEOM_CUDA_HEADER_BOTH
       int SolveQuartic(T a, T b, T c, T d, T * x) {
         // Find real solutions of the quartic equation : x^4 + a*x^3 + b*x^2 + c*x + d = 0
         // Input: a,b,c,d
         // Output: x[4] - real solutions
         // Returns number of real solutions (0 to 3)
         T e = b - 3. * a * a / 8.;
         T f = c + a * a * a / 8. - 0.5 * a * b;
         T g = d - 3. * a * a * a * a / 256. + a * a * b / 16. - a * c / 4.;
         T xx[4] = {vecgeom::kInfinity, vecgeom::kInfinity, vecgeom::kInfinity, vecgeom::kInfinity};
         T delta;
         T h = 0.;
         unsigned int ireal = 0;

         // special case when f is zero
         if (Abs(f) < 1E-6) {
           delta = e * e - 4. * g;
           if (delta < 0)
             return 0;
           delta = Sqrt(delta);
           h = 0.5 * (-e - delta);
           if (h >= 0) {
             h = Sqrt(h);
             x[ireal++] = -h - 0.25 * a;
             x[ireal++] = h - 0.25 * a;
           }
           h = 0.5 * (-e + delta);
           if (h >= 0) {
             h = Sqrt(h);
             x[ireal++] = -h - 0.25 * a;
             x[ireal++] = h - 0.25 * a;
           }
           Sort4(x);
           return ireal;
         }

         if (Abs(g) < 1E-6) {
           x[ireal++] = -0.25 * a;
           // this actually wants to solve a second order equation
           // we should specialize if it happens often
           unsigned int ncubicroots = SolveCubic<T>(0, e, f, xx);
           // this loop is not nice
           for (unsigned int i = 0; i < ncubicroots; i++)
             x[ireal++] = xx[i] - 0.25 * a;
           Sort4(x); // could be Sort3
           return ireal;
         }

         ireal = SolveCubic<T>(2. * e, e * e - 4. * g, -f * f, xx);
         if (ireal == 1) {
           if (xx[0] <= 0)
             return 0;
           h = Sqrt(xx[0]);
         } else {
           // 3 real solutions of the cubic
           for (unsigned int i = 0; i < 3; i++) {
             h = xx[i];
             if (h >= 0)
               break;
           }
           if (h <= 0)
             return 0;
           h = Sqrt(h);
         }
         T j = 0.5 * (e + h * h - f / h);
         ireal = 0;
         delta = h * h - 4. * j;
         if (delta >= 0) {
           delta = Sqrt(delta);
           x[ireal++] = 0.5 * (-h - delta) - 0.25 * a;
           x[ireal++] = 0.5 * (-h + delta) - 0.25 * a;
         }
         delta = h * h - 4. * g / j;
         if (delta >= 0) {
           delta = Sqrt(delta);
           x[ireal++] = 0.5 * (h - delta) - 0.25 * a;
           x[ireal++] = 0.5 * (h + delta) - 0.25 * a;
         }
         Sort4(x);
         return ireal;
       }

class PlacedTorus2;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct TorusImplementation2 {
  
   static const int transC = transCodeT;
   static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedTorus2;
  using UnplacedShape_t = UnplacedTorus2;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedTorus<%i, %i>", transCodeT, rotCodeT);
  }

  /////GenericKernel Contains/Inside implementation
  template <typename Backend, bool ForInside, bool notForDisk>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedTorus2 const &torus,
          Vector3D<typename Backend::precision_v> const &point,
          typename Backend::bool_v &completelyinside,
      typename Backend::bool_v &completelyoutside)

   {
   // using vecgeom::GenericKernels;
   // here we are explicitely unrolling the loop since  a for statement will likely be a penality
   // check if second call to Abs is compiled away
   // and it can anyway not be vectorized
    /* rmax */
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;


//    // very fast check on z-height
//    completelyoutside = point[2] > MakePlusTolerant<ForInside>( torus.rmax() );
//    if (Backend::early_returns) {
//         if ( IsFull(completelyoutside) ) {
//           return;
//         }
//    }

    Float_t rxy = Sqrt(point[0]*point[0] + point[1]*point[1]);
    Float_t radsq = (rxy - torus.rtor()) * (rxy -  torus.rtor()) + point[2]*point[2];

    completelyoutside = radsq > MakePlusTolerantSquare<ForInside>( torus.rmax(),torus.rmax2() );//rmax
    if (ForInside) {
      completelyinside = radsq < MakeMinusTolerantSquare<ForInside>( torus.rmax(),torus.rmax2() );
    }
    if (Backend::early_returns) {
      if ( IsFull(completelyoutside) ) {
        return;
      }
    }
    /* rmin */
    completelyoutside |= radsq < MakeMinusTolerantSquare<ForInside>( torus.rmin(),torus.rmin2() );//rmin
    if (ForInside) {
      completelyinside &= radsq > MakePlusTolerantSquare<ForInside>( torus.rmin(),torus.rmin2() );
    }

    // NOT YET NEEDED WHEN NOT PHI TREATMENT
    if (Backend::early_returns) {
        if ( IsFull(completelyoutside) ) {
            return;
          }
    }
    
    /* phi */
    if(( torus.dphi() < kTwoPi  )&&(notForDisk)){
        Bool_t completelyoutsidephi;
        Bool_t completelyinsidephi;
        torus.GetWedge().GenericKernelForContainsAndInside<Backend,ForInside>( point,
            completelyinsidephi, completelyoutsidephi );

        completelyoutside |= completelyoutsidephi;
        if(ForInside)
            completelyinside &= completelyinsidephi;
    }
    
  }

  template <class Backend, bool notForDisk>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      UnplacedTorus2 const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside)
  {
   typedef typename Backend::bool_v Bool_t;
    Bool_t unused;
    Bool_t outside;
    GenericKernelForContainsAndInside<Backend, false,notForDisk>(torus,
    point, unused, outside);
    inside = !outside;
 }
  //template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void InsideKernel(
     UnplacedTorus2 const &torus,
     Vector3D<typename Backend::precision_v> const &point,
     typename Backend::inside_v &inside) {

  typedef typename Backend::bool_v      Bool_t;
  //
  Bool_t completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Backend,true,true>(torus, 
  point, completelyinside, completelyoutside);
  inside = EInside::kSurface;
  MaskedAssign(completelyoutside, EInside::kOutside, &inside);
  MaskedAssign(completelyinside, EInside::kInside, &inside);
 }


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains( UnplacedTorus2 const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {
      ContainsKernel<Backend,true>(torus, point, inside);
}
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContainsDisk( UnplacedTorus2 const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside) {
      ContainsKernel<Backend,false>(torus, point, inside);
}
  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedTorus2 const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside){
   localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
   UnplacedContains<Backend>(unplaced, localPoint, inside);

  }
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedTorus2 const &torus,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::inside_v &inside) {
    InsideKernel<Backend>(torus, transformation.Transform<transCodeT, rotCodeT>(point),  inside);
}
  

  /////End GenericKernel Contains/Inside implementation

template <class T>
VECGEOM_CUDA_HEADER_BOTH
static
T CheckZero(T b, T c, T d, T e, T x){
  T x2 = x * x;
  return x2 * x2 + b * x2 * x + c * x2 + d * x + e;
}

template <class T>
VECGEOM_CUDA_HEADER_BOTH
static T NewtonIter(T b, T c, T d, T e, T x, T fold) {
  T x2 = x * x;
  T fprime = 4 * x2 * x + 3 * b * x2 + 2 * c * x + d;
  return x - fold / fprime;
}

//_____________________________________________________________________________
template <class T>
VECGEOM_CUDA_HEADER_BOTH
static T DistSqrToTorusR(UnplacedTorus2 const &torus, Vector3D<T> const &point, Vector3D<T> const &dir, T dist) {
  // Computes the squared distance to "axis" or "defining ring" of the torus from point point + t*dir;
  Vector3D<T> p = point + dir * dist;
  T rxy = p.Perp();
  return (rxy - torus.rtor()) * (rxy - torus.rtor()) + p.z() * p.z();
}

  template <class Backend, bool ForRmin>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static
  typename Backend::precision_v ToBoundary(
          UnplacedTorus2 const& torus,
          Vector3D<typename Backend::precision_v> const &pt,
          Vector3D<typename Backend::precision_v> const &dir,
          Precision radius, bool out )
  {
    // to be taken from ROOT
    // Returns distance to the surface or the torus from a point, along
    // a direction. Point is close enough to the boundary so that the distance
    // to the torus is decreasing while moving along the given direction.
    typedef typename Backend::precision_v Real_v;

    // Compute coeficients of the quartic
    Real_v s = vecgeom::kInfinity;
    Real_v tol = vecgeom::kTolerance;
    Real_v r0sq = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
    Real_v rdotn = pt[0] * dir[0] + pt[1] * dir[1] + pt[2] * dir[2];
    Real_v rsumsq = torus.rtor2() + radius * radius;
    Real_v a = 4. * rdotn;
    Real_v b = 2. * (r0sq + 2. * rdotn * rdotn - rsumsq + 2. * torus.rtor2() * dir[2] * dir[2]);
    Real_v c = 4. * (r0sq * rdotn - rsumsq * rdotn + 2. * torus.rtor2() * pt[2] * dir[2]);
    Real_v d = r0sq * r0sq - 2. * r0sq * rsumsq + 4. * torus.rtor2() * pt[2] * pt[2]
                                     + (torus.rtor2() - radius * radius) * (torus.rtor2() - radius * radius);

    Real_v x[4] = { vecgeom::kInfinity, vecgeom::kInfinity, vecgeom::kInfinity, vecgeom::kInfinity };
    unsigned int nsol = 0;

    // special condition
    if (Abs(dir[2]) < 1E-3 && Abs(pt[2]) < 0.1 * radius) {
      Real_v r0 = torus.rtor() - Sqrt((radius- pt[2]) * (radius+ pt[2]));
      Real_v invdirxy2 = 1./(1 - dir.z()*dir.z());
      Real_v b0 = (pt[0] * dir[0] + pt[1] * dir[1]) * invdirxy2;
      Real_v c0 = (pt[0] * pt[0] + (pt[1] - r0) * (pt[1] + r0)) * invdirxy2;
      Real_v delta = b0 * b0 - c0;
      if (delta > 0) {
        x[nsol] = -b0 - Sqrt(delta);
        if (x[nsol] > -tol)
          nsol++;
        x[nsol] = -b0 + Sqrt(delta);
        if (x[nsol] > -tol)
          nsol++;
      }
      r0 = torus.rtor() + Sqrt((radius- pt[2]) * (radius+ pt[2]));
      c0 = (pt[0] * pt[0] + (pt[1] - r0) * (pt[1] + r0)) * invdirxy2;
      delta = b0 * b0 - c0;
      if (delta > 0) {
        x[nsol] = -b0 - Sqrt(delta);
        if (x[nsol] > -tol)
          nsol++;
        x[nsol] = -b0 + Sqrt(delta);
        if (x[nsol] > -tol)
          nsol++;
      }
      if (nsol) {
        Sort4(x);
      }
    }
    else { // generic case
      nsol = SolveQuartic(a, b, c, d, x);
    }
    if (!nsol)
      return vecgeom::kInfinity;

    // look for first positive solution
    Real_v ndotd;
    bool inner = Abs(radius - torus.rmin()) < vecgeom::kTolerance;
    for (unsigned int i = 0; i < nsol; i++) {
      if (x[i] < -10)
        continue;

      Vector3D<Precision> r0 = pt + x[i]*dir;
      if (torus.dphi() < vecgeom::kTwoPi) {
        if (! torus.GetWedge().ContainsWithBoundary<Backend>(r0))
            continue;
      }

      Vector3D<Precision> norm = r0;
      r0.z()=0.;
      r0.Normalize();
      r0*=torus.rtor();
      norm -= r0;
     // norm = pt
     // for (unsigned int ipt = 0; ipt < 3; ipt++)
     //   norm[ipt] = pt[ipt] + x[i] * dir[ipt] - r0[ipt];
     // ndotd = norm[0] * dir[0] + norm[1] * dir[1] + norm[2] * dir[2];
      ndotd = norm.Dot(dir);
      if (inner ^ out) {
        if (ndotd < 0)
          continue; // discard this solution
      } else {
        if (ndotd > 0)
          continue; // discard this solution
      }
      s = x[i];
      // refine solution with Newton iterations
      Real_v eps = vecgeom::kInfinity;
      Real_v delta = s * s * s * s + a * s * s * s + b * s * s + c * s + d;
      Real_v eps0 = -delta / (4. * s * s * s + 3. * a * s * s + 2. * b * s + c);
      while (Abs(eps) > vecgeom::kTolerance) {
        if (Abs(eps0) > 100)
          break;
        s += eps0;
        if (Abs(s + eps0) < vecgeom::kTolerance)
          break;
        delta = s * s * s * s + a * s * s * s + b * s * s + c * s + d;
        eps = -delta / (4. * s * s * s + 3. * a * s * s + 2. * b * s + c);
        if (Abs(eps) > Abs(eps0))
          break;
        eps0 = eps;
      }
      // discard this solution
      if (s < -vecgeom::kTolerance)
        continue;
      return Max(0., s);
    }
    return vecgeom::kInfinity;
  }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void DistanceToIn(
        UnplacedTorus2 const &torus,
        Transformation3D const &transformation,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &direction,
        typename Backend::precision_v const &stepMax,
        typename Backend::precision_v &distance) {
      
      typedef typename Backend::precision_v Float_t;
      typedef typename Backend::bool_v Bool_t;
      
      Vector3D<Float_t> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
      Vector3D<Float_t> localDirection = transformation.Transform<transCodeT, rotCodeT>(direction);

      ////////First naive implementation
      distance = kInfinity;

      //Check Bounding Cylinder first
      Bool_t inBounds;
      Bool_t done;
      Float_t tubeDistance = kInfinity;

#ifndef VECGEOM_NO_SPECIALIZATION
      // call the tube functionality -- first of all we check whether we are inside
      // bounding volume
      TubeImplementation<translation::kIdentity,
        rotation::kIdentity, TubeTypes::HollowTube>::UnplacedContains<Backend>(
            torus.GetBoundingTube(), localPoint, inBounds);


      // only need to do this check if all particles (in vector) are outside ( otherwise useless )
        TubeImplementation<translation::kIdentity, rotation::kIdentity, TubeTypes::HollowTube>::DistanceToIn<Backend>(
            torus.GetBoundingTube(), transformation, localPoint, localDirection, stepMax, tubeDistance);
#else
      // call the tube functionality -- first of all we check whether we are inside
      // bounding volume
      TubeImplementation<translation::kIdentity,
        rotation::kIdentity, TubeTypes::UniversalTube>::UnplacedContains<Backend>(
            torus.GetBoundingTube(), localPoint, inBounds);


      // only need to do this check if all particles (in vector) are outside ( otherwise useless )
      if( !inBounds )
      TubeImplementation<translation::kIdentity,
            rotation::kIdentity, TubeTypes::UniversalTube>::DistanceToIn<Backend>(
                torus.GetBoundingTube(), transformation, localPoint,
                localDirection, stepMax, tubeDistance);
      else
          tubeDistance = 0.;
#endif // VECGEOM_NO_SPECIALIZATION
      done = (!inBounds && tubeDistance == kInfinity);

       if (Backend::early_returns) {
           if ( IsFull(done) ) {
            return;
       }
       }


       // Bool_t hasphi = (torus.dphi() < vecgeom::kTwoPi);

       // Check if bounding tube intersection is due to phi in which case we are done
       Precision daxis = DistSqrToTorusR(torus, localPoint, localDirection, tubeDistance);
       if (daxis >= torus.rmin2() && daxis < torus.rmax2()){
            distance=tubeDistance;
             return ;
       }
       // Not a phi crossing -> propagate until we cross the ring.
      // pt = localPoint + tubeDistance * localDirection;


       // Point pt is inside the bounding ring, no phi crossing so far.
       // Check if we are in the hole.
// we comment out treatment of RMIN for now
       //       if (daxis < 0)
//         daxis = Daxis(pt, dir, 0);
//       if (daxis < fRmin + 1.E-8) {
//         // We are in the hole. Check if we came from outside.
//         if (snext > 0) {
//           // we can cross either the inner torus or exit the other hole.
//           snext += 0.1 * eps;
//           for (i = 0; i < 3; i++)
//             pt[i] += 0.1 * eps * dir[i];
//         }
//         // We are in the hole from the beginning.
//         // find first crossing with inner torus
//         dd = ToBoundary(pt, dir, fRmin, kFALSE);
//         // find exit distance from inner bounding ring
//         if (hasphi)
//           dring = TGeoTubeSeg::DistFromInsideS(pt, dir, fR - fRmin, fR + fRmin, fRmin, c1, s1, c2, s2, cm, sm, cdfi);
//         else
//           dring = TGeoTube::DistFromInsideS(pt, dir, fR - fRmin, fR + fRmin, fRmin);
//         if (dd < dring)
//           return (snext + dd);
//         // we were exiting a hole inside phi hole
//         snext += dring + eps;
//         for (i = 0; i < 3; i++)
//           pt[i] = point[i] + snext * dir[i];
//         snext += DistFromOutside(pt, dir, 3);
//         return snext;
//       } // end inside inner tube

       // We are inside the outer ring, having daxis>fRmax
       // Compute distance to exit the bounding ring (again)
       //if (snext > 0) {
         // we can cross either the inner torus or exit the other hole.
        // snext += 0.1 * eps;
        // for (i = 0; i < 3; i++)
        //   pt[i] += 0.1 * eps * dir[i];
      // }
       // Check intersection with outer torus
       //Float_t dd = ToBoundary<Backend,false>(torus,point,dir,torus.rmax(),true);
       Float_t dd = ToBoundary<Backend, false>(torus, localPoint, localDirection, torus.rmax(), false);
//       if (hasphi)
//         dring = TGeoTubeSeg::DistFromInsideS(pt, dir, TMath::Max(0., fR - fRmax - eps), fR + fRmax + eps, fRmax + eps,
//                                              c1, s1, c2, s2, cm, sm, cdfi);
//       else
//         dring = TGeoTube::DistFromInsideS(pt, dir, TMath::Max(0., fR - fRmax - eps), fR + fRmax + eps, fRmax + eps);
//       if (dd < dring) {
//         snext += dd;
//         return snext;
//       }
//       // We are exiting the bounding ring before crossing the torus -> propagate
//       snext += dring + eps;
//       for (i = 0; i < 3; i++)
//         pt[i] = point[i] + snext * dir[i];
//       snext += DistFromOutside(pt, dir, 3);
       distance = dd;
       return;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedTorus2 const &torus,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &dir,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    distance = kInfinity;
   
//#ifndef VECGEOM_NO_SPECIALIZATION
//      // call the tube functionality -- first of all we check whether we are inside
//      // bounding volume
//      TubeImplementation<translation::kIdentity,
//        rotation::kIdentity, TubeTypes::HollowTube>::UnplacedContains<Backend>(
//            torus.GetBoundingTube(), point, inBounds);
//#else
//      // call the tube functionality -- first of all we check whether we are inside
//      // bounding volume
//      TubeImplementation<translation::kIdentity,
//        rotation::kIdentity, TubeTypes::UniversalTube>::UnplacedContains<Backend>(
//            torus.GetBoundingTube(), point, inBounds);
//
//#endif // VECGEOM_NO_SPECIALIZATION
//
//       done = (!inBounds );
//       if (Backend::early_returns) {
//       if ( IsFull(done) ) {
//             distance = 0;
//            return ;
//       }
//       }

    bool hasphi = (torus.dphi()<kTwoPi);
    bool hasrmin = (torus.rmin()>0);

    Float_t dout = ToBoundary<Backend,false>(torus,point,dir,torus.rmax(),true);
    Float_t din(kInfinity);
    if( hasrmin )
    {
      din = ToBoundary<Backend,true>(torus,point,dir,torus.rmin(),true);
    }
    distance = Min(dout,din);

    if( hasphi )
    {
    // TODO
      Float_t distPhi1;
      Float_t distPhi2;
      torus.GetWedge().DistanceToOut<Backend>(point,dir,distPhi1,distPhi2);
      Bool_t smallerphi = distPhi1 < distance;
      if(! IsEmpty(smallerphi) )
    {
          Vector3D<Float_t> intersectionPoint = point + dir*distPhi1;
          Bool_t insideDisk ;
          UnplacedContainsDisk<Backend>(torus,intersectionPoint,insideDisk);
          
          if( ! IsEmpty(insideDisk))//Inside Disk
          {
            Float_t diri=intersectionPoint.x()*torus.GetWedge().GetAlong1().x()+
                     intersectionPoint.y()*torus.GetWedge().GetAlong1().y();
            Bool_t rightside = (diri >=0);

        MaskedAssign( rightside && smallerphi && insideDisk, distPhi1, &distance);

      }
         
        }
      smallerphi = distPhi2 < distance;
      if(! IsEmpty(smallerphi) )
    {
        
      Vector3D<Float_t> intersectionPoint = point + dir*distPhi2;
          Bool_t insideDisk;
          UnplacedContainsDisk<Backend>(torus,intersectionPoint,insideDisk);
          if( ! IsEmpty(insideDisk))//Inside Disk
          {
            Float_t diri2=intersectionPoint.x()*torus.GetWedge().GetAlong2().x()+
            intersectionPoint.y()*torus.GetWedge().GetAlong2().y();
            Bool_t rightside = (diri2 >=0);
        MaskedAssign( rightside && (distPhi2 < distance) &&smallerphi && insideDisk, distPhi2, &distance);

      }
         }
    }
  
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedTorus2 const &torus,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {

    typedef typename Backend::precision_v Float_t;
    Vector3D<Float_t> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
   
    // implementation taken from TGeoTorus
    Float_t rxy = Sqrt( localPoint[0]*localPoint[0] + localPoint[1]*localPoint[1] );
    Float_t rad = Sqrt( (rxy - torus.rtor())*(rxy-torus.rtor()) + localPoint[2]*localPoint[2] );
    safety = rad - torus.rmax();
    if( torus.rmin() )
     {safety = Max( torus.rmin()- rad, rad - torus.rmax() );}
  
     bool hasphi = (torus.dphi()<kTwoPi);
     if ( hasphi && (rxy !=0.) ) {
       Float_t safetyPhi = torus.GetWedge().SafetyToIn<Backend>(localPoint);
       safety = Max(safetyPhi,safety);
     }
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedTorus2 const &torus,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety) {

    typedef typename Backend::precision_v Float_t;
    Float_t rxy = Sqrt( point[0]*point[0] + point[1]*point[1] );
    Float_t rad = Sqrt( (rxy - torus.rtor())*(rxy-torus.rtor()) + point[2]*point[2] );
    safety= torus.rmax() - rad;
    if( torus.rmin() ) {
        safety = Min( rad - torus.rmin(), torus.rmax() - rad );
    }
    
    // TODO: extend implementation for phi sector case
    bool hasphi = (torus.dphi() < kTwoPi);
    if (hasphi) {
      Float_t safetyPhi = torus.GetWedge().SafetyToOut<Backend>(point);
      safety = Min(safetyPhi, safety);
    }
  }
}; // end struct

} } // end namespace


#endif // VECGEOM_VOLUMES_KERNEL_TORUSIMPLEMENTATION_H_
