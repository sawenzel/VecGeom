/*
 * PolyconeImplementation.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */


#ifndef VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_POLYCONEIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedPolycone.h"
#include "volumes/kernel/ConeImplementation.h"

#include <cassert>
#include <cstdio>

namespace vecgeom {

  //VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(PolyconeImplementation,
  //     TranslationCode,transCodeT, RotationCode,rotCodeT)

  VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(PolyconeImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolycone;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct PolyconeImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedPolycone;
  using UnplacedShape_t = UnplacedPolycone;

  // here put all the implementations
  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
      printf("SpecializedPolycone<%i, %i>", transCodeT, rotCodeT);
  }

  /////GenericKernel Contains/Inside implementation
  template <typename Backend, bool ForInside>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedPolycone const &polycone,
            Vector3D<typename Backend::precision_v> const &point,
            typename Backend::bool_v &completelyinside,
            typename Backend::bool_v &completelyoutside)
     {
        // TBD
     }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void ContainsKernel(
        UnplacedPolycone const &polycone,
        Vector3D<typename Backend::precision_v> const &point,
        typename Backend::bool_v &inside)
    {
        // add z check
        if( point.z() < polycone.fZs[0] || point.z() > polycone.fZs[polycone.GetNSections()] )
        {
            inside = Backend::kFalse;
            return;
        }

        // now we have to find a section
        PolyconeSection const & sec = polycone.GetSection(point.z());

        Vector3D<Precision> localp;
        ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::Contains<Backend>(
                *sec.fSolid,
                Transformation3D(),
                point - Vector3D<Precision>(0,0,sec.fShift),
                localp,
                inside
        );
        return;
    }

    //template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    static void InsideKernel(
       UnplacedPolycone const &polycone,
       Vector3D<typename Backend::precision_v> const &point,
       typename Backend::inside_v &inside) {

        typename Backend::bool_v contains;
        ContainsKernel<Backend>(polycone,point,contains);
        if(contains){
            inside = EInside::kInside;
        }
        else
        {
            inside = EInside::kOutside;
        }
    }


    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void UnplacedContains( UnplacedPolycone const &polycone,
        Vector3D<typename Backend::precision_v> const &point,
        typename Backend::bool_v &inside) {
      // TODO: do this generically WITH a generic contains/inside kernel
      // forget about sector for the moment
      ContainsKernel<Backend>(polycone, point, inside);
    }

    template <typename Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Contains(
        UnplacedPolycone const &unplaced,
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
    static void Inside(UnplacedPolycone const &polycone,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       typename Backend::inside_v &inside) {

      InsideKernel<Backend>(polycone, transformation.Transform<transCodeT, rotCodeT>(point),  inside);

  }


      template <class Backend>
      VECGEOM_CUDA_HEADER_BOTH
      VECGEOM_INLINE
      static void DistanceToIn(
          UnplacedPolycone const &polycone,
          Transformation3D const &transformation,
          Vector3D<typename Backend::precision_v> const &point,
          Vector3D<typename Backend::precision_v> const &direction,
          typename Backend::precision_v const &stepMax,
          typename Backend::precision_v &distance) {

         Vector3D<typename Backend::precision_v> p = transformation.Transform<transCodeT,rotCodeT>(point);
         Vector3D<typename Backend::precision_v> v = transformation.TransformDirection<rotCodeT>(direction);
        // TODO
        // TODO: add bounding box check maybe??

    distance=kInfinity;
        int increment = (v.z() > 0) ? 1 : -1;
        if (std::fabs(v.z()) < kTolerance) increment = 0;
        int index = polycone.GetSectionIndex(p.z());
        if(index == -1) index = 0;
        if(index == -2) index = polycone.GetNSections()-1;
       
    //std::cout<<" Entering DIN index="<<index<<" NSec="<<polycone.GetNSections()<<std::endl;
        do{
        // now we have to find a section

         PolyconeSection const & sec = polycone.GetSection(index);
        
         ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::DistanceToIn<Backend>(
                *sec.fSolid,
                Transformation3D(),
                p - Vector3D<Precision>(0,0,sec.fShift),
                v,
                stepMax,
                distance);
     // std::cout<<"dist="<<distance<<" index="<<index<<" p.z()="<<p.z()-sec.fShift<<std::endl;

        if (distance < kInfinity || !increment)
        break;
        index += increment;
      
       }
       while (index >= 0 && index < polycone.GetNSections());
        
       return;

       
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void DistanceToOut(
        UnplacedPolycone const &polycone,
        Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &dir,
        typename Backend::precision_v const &stepMax,
        typename Backend::precision_v &distance) {

    Vector3D<typename Backend::precision_v>  pn(point);

     if (polycone.GetNSections()==1)
    {
     const PolyconeSection& section = polycone.GetSection(0);
       
     ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::DistanceToOut<Backend>(
                *section.fSolid,
                point - Vector3D<Precision>(0,0,section.fShift),dir,stepMax,distance);
     return;
     }

    int indexLow = polycone.GetSectionIndex(point.z()-kTolerance);
    int indexHigh = polycone.GetSectionIndex(point.z()+kTolerance);
    int index = 0;
 
    if ( indexLow != indexHigh && (indexLow >= 0 ))
    { //we are close to Surface, section has to be identified
      const PolyconeSection& section = polycone.GetSection(indexLow);
      
      bool inside;
      Vector3D<Precision> localp;
      ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::Contains<Backend>(
                *section.fSolid,
                Transformation3D(),
                point - Vector3D<Precision>(0,0,section.fShift),
                localp,
                inside);
      if(!inside){index=indexHigh;}
      else{index=indexLow;}
    
    }
  else{index=indexLow;
    if(index<0)index=polycone.GetSectionIndex(point.z());
  } 
    if(index < 0 ){distance = 0.; return; }

  Precision totalDistance = 0.;
  Precision dist;
  int increment = (dir.z() > 0) ? 1 : -1;
  if (std::fabs(dir.z()) < kTolerance) increment = 0;
  int istep = 0; 
  do
    {
      const PolyconeSection& section = polycone.GetSection(index);
    
    if (totalDistance != 0||(istep < 2))
    {
      pn = point + (totalDistance ) * dir; // point must be shifted, so it could eventually get into another solid
      pn.z() -= section.fShift;
      typename Backend::int_v inside;
      ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::Inside<Backend>(
                *section.fSolid,
                Transformation3D(),
                pn,
                inside);
      if (inside == EInside::kOutside)
      {
        break;
      }
    }
    else pn.z() -= section.fShift;
    istep = istep+1;
    
   
   ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::DistanceToOut<Backend>(
                *section.fSolid,
                pn,dir,stepMax,dist);
   //std::cout<<"Section dist="<<dist<<" td="<<totalDistance<<" index="<<index<<std::endl;
   //Section Surface case   
   if(std::fabs(dist) < 0.5*kTolerance)
   { int index1 = index;
        if(( index > 0) && ( index < polycone.GetNSections()-1 )){index1 += increment;}
        else{
        if((index == 0) && ( increment > 0 ))index1 += increment;
        if((index == polycone.GetNSections()-1) && (increment<0 ))index1 += increment;
        }

        Vector3D<Precision> pte = point+(totalDistance+dist)*dir;
        const PolyconeSection& section1 = polycone.GetSection(index1);
        bool inside1;
        pte.z() -= section1.fShift;
        Vector3D<Precision> localp;
        ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::Contains<Backend>(
                *section1.fSolid,
                Transformation3D(),
                pte,
                localp,
                inside1);
        if (!inside1)
        {
         break;
        }
    }
   
    totalDistance += dist;
    index += increment;
   
  }
  while (index >= 0 && index < polycone.GetNSections());
 
  distance=totalDistance;
 
  return ;
   
}

    

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void SafetyToIn(UnplacedPolycone const &polycone,
                           Transformation3D const &transformation,
                           Vector3D<typename Backend::precision_v> const &point,
                           typename Backend::precision_v &safety) {


    Vector3D<typename Backend::precision_v> p = transformation.Transform<transCodeT,rotCodeT>(point);
   
    int index = polycone.GetSectionIndex(p.z());
    bool needZ = false;
    if(index < 0)
      {needZ = true;
       if(index == -1) index = 0;
       if(index == -2) index = polycone.GetNSections()-1;
      }
    Precision minSafety=0 ;//= SafetyFromOutsideSection(index, p);
    PolyconeSection const & sec = polycone.GetSection(index);
    if(needZ)
      {
      safety = ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::SafetyToInUSOLIDS<Backend,false>(
                *sec.fSolid,
                Transformation3D(),
                p - Vector3D<Precision>(0,0,sec.fShift));
      return;
      }
    else
       safety = ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::SafetyToInUSOLIDS<Backend,true>(
                *sec.fSolid,
                Transformation3D(),
                p - Vector3D<Precision>(0,0,sec.fShift));
      
    if (safety < kTolerance) return ;
    minSafety=safety;
    //std::cout<<" entering STI minSaf="<<minSafety<<" index="<<index<<std::endl;
    Precision zbase = polycone.fZs[index + 1];
    for (int i = index + 1; i < polycone.GetNSections(); ++i)
    {
     Precision dz = polycone.fZs[i] - zbase;
     if (dz >= minSafety) break;
     
     PolyconeSection const & sec = polycone.GetSection(i);
     safety = ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::SafetyToInUSOLIDS<Backend,false>(
                *sec.fSolid,
                Transformation3D(),
                p - Vector3D<Precision>(0,0,sec.fShift));
     //std::cout<<"Din="<<safety<<" i="<<i<<"dz="<<dz<<std::endl;
     if (safety < minSafety) minSafety = safety;
    }

    zbase = polycone.fZs[index - 1];
    for (int i = index - 1; i >= 0; --i)
    {
     Precision dz = zbase - polycone.fZs[i];
     if (dz >= minSafety) break;
     PolyconeSection const & sec = polycone.GetSection(i);
     safety = ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::SafetyToInUSOLIDS<Backend,false>(
                *sec.fSolid,
                Transformation3D(),
                p - Vector3D<Precision>(0,0,sec.fShift));
     //std::cout<<"Din-1="<<safety<<" i="<<i<<"dz="<<dz<<std::endl;
     if (safety < minSafety) minSafety = safety;
    }
    safety = minSafety;
    return ;
    /*
     if (!aAccurate)
    return enclosingCylinder->SafetyFromOutside(p);

  int index = GetSection(p.z);
  double minSafety = SafetyFromOutsideSection(index, p);
  if (minSafety < 1e-6) return minSafety;

  double zbase = fZs[index + 1];
  for (int i = index + 1; i <= fMaxSection; ++i)
  {
    double dz = fZs[i] - zbase;
    if (dz >= minSafety) break;
    double safety = SafetyFromOutsideSection(i, p);
    if (safety < minSafety) minSafety = safety;
  }

  zbase = fZs[index - 1];
  for (int i = index - 1; i >= 0; --i)
  {
    double dz = zbase - fZs[i];
    if (dz >= minSafety) break;
    double safety = SafetyFromOutsideSection(i, p);
    if (safety < minSafety) minSafety = safety;
  }
  return minSafety;
    */
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void SafetyToOut(UnplacedPolycone const &polycone,
                            Vector3D<typename Backend::precision_v> const &point,
                            typename Backend::precision_v &safety) {

    
 int index = polycone.GetSectionIndex(point.z());
   
    if (index < 0 ){ safety=0;return ;}
  
  
  
  PolyconeSection const & sec = polycone.GetSection(index);
  Vector3D<typename Backend::precision_v> p = point - Vector3D<Precision>(0,0,sec.fShift);
  safety = ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::SafetyToOutUSOLIDS<Backend,false>(
       *sec.fSolid,p);
  Precision minSafety =safety;
  //std::cout<<"Sout0="<<safety<<" index="<<index<<std::endl;
  if (minSafety == kInfinity) {safety = 0.;return ;}
  if (minSafety < kTolerance) {safety = 0.; return ;}

  Precision zbase = polycone.fZs[index + 1];
  for (int i = index + 1; i < polycone.GetNSections(); ++i)
  {
    Precision dz = polycone.fZs[i] - zbase;
    if (dz >= minSafety) break;
    PolyconeSection const & sec = polycone.GetSection(i);
    p = point - Vector3D<Precision>(0,0,sec.fShift);
    safety = ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::SafetyToInUSOLIDS<Backend,false>(
        *sec.fSolid,  Transformation3D(),p);
    if(safety < minSafety)minSafety =safety;
    // std::cout<<"Sout+1="<<safety<<" i="<<i<<"dz="<<dz<<std::endl;
  }

  if (index > 0)
  {
    zbase = polycone.fZs[index - 1];
    for (int i = index - 1; i >= 0; --i)
    {
    Precision dz = zbase - polycone.fZs[i];
    if (dz >= minSafety) break;
    PolyconeSection const & sec = polycone.GetSection(i);
    p = point - Vector3D<Precision>(0,0,sec.fShift);
    safety = ConeImplementation< translation::kIdentity, rotation::kIdentity, ConeTypes::UniversalCone>::SafetyToInUSOLIDS<Backend,false>(
         *sec.fSolid,  Transformation3D(),p);
    if(safety < minSafety)minSafety =safety;
    //std::cout<<"Sout-1="<<safety<<" i="<<i<<"dz="<<dz<<std::endl;
    }
  }
 
  safety=minSafety;
  return ;
   
}



}; // end PolyconeImplementation

}} // end namespace


#endif /* POLYCONEIMPLEMENTATION_H_ */
