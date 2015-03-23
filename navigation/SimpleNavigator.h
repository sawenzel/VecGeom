/// \file SimpleNavigator.h
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 12.03.2014

#ifndef SIMPLE_NAVIGATOR_H_
#define SIMPLE_NAVIGATOR_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/SOA3D.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/NavigationState.h"
#include "navigation/NavStatePool.h"

#ifdef VECGEOM_ROOT
#include "management/RootGeoManager.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#endif
#include <cassert>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class SimpleNavigator
{

public:
   /**
    * function to locate a global point in the geometry hierarchy
    * input: pointer to starting placed volume in the hierarchy and a global point, we also give an indication if we call this from top
    * output: pointer to the deepest volume in hierarchy that contains the particle and the navigation state
    *
    * scope: function to be used on both CPU and GPU
    */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   VPlacedVolume const *
   LocatePoint( VPlacedVolume const * /* volume */,
                Vector3D<Precision> const & /* globalpoint */,
                NavigationState & /* state (volume path) to be returned */,
                bool /*top*/) const;

  VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   SimpleNavigator(){}

   /**
    * function to locate a global point in the geometry hierarchy
    * input:  A local point in the referenceframe of the current deepest volume in the path,
    * the path itself which gets modified
    * output: path which may be modified
    *
    * scope: function to be used on both CPU and GPU
    */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   VPlacedVolume const *
   RelocatePointFromPath( Vector3D<Precision> const & /* localpoint */,
                          NavigationState & /* state to be modified */
                        ) const;


   /**
    * function to check whether global point has same path as given by currentstate
    * input:  A global point
    *         the path itself
    *         a new path object which is filled
    * output: yes or no
    * side effects: modifies newstate to be path of globalpoint
    *
    * scope: function to be used on both CPU and GPU
    */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   bool
   HasSamePath(
            Vector3D<Precision> const & /* globalpoint */,
            NavigationState const & /* currentstate */,
            NavigationState & /* newstate */
            ) const;


   /**
   * A function to navigate ( find next boundary and/or the step to do )
   */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   void FindNextBoundaryAndStep( Vector3D<Precision> const & /* global point */,
                          Vector3D<Precision> const & /* global dir */,
                          NavigationState const & /* currentstate */,
                          NavigationState & /* newstate */,
                          Precision const & /* proposed physical step */,
                          Precision & /*step*/
                         ) const;

   /**
    * A function to get back the safe distance; given a NavigationState object and a current global point
    * point
    */
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   Precision GetSafety( Vector3D<Precision> const & /*global_point*/,
               NavigationState const & /* currentstate */
   ) const;

   /**
    * A function to get back the safe distance for a basket (container) of points
    * with corresponding array of NavigationState objects
    *
    * particularities: the stateless nature of the SimpleNavigator requires that the caller
    * also provides a workspace Container3D ( to store intermediate results )
    *
    * the safety results will be made available in the output array
    *
    * The Container3D has to be either SOA3D or AOS3D
    */
   template <typename Container3D>
   VECGEOM_CUDA_HEADER_BOTH
   void GetSafeties( Container3D const & /*global_points*/,
                   NavigationState **  /*currentstates*/,
                   Container3D & /*workspace for localpoints*/,
                   Precision * /*safeties*/
   ) const;

   /**
    * Navigation interface for baskets; templates on Container3D which might be a SOA3D or AOS3D container
    * Note that the user has to provide a couple of workspace memories; This is the easiest way to make the navigator fully
    * threadsafe
    */
   template <typename Container3D>
   VECGEOM_CUDA_HEADER_BOTH
   void FindNextBoundaryAndStep(
         Container3D const & /*global point*/,
         Container3D const & /*global dirs*/,
         Container3D & /*workspace for localpoints*/,
         Container3D & /*workspace for localdirs*/,
         NavStatePool const&  /* array of pointers to NavigationStates for currentstates */,
         NavStatePool &  /* array of pointers to NabigationStates for outputstates */,
         Precision const * /* pSteps -- proposed steps */,
         Precision * /* safeties */,
         Precision * /* distances; steps */,
         int * /* workspace to keep track of nextdaughter ids */
        ) const;


   /**
    * A verbose function telling about possible hit targets and steps; starting from a navigation state
    * and a global point and direction ( we need to check for consistency ... ); mainly for debugging purposes
    */
   void InspectEnvironmentForPointAndDirection(
         Vector3D<Precision> const & /* global point */,
         Vector3D<Precision> const & /* global direction */,
         NavigationState const & /* current state */
   ) const;

   /**
    * A verbose function telling about safety calculation starting from a navigation state
    * and a global point; mainly for debugging purposes
    */
   void InspectSafetyForPoint(
           Vector3D<Precision> const & /* global point */,
           NavigationState const & /* current state */
      ) const;

}; // end of class declaration

VPlacedVolume const *
SimpleNavigator::LocatePoint( VPlacedVolume const * vol, Vector3D<Precision> const & point,
                       NavigationState & path, bool top ) const
{
   VPlacedVolume const * candvolume = vol;
   Vector3D<Precision> tmp(point);
   if( top )
   {
      assert( vol != NULL );
      candvolume = ( vol->UnplacedContains( point ) ) ? vol : 0;
   }
   if( candvolume )
   {
      path.Push( candvolume );
      Vector<Daughter> const * daughters = candvolume->GetLogicalVolume()->daughtersp();

      bool godeeper = true;
      while( godeeper && daughters->size() > 0)
      {
         godeeper = false;
         for(int i=0; i<daughters->size(); ++i)
         {
            VPlacedVolume const * nextvolume = (*daughters)[i];
            Vector3D<Precision> transformedpoint;

            if( nextvolume->Contains( tmp, transformedpoint ) )
            {
               path.Push( nextvolume );
               tmp = transformedpoint;
               candvolume =  nextvolume;
               daughters = candvolume->GetLogicalVolume()->daughtersp();
               godeeper=true;
               break;
            }
         }
      }
   }
   return candvolume;
}

VPlacedVolume const *
SimpleNavigator::RelocatePointFromPath( Vector3D<Precision> const & localpoint,
                              NavigationState & path ) const
{
   // idea: do the following:
   // ----- is localpoint still in current mother ? : then go down
   // if not: have to go up until we reach a volume that contains the
   // localpoint and then go down again (neglecting the volumes currently stored in the path)
   VPlacedVolume const * currentmother = path.Top();
   if( currentmother != NULL )
   {
        Vector3D<Precision> tmp = localpoint;
      // go up iteratively
      while( currentmother && ! currentmother->UnplacedContains( tmp ) )
      {
         path.Pop();
         Vector3D<Precision> pointhigherup = currentmother->GetTransformation()->InverseTransform( tmp );
         tmp=pointhigherup;
         currentmother=path.Top();
      }

      if(currentmother)
      {
         path.Pop();
         // may inline this
         return LocatePoint(currentmother, tmp, path, false);
      }
   }
   return currentmother;
}


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool
SimpleNavigator::HasSamePath( Vector3D<Precision> const & globalpoint,
                       NavigationState const & currentstate,
                       NavigationState & newstate ) const
{
   Transformation3D const & m = currentstate.TopMatrix();
   Vector3D<Precision> localpoint = m.Transform(globalpoint);
   newstate = currentstate;
   RelocatePointFromPath( localpoint, newstate );
   return currentstate.HasSamePathAsOther( newstate );
}


void
SimpleNavigator::FindNextBoundaryAndStep( Vector3D<Precision> const & globalpoint,
                                          Vector3D<Precision> const & globaldir,
                                          NavigationState     const & currentstate,
                                          NavigationState           & newstate,
                                          Precision           const & pstep,
                                          Precision                 & step
                                        ) const
{
   // this information might have been cached in previous navigators??
   Transformation3D const & m = const_cast<NavigationState &> ( currentstate ).TopMatrix();
   Vector3D<Precision> localpoint=m.Transform(globalpoint);
   Vector3D<Precision> localdir=m.TransformDirection(globaldir);

   VPlacedVolume const * currentvolume = currentstate.Top();
   int nexthitvolume = -1; // means mother

   if(currentvolume) step = currentvolume->DistanceToOut( localpoint, localdir, pstep );

   // NOTE: IF STEP IS NEGATIVE HERE, SOMETHING IS TERRIBLY WRONG. WE CAN TRY TO HANDLE THE SITUATION
   // IN TRYING TO PROPOSE THE RIGHT LOCATION IN NEWSTATE AND RETURN
   // I WOULD MUCH FAVOUR IF THIS WAS DONE OUTSIDE OF THIS FUNCTION BY THE USER
   if( step < 0. )
   {
       newstate = currentstate;
       RelocatePointFromPath( localpoint, newstate );
       return;
   }

   // iterate over all the daughter
   Vector<Daughter> const * daughters = currentvolume->GetLogicalVolume()->daughtersp();

   for(int d = 0; d<daughters->size(); ++d)
   {
      VPlacedVolume const * daughter = daughters->operator [](d);
      //    previous distance becomes step estimate, distance to daughter returned in workspace
      Precision ddistance = daughter->DistanceToIn( localpoint, localdir, step );

      nexthitvolume = (ddistance < step) ? d : nexthitvolume;
      step      = (ddistance < step) ? ddistance  : step;
   }

   // now we have the candidates
   // try
   newstate = currentstate;

   // is geometry further away than physics step?
   if(step > pstep)
   {
       // don't need to do anything
       step = pstep;
       newstate.SetBoundaryState( false );
       return;
   }
   newstate.SetBoundaryState( true );


   // TODO: this is tedious, please provide operators in Vector3D!!
   // WE SHOULD HAVE A FUNCTION "TRANSPORT" FOR AN OPERATION LIKE THIS
   Vector3D<Precision> newpointafterboundary = localdir;
   newpointafterboundary*=(step + 1e-9);
   newpointafterboundary+=localpoint;

   if( nexthitvolume != -1 ) // not hitting mother
   {
      // continue directly further down
      VPlacedVolume const * nextvol = daughters->operator []( nexthitvolume );
      Transformation3D const * trans = nextvol->GetTransformation();

      // this should be inlined here
      LocatePoint( nextvol, trans->Transform(newpointafterboundary), newstate, false );
      // newstate.Print();
   }
   else
   {
      // continue directly further up
      //LocateLocalPointFromPath_Relative_Iterative( newpointafterboundary, newpointafterboundaryinnewframe, outpath, globalm );
      RelocatePointFromPath( newpointafterboundary, newstate );
      // newstate.Print();
   }
}

// this is just the brute force method; need to see whether it makes sense to combine it into
// the FindBoundaryAndStep function
Precision SimpleNavigator::GetSafety(Vector3D<Precision> const & globalpoint,
                            NavigationState const & currentstate) const
{
   // this information might have been cached already ??
   Transformation3D const & m = const_cast<NavigationState &>(currentstate).TopMatrix();
   Vector3D<Precision> localpoint=m.Transform(globalpoint);

   // safety to mother
   VPlacedVolume const * currentvol = currentstate.Top();
   double safety = currentvol->SafetyToOut( localpoint );

   //assert( safety > 0 );

   // safety to daughters
   Vector<Daughter> const * daughters = currentvol->GetLogicalVolume()->daughtersp();
   int numberdaughters = daughters->size();
   for(int d = 0; d<numberdaughters; ++d)
   {
      VPlacedVolume const * daughter = daughters->operator [](d);
      double tmp = daughter->SafetyToIn( localpoint );
      safety = Min(safety, tmp);
   }
   return safety;
}


template<typename Container3D>
void SimpleNavigator::GetSafeties(Container3D const & globalpoints,
                                NavigationState ** states,
                                Container3D & workspaceforlocalpoints,
                                Precision * safeties ) const
{
    int np=globalpoints.size();
    //TODO: we have to care for the padding and tail
    workspaceforlocalpoints.resize(np);
    for( int i=0; i<np; ++i ){
       // TODO: we might be able to cache the matrices because some of the paths will be identical
       // need to have a quick way ( hash ) to compare paths
       Transformation3D const & m = states[i]->TopMatrix();
       workspaceforlocalpoints.set(i, m.Transform( globalpoints[i] ));
    }

    // vectorized calculation of safety to mother
    // we utilize here the fact that the Top() volumes of all NavigationStates
    // should be the same ( by definition of a basket )
    VPlacedVolume const * currentvol = states[0]->Top();

    currentvol->SafetyToOut( workspaceforlocalpoints, safeties );

    // safety to daughters; brute force but each function vectorized
    Vector<Daughter> const * daughters = currentvol->GetLogicalVolume()->daughtersp();
    int numberdaughters = daughters->size();
    for (int d = 0; d<numberdaughters; ++d) {
         VPlacedVolume const * daughter = daughters->operator [](d);
         daughter->SafetyToInMinimize( workspaceforlocalpoints, safeties );
    }
    return;
}

/**
 * Navigation interface for baskets; templates on Container3D which might be a SOA3D or AOS3D container
 */

template <typename Container3D>
void SimpleNavigator::FindNextBoundaryAndStep(
         Container3D const & globalpoints,
         Container3D const & globaldirs,
         Container3D       & localpoints,
         Container3D       & localdirs,
         NavStatePool const& currentstates,
         NavStatePool      & newstates,
         Precision const   * pSteps,
         Precision         * safeties,
         Precision         * distances,
         int               * nextnodeworkspace
        ) const
{
   // assuming that points and dirs are always global ones,
   // we need to transform to local coordinates first of all
   int np = globalpoints.size();
   localpoints.resize(np);
   localdirs.resize(np);
   for (int i=0;i<np;++i)
   {
      // TODO: we might be able to cache the matrices because some of the paths will be identical
      // need to have a quick way ( hash ) to compare paths
      Transformation3D const & m = currentstates[i]->TopMatrix();
      localpoints.set(i, m.Transform(globalpoints[i]));
      localdirs.set(i, m.TransformDirection(globaldirs[i]));
   }

   // attention here: the placed volume will of course differ for the particles;
   // however the distancetoout function and the daughterlist are the same for all particles
   VPlacedVolume const * currentvolume = currentstates[0]->Top();

   currentvolume->SafetyToOut( localpoints, safeties );
   // calculate distance to Boundary of current volume in vectorized way
   // also initialized nextnodeworkspace to -1 == hits or -2 stays in volume
   currentvolume->DistanceToOut( localpoints, localdirs,
           pSteps, distances, nextnodeworkspace );

   // iterate over all the daughter
   Vector<Daughter> const * daughters = currentvolume->GetLogicalVolume()->daughtersp();
   for (int daughterindex=0; daughterindex < daughters->size(); ++daughterindex)
   {
      VPlacedVolume const * daughter = daughters->operator [](daughterindex);

      daughter->SafetyToInMinimize( localpoints, safeties );
      // we call a version of the DistanceToIn function which is reductive:
      // it takes the existing data in distances as the proposed step
      // if we distance to this daughter is smaller than the step
      // both the distance and
      // the list of the best nextnode ids is updated
      daughter->DistanceToInMinimize( localpoints, localdirs,
              daughterindex, distances, nextnodeworkspace );
   }

   // now we can relocate
   // TODO: This work is often wasted since not used outside
   // TODO: consider moving this outside and calling it only when really finally needed
   // function needs to be implemented
   // VectorRelocateFromPaths( localpoints, localdirs,
   //      distances, nextnodeworkspace, const_cast<NavigationState const **>(currentstates),
   //      newstates, np );

   // do the relocation
   // now we have the candidates
   for( int i=0;i<np;++i )
   {
     *newstates[i] = *currentstates[i];

     // is geometry further away than physics step?
     if( distances[i]>pSteps[i] ) {
       // don't need to do anything
       distances[i] = pSteps[i];
       newstates[i]->SetBoundaryState( false );
       continue;
     }
     newstates[i]->SetBoundaryState( true );

     // TODO: this is tedious, please provide operators in Vector3D!!
     // WE SHOULD HAVE A FUNCTION "TRANSPORT" FOR AN OPERATION LIKE THIS
     Vector3D<Precision> newpointafterboundary = localdirs[i];
     newpointafterboundary*=(distances[i] + 1e-9);
     newpointafterboundary+=localpoints[i];

     if( nextnodeworkspace[i] > -1 ) // not hitting mother
     {
        // continue directly further down
        VPlacedVolume const * nextvol = daughters->operator []( nextnodeworkspace[i] );
        Transformation3D const * trans = nextvol->GetTransformation();

        // this should be inlined here
        LocatePoint( nextvol,
                trans->Transform(newpointafterboundary), *newstates[i], false );
     }
     else
     {
        // continue directly further up
        RelocatePointFromPath( newpointafterboundary, *newstates[i] );
     }

   } // end loop for relocation
}

} } // End global namespace

#endif /* SIMPLE_NAVIGATOR_H_ */
