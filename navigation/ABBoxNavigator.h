/*
 * ABBoxNavigator.h
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#ifndef ABBOXNAVIGATOR_H_
#define ABBOXNAVIGATOR_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/SimpleNavigator.h"
#include "navigation/NavigationState.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include <map>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// Singleton class for ABBox manager
// keeps a (centralized) map of volume pointers to vectors of aligned bounding boxes
// the alternative would to include such a thing into logical volumes
class ABBoxManager {
public:
    // scalar or vector vectors
    typedef Vector3D<Precision> ABBox_t;

    // use old style arrays here as std::vector has some problems
    // with Vector3D<kVc::Double_t>
    typedef ABBox_t * ABBoxContainer_t;

    typedef std::pair<int, double> BoxIdDistancePair_t;
    typedef std::vector<BoxIdDistancePair_t> HitContainer_t;

    // build an abstraction of sort to sort vectors and lists portably
    template<typename C, typename Compare> void sort(C & v, Compare cmp) {
        std::sort(v.begin(), v.end(), cmp);
    }

    struct HitBoxComparatorFunctor{
    bool operator()( BoxIdDistancePair_t const & left, BoxIdDistancePair_t const & right ) {
        return left.second < right.second;
      }
    };

    using FP_t = HitBoxComparatorFunctor;

private:
    ABBoxManager(){};
    std::map< LogicalVolume const *, ABBoxContainer_t > fVolToABBoxesMap;

public:
   // computes the aligned bounding box for a certain placed volume
   static void ComputeABBox( VPlacedVolume const * pvol, ABBox_t * lower, ABBox_t * upper );

   static ABBoxManager & Instance() {
    static ABBoxManager instance;
        return instance;
    }

    // initialize ABBoxes for a certain logical volume
    // very first version that just creates as many boxes as there are daughters
    // in reality we might have a lot more boxes than daughters (but not less)
    void InitABBoxes( LogicalVolume const * lvol );

    // remove the boxes from the list
    void RemoveABBoxes( LogicalVolume const * lvol);

    // returns the Container for a given logical volume or NULL if
    // it does not exist
    ABBoxContainer_t GetABBoxes( LogicalVolume const * lvol, int & size ) {
       size = lvol->daughtersp()->size();
       return fVolToABBoxesMap[lvol];
    }


};


// A navigator using aligned bounding box = ABBox (hierarchies) to quickly find
// potential hit targets.
// This navigator goes into the direction of "voxel" navigators used in Geant4
// and ROOT. Checking single-rays against a set of aligned bounding boxes can be done
// in a vectorized fashion.
class ABBoxNavigator
{

public:
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ABBoxNavigator(){}

  int GetHitCandidates( LogicalVolume const * lvol,
          Vector3D<Precision> const & point,
          Vector3D<Precision> const & dir,
          ABBoxManager::ABBoxContainer_t const & corners, int size,
          ABBoxManager::HitContainer_t & hitlist
  ) const;


  // convert index to physical daugher
  VPlacedVolume const * LookupDaughter( LogicalVolume const *lvol, int id ) const {
      return lvol->daughtersp()->operator []( id );
  }

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
}; // end of class declaration


void
ABBoxNavigator::FindNextBoundaryAndStep( Vector3D<Precision> const & globalpoint,
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

   // StepType st = kPhysicsStep; // physics or geometry step

   step = pstep;

   // do a quick and vectorized search using aligned bounding boxes
   // obtains a sorted container ( vector or list ) of hitboxstructs
   LogicalVolume const * currentlvol = currentstate.Top()->GetLogicalVolume();
   ABBoxManager::Instance().InitABBoxes( currentlvol );

   ABBoxManager::HitContainer_t hitlist;
   int size;
   ABBoxManager::ABBoxContainer_t bboxes =  ABBoxManager::Instance().GetABBoxes( currentlvol , size );
   GetHitCandidates( currentlvol,
                     localpoint,
                     localdir,
                     bboxes,
                     size, hitlist );

   // assumption: here hitlist is sorted in ascending distance order
   for( auto hitbox : hitlist )
   {
      VPlacedVolume const * candidate = LookupDaughter( currentlvol, hitbox.first );

      // only consider those hitboxes which are within potential reach of this step
      if( ! ( step < hitbox.second )) {
        Precision ddistance = candidate->DistanceToIn( localpoint, localdir, step );

        nexthitvolume = (ddistance < step) ? hitbox.first : nexthitvolume;
        step      = (ddistance < step) ? ddistance  : step;
      }
      else
      {
          break;
      }
   }

   // if nothing hit so far we will need to calculate distance to out
   if( nexthitvolume == -1 ){
     step = currentvolume->DistanceToOut( localpoint, localdir, pstep );

      // NOTE: IF STEP IS NEGATIVE HERE, SOMETHING IS TERRIBLY WRONG. WE CAN TRY TO HANDLE THE SITUATION
      // IN TRYING TO PROPOSE THE RIGHT LOCATION IN NEWSTATE AND RETURN
      // I WOULD MUCH FAVOUR IF THIS WAS DONE OUTSIDE OF THIS FUNCTION BY THE USER
     if( step < 0. )
     {
        newstate = currentstate;
        SimpleNavigator nav;
        nav.RelocatePointFromPath( localpoint, newstate );
        return;
     }
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
      VPlacedVolume const * nextvol = LookupDaughter( currentlvol, nexthitvolume );
      Transformation3D const * trans = nextvol->GetTransformation();

      SimpleNavigator nav;
      nav.LocatePoint( nextvol, trans->Transform(newpointafterboundary), newstate, false );
   }
   else
   {
      SimpleNavigator nav;
      nav.RelocatePointFromPath( newpointafterboundary, newstate );
   }
}

// this is just the brute force method; need to see whether it makes sense to combine it into
// the FindBoundaryAndStep function
Precision ABBoxNavigator::GetSafety(Vector3D<Precision> const & globalpoint,
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

inline
 int ABBoxNavigator::GetHitCandidates(
                         LogicalVolume const * lvol,
                         Vector3D<Precision> const & point,
                         Vector3D<Precision> const & dir,
                         ABBoxManager::ABBoxContainer_t const & corners, int size,
                         ABBoxManager::HitContainer_t & hitlist) const {

    Vector3D<Precision> invdir(1./dir.x(), 1./dir.y(), 1./dir.z());
    int vecsize = size;
    int hitcount = 0;
    int sign[3]; sign[0] = invdir.x() < 0; sign[1] = invdir.y() < 0; sign[2] = invdir.z() < 0;
    // interpret as binary number and do a switch statement
    // do a big switch statement here
   // int code = 2 << size[0] + 2 << size[1] + 2 << size[2];
    for( auto box = 0; box < vecsize; ++box ){
         double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel2<kScalar, double>(
            &corners[2*box],
            point,
           invdir,
           sign[0],sign[1],sign[2],
            0, vecgeom::kInfinity );
            if( distance < vecgeom::kInfinity ){
                hitcount++;
             hitlist.push_back( ABBoxManager::BoxIdDistancePair_t( box, distance) );
            }
        }

    //    switch( size[0] + size[1] + size[2] ){
//    case 0: {
//        for( auto box = 0; box < vecsize; ++box ){
//        double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel<kScalar,0,0,0>(
//           &corners[2*box],
//           point,
//           invdir,
//           0, vecgeom::kInfinity );
//           if( distance < vecgeom::kInfinity ) hitcount++;
//         }       break; }
//    case 3: {
//        for( auto box = 0; box < vecsize; ++box ){
//                double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel<kScalar,1,1,1>(
//                   &corners[2*box],
//                   point,
//                   invdir,
//                   0, vecgeom::kInfinity );
//                   if( distance < vecgeom::kInfinity ) hitcount++;
//                 }       break; }
//    default : std::cerr << "DEFAULT CALLED\n";
//    }
#ifdef INNERTIMER
    timer.Stop();
    std::cerr << "# CACHED hitting " << hitcount << "\n";
    std::cerr << "# CACHED timer " << timer.Elapsed() << "\n";
#endif
    return hitcount;
}
} } // End global namespace

#endif /* ABBOXNAVIGATOR_H_ */
