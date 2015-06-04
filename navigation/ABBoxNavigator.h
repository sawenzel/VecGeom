/*
 * ABBoxNavigator.h
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#ifndef ABBOXNAVIGATOR_H_
#define ABBOXNAVIGATOR_H_

#ifdef OFFLOAD_MODE
#pragma offload_attribute(push, target(mic))
#endif

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/SimpleNavigator.h"
#include "navigation/NavigationState.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"

#ifdef VECGEOM_VC
#include "backend/vc/Backend.h"
#endif

#include <map>
#include <vector>
#include <cassert>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// Singleton class for ABBox manager
// keeps a (centralized) map of volume pointers to vectors of aligned bounding boxes
// the alternative would be to include such a thing into logical volumes
class ABBoxManager {
public:
    // scalar or vector vectors
    typedef Vector3D<Precision> ABBox_t;
#ifdef VECGEOM_VC // just temporary typedef ---> will be removed with new backend structure
    typedef Vc::float_v Real_v;
    typedef Vc::float_m Bool_v;
    constexpr static unsigned int Real_vSize = Real_v::Size;
#else
    typedef float Real_v;
    typedef bool Bool_v;
    constexpr static unsigned int Real_vSize = 1;
#endif

    typedef float Real_t;
    typedef Vector3D<Real_v> ABBox_v;

    // use old style arrays here as std::vector has some problems
    // with Vector3D<kVc::Double_t>
    typedef ABBox_t * ABBoxContainer_t;
    typedef ABBox_v * ABBoxContainer_v;

    typedef std::pair<int, double> BoxIdDistancePair_t;
    typedef std::vector<BoxIdDistancePair_t> HitContainer_t;

    // build an abstraction of sort to sort vectors and lists portably
    template<typename C, typename Compare>
    static
    void sort(C & v, Compare cmp) {
        std::sort(v.begin(), v.end(), cmp);
    }

    struct HitBoxComparatorFunctor{
    bool operator()( BoxIdDistancePair_t const & left, BoxIdDistancePair_t const & right ) {
        return left.second < right.second;
      }
    };

    using FP_t = HitBoxComparatorFunctor;

private:
    std::map< LogicalVolume const *, ABBoxContainer_t > fVolToABBoxesMap;
    std::map< LogicalVolume const *, ABBoxContainer_v > fVolToABBoxesMap_v;

    // we have to make this thread safe
    HitContainer_t fAllocatedHitContainer;


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

    // doing the same for many logical volumes
    template<typename Container> void InitABBoxes( Container const & lvolumes ){
        for( auto lvol : lvolumes ){
            InitABBoxes( lvol );
        }
    }

    void InitABBoxesForCompleteGeometry( ){
        std::vector<LogicalVolume const *> logicalvolumes;
        GeoManager::Instance().getAllLogicalVolumes( logicalvolumes );
        InitABBoxes( logicalvolumes );
    }

    // remove the boxes from the list
    void RemoveABBoxes( LogicalVolume const * lvol);

    // returns the Container for a given logical volume or NULL if
    // it does not exist
    ABBoxContainer_t GetABBoxes( LogicalVolume const * lvol, int & size ) {
       size = lvol->daughtersp()->size();
       return fVolToABBoxesMap[lvol];
    }

    // returns the Container for a given logical volume or NULL if
    // it does not exist
    ABBoxContainer_v GetABBoxes_v( LogicalVolume const * lvol, int & size ) {
      int ndaughters = lvol->daughtersp()->size();
      int extra = (ndaughters % Real_vSize > 0) ? 1 : 0;
      size = ndaughters / Real_vSize + extra;
      return fVolToABBoxesMap_v[lvol];
    }

    HitContainer_t & GetAllocatedHitContainer(){
        return fAllocatedHitContainer;
    }

};

// output for hitboxes
template <typename stream>
stream & operator<<(stream & s, std::vector<ABBoxManager::BoxIdDistancePair_t> const & list){
    for(auto i : list){
        s << "(" << i.first << "," << i.second << ")" << " ";
    }
    return s;
}

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

  int GetHitCandidates_v( LogicalVolume const * lvol,
            Vector3D<Precision> const & point,
            Vector3D<Precision> const & dir,
            ABBoxManager::ABBoxContainer_v const & corners, int size,
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

//#define VERBOSE
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
#ifdef VERBOSE
    static int counter = 0;
    if( counter % 1 == 0 )
    std::cerr << counter << " " << globalpoint << " \n";

    counter++;
#endif

   Transformation3D m;
   currentstate.TopMatrix(m);
   Vector3D<Precision> localpoint=m.Transform(globalpoint);
   Vector3D<Precision> localdir=m.TransformDirection(globaldir);

   VPlacedVolume const * currentvolume = currentstate.Top();
   int nexthitvolume = -1; // means mother

   // StepType st = kPhysicsStep; // physics or geometry step
   step = currentvolume->DistanceToOut( localpoint, localdir, pstep );

   // NOTE: IF STEP IS NEGATIVE HERE, SOMETHING IS TERRIBLY WRONG. WE CAN TRY TO HANDLE THE SITUATION
   // IN TRYING TO PROPOSE THE RIGHT LOCATION IN NEWSTATE AND RETURN
   // I WOULD MUCH FAVOUR IF THIS WAS DONE OUTSIDE OF THIS FUNCTION BY THE USER
    if( step < 0. )
    {
       // TODO: instead of directly exiting we could see whether we hit a daughter
       // which is usally a logic thing to do
      // std::cerr << "negative DO\n";
     //  step = 0.;
     //  currentstate.CopyTo(&newstate);
     //  newstate.Pop();
     //  SimpleNavigator nav;
     //  nav.RelocatePointFromPath( localpoint, newstate );
      // return;
        step = kInfinity;
    }

   // if( step > 1E20 )
   //     std::cerr << "infinite DO\n";
   // TODO: compare steptoout and physics step and take minimum



   // do a quick and vectorized search using aligned bounding boxes
   // obtains a sorted container ( vector or list ) of hitboxstructs
   LogicalVolume const * currentlvol = currentstate.Top()->GetLogicalVolume();
  // ABBoxManager::Instance().InitABBoxes( currentlvol );

#ifdef VERBOSE
   std::cerr << " I am in " << currentlvol->GetLabel() << "\n";
#endif
   if( currentlvol->daughtersp()->size() > 0 ){
#ifdef VERBOSE
       std::cerr << " searching through " << currentlvol->daughtersp()->size() << " daughters\n";
#endif
     ABBoxManager::HitContainer_t & hitlist = ABBoxManager::Instance().GetAllocatedHitContainer();
//       hitlist.clear();
       int size;
//       ABBoxManager::ABBoxContainer_t bboxes1 =  ABBoxManager::Instance().GetABBoxes( currentlvol , size );
//       GetHitCandidates( currentlvol,
//                         localpoint,
//                         localdir,
//                         bboxes1,
//                        size, hitlist );
#ifdef VERBOSE
       int c1 = hitlist.size();
      std::cerr << hitlist << "\n";
#endif
       hitlist.clear();
       ABBoxManager::ABBoxContainer_v bboxes =  ABBoxManager::Instance().GetABBoxes_v( currentlvol , size );
            GetHitCandidates_v( currentlvol,
                          localpoint,
                          localdir,
                          bboxes,
                          size, hitlist );
#ifdef VERBOSE
            int c2 = hitlist.size();
        std::cerr << hitlist << "\n";
        std::cerr << " hitting scalar " << c1 << " vs vector " << c2 << "\n";
 if( c1 != c2 )
     std::cerr << "HUHU " << c1 << " " << c2;
        #endif

        // sorting the histlist
        ABBoxManager::sort( hitlist, ABBoxManager::HitBoxComparatorFunctor() );

        // assumption: here hitlist is sorted in ascending distance order
#ifdef VERBOSE
        std::cerr << " hitting " << hitlist.size() << " boundary boxes\n";
#endif
        for( auto hitbox : hitlist )
        {
             VPlacedVolume const * candidate = LookupDaughter( currentlvol, hitbox.first );

            // only consider those hitboxes which are within potential reach of this step
            if( ! ( step < hitbox.second )) {
            //      std::cerr << "checking id " << hitbox.first << " at box distance " << hitbox.second << "\n";
             if( hitbox.second < 0 ){
                bool checkindaughter = candidate->Contains( localpoint );
                if( checkindaughter == true ){
                    // need to relocate
                    step = 0;
                    nexthitvolume = hitbox.first;
                    // THE ALTERNATIVE WOULD BE TO PUSH THE CURRENT STATE AND RETURN DIRECTLY
                    break;
                }
            }
            Precision ddistance = candidate->DistanceToIn( localpoint, localdir, step );
#ifdef VERBOSE
            std::cerr << "distance to " << candidate->GetLabel() << " is " << ddistance << "\n";
#endif
            nexthitvolume = (ddistance < step) ? hitbox.first : nexthitvolume;
            step      = (ddistance < step) ? ddistance  : step;
        }
      else
      {
          break;
      }
   }
   }

   // now we have the candidates
   // try
   currentstate.CopyTo(&newstate);

   // is geometry further away than physics step?
   // not necessarily true
   if(step > pstep)
   {
       assert( true && "impossible state");
       // don't need to do anything
       step = pstep;
       newstate.SetBoundaryState( false );
       return;
   }
   newstate.SetBoundaryState( true );

   assert( step >= 0 && "step negative");

   if( step > 1E30 )
     {
      //std::cout << "WARNING: STEP INFINITY; should never happen unless outside\n";
           //InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );

           // set step to zero and retry one level higher
           step = 0;
           newstate.Pop();
           return;
      }

      if( step < 0. )
      {
        //std::cout << "WARNING: STEP NEGATIVE\n";
        //InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
         step = 0.;
      }

   // TODO: this is tedious, please provide operators in Vector3D!!
   // WE SHOULD HAVE A FUNCTION "TRANSPORT" FOR AN OPERATION LIKE THIS
   Vector3D<Precision> newpointafterboundary = localdir;
   newpointafterboundary*=(step + 1e-6);
   newpointafterboundary+=localpoint;

   if( nexthitvolume != -1 ) // not hitting mother
   {
      // continue directly further down
      VPlacedVolume const * nextvol = LookupDaughter( currentlvol, nexthitvolume );
      Transformation3D const * trans = nextvol->GetTransformation();

      SimpleNavigator nav;
      nav.LocatePoint( nextvol, trans->Transform(newpointafterboundary), newstate, false );
      assert( newstate.Top() != currentstate.Top() && " error relocating when entering ");
      return;
   }
   else // hitting mother
   {
      SimpleNavigator nav;
      nav.RelocatePointFromPath( newpointafterboundary, newstate );


      // can I push particle ?
      // int correctstep = 0;
      while( newstate.Top() == currentstate.Top() )
      {
     //     newstate.Print();
     //     step+=1E-6;
     //     SimpleNavigator nav;
     //     newstate.Clear();
     //     nav.LocatePoint( GeoManager::Instance().GetWorld(), globalpoint + (step)*globaldir, newstate, true );
     //     std::cerr << "correcting " << correctstep << " remaining dist to out "
      //              << currentvolume->DistanceToOut( localpoint + step*localdir, localdir, pstep )
      //              << " " << currentvolume->Contains( localpoint + step*localdir )
      //    << " " << currentvolume->SafetyToIn( localpoint + step*localdir )
      //    << " " << currentvolume->SafetyToOut( localpoint + step*localdir ) << "\n";
      //    currentvolume->PrintType();

      //    correctstep++;
       //   std::cerr << "Matrix error " << const_cast<NavigationState &> ( currentstate ).CalcTransformError( globalpoint, globaldir );
        newstate.Pop();
      }
//      if( newstate.Top() == currentstate.Top() )
//      {
//         std::cerr << "relocate failed; trying to locate from top for step " << step << "\n";
//         newstate.Clear();
//         SimpleNavigator nav;
//         nav.LocatePoint( GeoManager::Instance().GetWorld(), globalpoint + (step+1E-6)*globaldir, newstate, true );
//         //  std::cerr << "newstate top " << newstate.Top()->GetLabel() << "\n";
//      }
//      if( newstate.Top() == currentstate.Top() )
//      {
//         SimpleNavigator nav;
//         nav.InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
//      }
      assert( newstate.Top() != currentstate.Top() && " error relocating when leaving ");
   }
}

// this is just the brute force method; need to see whether it makes sense to combine it into
// the FindBoundaryAndStep function
Precision ABBoxNavigator::GetSafety(Vector3D<Precision> const & globalpoint,
                            NavigationState const & currentstate) const
{
   // this information might have been cached already ??
   Transformation3D m;
   currentstate.TopMatrix(m);
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

} } // End global namespace

#ifdef OFFLOAD_MODE
#pragma offload_attribute(pop)
#endif

#endif /* ABBOXNAVIGATOR_H_ */
