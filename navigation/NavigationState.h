/*
 * navigationstate.h
 *
 *  Created on: Mar 12, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_
#define VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_

#include <string>
#include <iostream>

#include "backend/backend.h"
#include "base/transformation3d.h"
#include "volumes/placed_volume.h"

#ifdef VECGEOM_ROOT
#include "management/rootgeo_manager.h"
#endif

class TGeoBranchArray;

namespace VECGEOM_NAMESPACE
{
/**
 * a class describing a current geometry state
 * likely there will be such an object for each
 * particle/track currently treated
 */
class NavigationState
{
private:
   int maxlevel_;
   int currentlevel_;
   VPlacedVolume const * * path_;
   mutable Transformation3D global_matrix_;

   // add other navigation state here, stuff like:
   bool onboundary_; // flag indicating whether track is on boundary of the "Top()" placed volume

   // some private management methods
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void InitInternalStorage();

public:
   // constructors and assignment operators
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   NavigationState( int );

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   NavigationState( NavigationState const & rhs );

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   NavigationState & operator=( NavigationState const & rhs );

#ifdef VECGEOM_ROOT
   TGeoBranchArray * ToTGeoBranchArray() const;
   NavigationState & operator=( TGeoBranchArray const & rhs );
#endif

   VECGEOM_INLINE
   ~NavigationState( );


   // what else: operator new etc...

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   int GetMaxLevel() const {return maxlevel_;}

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   int GetCurrentLevel() const {return currentlevel_;}

   // better to use pop and push
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void
   Push(VPlacedVolume const *);

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   VPlacedVolume const *
   Top() const;

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   VPlacedVolume const *
   At(int level) const {return path_[level];}

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   Transformation3D const &
   TopMatrix() const;

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   Vector3D<Precision>
   GlobalToLocal(Vector3D<Precision> const &);

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void Pop();

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   int Distance( NavigationState const & ) const;
//   int Distance(NavigationState const &) const;

   // clear all information
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void Clear();

   VECGEOM_INLINE
   void Print() const;

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   bool HasSamePathAsOther( NavigationState const & other ) const
   {
        if( other.currentlevel_ != currentlevel_ ) return false;
        for( int i= currentlevel_-1; i>=0; --i ){
            if( path_[i] != other.path_[i] ) return false;
        }
        return true;
   }

#ifdef VECGEOM_ROOT
   VECGEOM_INLINE
   void printVolumePath() const;

   /**
    * returns the number of FILLED LEVELS such that
    * state.GetNode( state.GetLevel() ) == state.Top()
    */
   VECGEOM_INLINE
   int GetLevel() const {return currentlevel_-1;}

   TGeoNode const * GetNode(int level) const {return
		   RootGeoManager::Instance().tgeonode( path_[level] );}
#endif

   /**
     function returning whether the point (current navigation state) is outside the detector setup
   */
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   bool IsOutside() const { return !(currentlevel_>0); }


   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   bool IsOnBoundary() const { return onboundary_; }

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void SetBoundaryState( bool b ) { onboundary_ = b; }

#ifdef VECGEOM_ROOT
   /**
    * function return the ROOT TGeoNode object which is equivalent to calling Top()
    * function included for convenience; to make porting Geant-V easier; we should eventually get rid of this function
    */
   VECGEOM_INLINE
   TGeoNode const * GetCurrentNode() const
   {
      return RootGeoManager::Instance().tgeonode(this->Top());
   }
#endif

   //void GetGlobalMatrixFromPath( Transformation3D *const m ) const;
   //Transformation3D const * GetGlobalMatrixFromPath() const;
};



NavigationState & NavigationState::operator=( NavigationState const & rhs )
{
   currentlevel_=rhs.currentlevel_;
   maxlevel_ = rhs.maxlevel_;
   onboundary_ = rhs.onboundary_;
   std::memcpy(path_, rhs.path_, sizeof(path_)*currentlevel_);
   return *this;
}


NavigationState::NavigationState( NavigationState const & rhs ) : maxlevel_(rhs.maxlevel_),
      currentlevel_(rhs.currentlevel_), onboundary_(rhs.onboundary_)
{
   InitInternalStorage();
   std::memcpy(path_, rhs.path_, sizeof(path_)*rhs.currentlevel_ );
}


// implementations follow
NavigationState::NavigationState( int maxlevel ) : maxlevel_(maxlevel), currentlevel_(0), global_matrix_(), onboundary_(0)
{
   InitInternalStorage();
}

void
NavigationState::InitInternalStorage()
{
   path_ = new VPlacedVolume const *[maxlevel_];
}


NavigationState::~NavigationState()
{
   delete[] path_;
}


void
NavigationState::Pop()
{
   if(currentlevel_ > 0){
       path_[--currentlevel_]=0;
   }
}

void
NavigationState::Clear()
{
   currentlevel_=0;
   onboundary_=false;
}

void
NavigationState::Push( VPlacedVolume const * v )
{
#ifdef DEBUG
   assert( currentlevel_ < maxlevel_ )
#endif
   path_[currentlevel_++]=v;
}

VPlacedVolume const *
NavigationState::Top() const
{
   return (currentlevel_ > 0 )? path_[currentlevel_-1] : 0;
}

VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
Transformation3D const &
NavigationState::TopMatrix() const
{
// this could be actually cached in case the path does not change ( particle stays inside a volume )
   global_matrix_.CopyFrom( *(path_[0]->transformation()) );
   for(int i=1;i<currentlevel_;++i)
   {
      global_matrix_.MultiplyFromRight( *(path_[i]->transformation()) );
   }
   return global_matrix_;
}

/**
 * function that transforms a global point to local point in reference frame of deepest volume in current navigation state
 * ( equivalent to using a global matrix )
 */
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision>
NavigationState::GlobalToLocal(Vector3D<Precision> const & globalpoint)
{
   Vector3D<Precision> tmp=globalpoint;
   Vector3D<Precision> current;
   for(int level=0;level<currentlevel_;++level)
   {
      Transformation3D const *m = path_[level]->transformation();
      current = m->Transform( tmp );
      tmp = current;
   }
   return tmp;
}

VECGEOM_INLINE
void NavigationState::Print() const
{
   std::cerr << "maxlevel " << maxlevel_ << std::endl;
   std::cerr << "currentlevel " << currentlevel_ << std::endl;
   std::cerr << "onboundary " << onboundary_ << std::endl;
   std::cerr << "deepest volume " << Top() << std::endl;
}


#ifdef VECGEOM_ROOT
VECGEOM_INLINE
/**
 * prints the path of the track as a verbose string ( like TGeoBranchArray in ROOT )
 * (uses internal root representation for the moment)
 */
void NavigationState::printVolumePath() const
{
   for(int i=0; i < currentlevel_; ++i)
   {
    std::cout << "/" << RootGeoManager::Instance().tgeonode( path_[i] )->GetName();
   }
   std::cout << "\n";
}
#endif

/**
 * calculates if other navigation state takes a different branch in geometry path or is on same branch
 * ( two states are on same branch if one can connect the states just by going upwards or downwards ( or do nothing ))
 */
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
int NavigationState::Distance( NavigationState const & other ) const
{
   int lastcommonlevel=0;
   int maxlevel = Max( GetCurrentLevel() , other.GetCurrentLevel() );

   //  algorithm: start on top and go down until paths split
   for(int i=0; i < maxlevel; i++)
   {
      VPlacedVolume const *v1 = this->path_[i];
      VPlacedVolume const *v2 = other.path_[i];
      if( v1 == v2 )
      {
         lastcommonlevel = i;
      }
      else
      {
         break;
      }
   }
   return (GetCurrentLevel()-lastcommonlevel) + ( other.GetCurrentLevel() - lastcommonlevel ) - 2;
}


}

#endif // VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_
