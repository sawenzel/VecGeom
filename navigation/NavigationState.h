/// \file NavigationState.h
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 12.03.2014

#ifndef VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_
#define VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_

#include "backend/Backend.h"
#include "VariableSizeObj.h"
#include "base/Transformation3D.h"
#include "volumes/PlacedVolume.h"
#ifdef VECGEOM_CUDA
#include "management/CudaManager.h"
#endif
#include "base/Global.h"

#ifdef VECGEOM_ROOT
#include "management/RootGeoManager.h"
#endif

#include <iostream>
#include <string>

class TGeoBranchArray;

// gcc 4.8.2's -Wnon-virtual-dtor is broken and turned on by -Weffc++, we
// need to disable it for SOA3D

#if __GNUC__ < 3 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 8)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Weffc++"
#define GCC_DIAG_POP_NEEDED
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * a class describing a current geometry state
 * likely there will be such an object for each
 * particle/track currently treated
 */
class NavigationState : protected VecCore::VariableSizeObjectInterface<NavigationState, VPlacedVolume const *> {
public:
   using Value_t = VPlacedVolume const *;
   using Base_t = VecCore::VariableSizeObjectInterface<NavigationState, Value_t>;
   using VariableData_t = VecCore::VariableSizeObj<Value_t>;

private:
   friend Base_t;

   // Required by VariableSizeObjectInterface
   VECGEOM_CUDA_HEADER_BOTH
   VariableData_t &GetVariableData() { return fPath; }
   VECGEOM_CUDA_HEADER_BOTH
   const VariableData_t &GetVariableData() const { return fPath; }

   int fCurrentLevel;
   // add other navigation state here, stuff like:
   bool fOnBoundary; // flag indicating whether track is on boundary of the "Top()" placed volume
   mutable Transformation3D global_matrix_;

   // pointer data follows; has to be last
   VecCore::VariableSizeObj<VPlacedVolume const *> fPath;

   // constructors and assignment operators are private
   // states have to be constructed using MakeInstance() function
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   NavigationState(size_t nvalues);

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   NavigationState(size_t new_size, NavigationState &other )  : fCurrentLevel(other.fCurrentLevel), fOnBoundary(other.fOnBoundary), global_matrix_(other.global_matrix_), fPath(new_size, other.fPath)  {
       // Raw memcpy of the content to another existing state.
       //
       // in case NavigationState was a virtual class: change to
       // std::memcpy(other->DataStart(), DataStart(), DataSize());

      if (new_size > other.fPath.fN) {
         memset(fPath.GetValues()+other.fPath.fN,0,new_size - other.fPath.fN);
      }
   }

   // some private management methods
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void InitInternalStorage();

private:

   // The data start should point to the address of the first data member,
   // after the virtual table
  // the purpose is probably for the Copy function
  const void*  DataStart() const {return (const void*)&fCurrentLevel;}
  const void*  ObjectStart() const {return (const void*)this;}
  void*  DataStart() {return (void*)&fCurrentLevel;}
  void*  ObjectStart() {return (void*)this;}

     // The actual size of the data for an instance, excluding the virtual table
  size_t DataSize() const {
     return SizeOf() + (size_t)ObjectStart() - (size_t)DataStart();
  }


public:
    // replaces the volume pointers from CPU volumes in fPath
     // to the equivalent pointers on the GPU
     // uses the CudaManager to do so
    void ConvertToGPUPointers();

    // replaces the pointers from GPU volumes in fPath
    // to the equivalent pointers on the CPU
    // uses the CudaManager to do so
    void ConvertToCPUPointers();

  // Enumerate the part of the private interface, we want to expose.
  using Base_t::MakeCopy;
  using Base_t::MakeCopyAt;
  using Base_t::ReleaseInstance;
  using Base_t::SizeOf;

   // produces a compact navigation state object of a certain depth
   // the caller can give a memory address where the object will
   // be placed
   // the caller has to make sure that the size of the external memory
   // is >= sizeof(NavigationState) + sizeof(VPlacedVolume*)*maxlevel
   //
   // Both MakeInstance, MakeInstanceAt, MakeCopy and MakeCopyAT are provided by 
   // VariableSizeObjectInterface

   VECGEOM_CUDA_HEADER_BOTH
   static NavigationState *MakeInstance(int maxlevel) {
      // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
      return Base_t::MakeInstance(maxlevel+1);
   }

   static NavigationState *MakeInstanceAt(int maxlevel, void *addr) {
      // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
      return Base_t::MakeInstanceAt(maxlevel+1, addr);
   }

   // returns the size in bytes of a NavigationState object with internal
   // path depth maxlevel
   VECGEOM_CUDA_HEADER_BOTH
   static size_t SizeOfInstance(int maxlevel) {
      // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
      return VariableSizeObjectInterface::SizeOf( maxlevel + 1 );
   }

   VECGEOM_CUDA_HEADER_BOTH
   int GetObjectSize() const {
      return SizeOf( GetMaxLevel() );
   }

   VECGEOM_CUDA_HEADER_BOTH
   int SizeOf() const {
      return NavigationState::SizeOfInstance(GetMaxLevel());
   }

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   NavigationState & operator=( NavigationState const & rhs );

   VECGEOM_CUDA_HEADER_BOTH
   void CopyTo( NavigationState * other ) const {
      // Raw memcpy of the content to another existing state.
      //
      // in case NavigationState was a virtual class: change to
      // std::memcpy(other->DataStart(), DataStart(), DataSize());
      bool alloc = other->fPath.fSelfAlloc;
      std::memcpy(other, this, this->SizeOf());
      other->fPath.fSelfAlloc = alloc;
   }
 
#ifdef VECGEOM_ROOT
   TGeoBranchArray * ToTGeoBranchArray() const;
   NavigationState & operator=( TGeoBranchArray const & rhs );
#endif

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   ~NavigationState( );


   // what else: operator new etc...

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   int GetMaxLevel() const { return fPath.fN-1; }

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   int GetCurrentLevel() const {return fCurrentLevel;}

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
   At(int level) const {return fPath[level];}

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
   VECGEOM_CUDA_HEADER_BOTH
   void Print() const;

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void Dump() const;

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   bool HasSamePathAsOther( NavigationState const & other ) const
   {
        if( other.fCurrentLevel != fCurrentLevel ) return false;
        for( int i= fCurrentLevel-1; i>=0; --i ){
            if( fPath[i] != other.fPath[i] ) return false;
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
   int GetLevel() const {return fCurrentLevel-1;}

   TGeoNode const * GetNode(int level) const {return
           RootGeoManager::Instance().tgeonode( fPath[level] );}
#endif

   /**
     function returning whether the point (current navigation state) is outside the detector setup
   */
   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   bool IsOutside() const { return !(fCurrentLevel>0); }


   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   bool IsOnBoundary() const { return fOnBoundary; }

   VECGEOM_INLINE
   VECGEOM_CUDA_HEADER_BOTH
   void SetBoundaryState( bool b ) { fOnBoundary = b; }

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
}; // end of class

NavigationState & NavigationState::operator=( NavigationState const & rhs )
{
   if (this != &rhs) {
      fCurrentLevel=rhs.fCurrentLevel;
      fOnBoundary = rhs.fOnBoundary;
      // what about the matrix????

      // Use memcpy.  Potential truncation if this is smaller than rhs.
      fPath = rhs.fPath;
   }
   return *this;
}

/*
NavigationState::NavigationState( NavigationState const & rhs ) :
        fMaxlevel(rhs.fMaxlevel),
        fCurrentLevel(rhs.fCurrentLevel),
        fOnBoundary(rhs.fOnBoundary),
        global_matrix_() ,
        fPath(&fBuffer[0])
{
   InitInternalStorage();
   std::memcpy(fPath, rhs.fPath, sizeof(*fPath)*rhs.fCurrentLevel );
}
*/

// private implementation of standard constructor
NavigationState::NavigationState( size_t nvalues ) :
         fCurrentLevel(0),
         fOnBoundary(false),
         global_matrix_(),
         fPath(nvalues)
{
   // clear the buffer
   std::memset(fPath.GetValues(), 0, nvalues*sizeof(VPlacedVolume*));
}

  VECGEOM_CUDA_HEADER_BOTH
NavigationState::~NavigationState()
{
   
}


void
NavigationState::Pop()
{
   if(fCurrentLevel > 0){
       fPath[--fCurrentLevel]=0;
   }
}

void
NavigationState::Clear()
{
   fCurrentLevel=0;
   fOnBoundary=false;
}

void
NavigationState::Push( VPlacedVolume const * v )
{
#ifdef DEBUG
   assert( fCurrentLevel < GetMaxLevel() );
#endif
   fPath[fCurrentLevel++]=v;
}

VPlacedVolume const *
NavigationState::Top() const
{
   return (fCurrentLevel > 0 )? fPath[fCurrentLevel-1] : 0;
}


VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
Transformation3D const &
NavigationState::TopMatrix() const
{
// this could be actually cached in case the path does not change ( particle stays inside a volume )
   global_matrix_.CopyFrom( *(fPath[0]->GetTransformation()) );
   for(int i=1;i<fCurrentLevel;++i)
   {
      global_matrix_.MultiplyFromRight( *(fPath[i]->GetTransformation()) );
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
   for(int level=0;level<fCurrentLevel;++level)
   {
      Transformation3D const *m = fPath[level]->GetTransformation();
      current = m->Transform( tmp );
      tmp = current;
   }
   return tmp;
}

VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void NavigationState::Dump() const
{
   unsigned int* ptr = (unsigned int*)this;
   printf("NavState::Dump(): data: %p(%lu) : %p(%lu) : %p(%lu) : %p(%lu)\n", &fCurrentLevel, sizeof(fCurrentLevel),
          &fOnBoundary, sizeof(fOnBoundary), &global_matrix_, sizeof(global_matrix_), &fPath, sizeof(fPath));
   for(unsigned int i=0; i<20; ++i) {
      printf("%p: ", ptr);
      for(unsigned int i=0; i<8; ++i) {
         printf(" %08x ", *ptr);
         ptr++;
      }
      printf("\n");
   }
}

VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
void NavigationState::Print() const
{
   // printf("bool: fOnBoundary=%i %p (%l bytes)\n", fOnBoundary, static_cast<void*>(fOnBoundary), sizeof(bool));
   // printf("Transf3D: matrix (%l bytes)\n", sizeof(Transformation3D) );
   // printf("VariableSizeObj: fPath=%p (%l bytes)\n", fPath, sizeof(fPath));
   printf("NavState: Level(cur/max)=%i/%i,  onBoundary=%s, topVol=<%s>, this=%p\n",
          fCurrentLevel, GetMaxLevel(), (fOnBoundary?"true":"false"), (Top()? Top()->GetLabel().c_str():"NULL"), this );
   // std::cerr << "NavState: Level(cur/max)=" << fCurrentLevel <<'/'<< GetMaxLevel()
   //           <<" onBoundary="<< fOnBoundary
   //           <<" topVol="<< Top() <<" this="<< this
   //           << std::endl;
   // // std::cerr << "maxlevel " << fMaxlevel << std::endl;
   // // std::cerr << "currentlevel " << fCurrentLevel << std::endl;
   // // std::cerr << "onboundary " << fOnBoundary << std::endl;
   // // std::cerr << "deepest volume " << Top() << std::endl;
}


#ifdef VECGEOM_ROOT
VECGEOM_INLINE
/**
 * prints the path of the track as a verbose string ( like TGeoBranchArray in ROOT )
 * (uses internal root representation for the moment)
 */
void NavigationState::printVolumePath() const
{
   for(int i=0; i < fCurrentLevel; ++i)
   {
    std::cout << "/" << RootGeoManager::Instance().tgeonode( fPath[i] )->GetName();
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
      VPlacedVolume const *v1 = this->fPath[i];
      VPlacedVolume const *v2 = other.fPath[i];
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

inline
void NavigationState::ConvertToGPUPointers() {
#ifdef HAVENORMALNAMESPACE
#ifdef VECGEOM_CUDA
      for(int i=0;i<fCurrentLevel;++i){
         fPath[i] = (vecgeom::cxx::VPlacedVolume*) vecgeom::CudaManager::Instance().LookupPlaced( fPath[i] ).GetPtr();
      }
#endif
#endif
}

inline
void NavigationState::ConvertToCPUPointers() {
#ifdef HAVENORMALNAMESPACE
#ifdef VECGEOM_CUDA
       for(int i=0;i<fCurrentLevel;++i)
         fPath[i]=vecgeom::CudaManager::Instance().LookupPlacedCPUPtr( (const void*) fPath[i] );
#endif
#endif
}

} } // End global namespace


#if defined(GCC_DIAG_POP_NEEDED)
  #pragma GCC diagnostic pop
  #undef GCC_DIAG_POP_NEEDED
#endif

#endif // VECGEOM_NAVIGATION_NAVIGATIONSTATE_H_
