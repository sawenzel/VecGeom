/*
 * NavStatePool.h
 *
 *  Created on: 14.11.2014
 *      Author: swenzel
 */

#ifndef NAVSTATEPOOL_H_
#define NAVSTATEPOOL_H_

#include "base/Global.h"
#include "navigation/NavigationState.h"
#ifdef VECGEOM_CUDA
#include "management/CudaManager.h"
#endif
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif

// a fixed (runtime) size  "array" or contiguous
// memory pool of navigation states
// testing some ideas to copy to gpu
// it is supposed to be long-lived ( it has some initialization time overhead because it allocates the
// GPU pointer at startup

#include <iostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( template <typename Type> class SOA3D; )

inline namespace VECGEOM_IMPL_NAMESPACE {

class NavStatePool
{

public:
    NavStatePool( int size, int depth ) :
        fCapacity( size ),
        fDepth( depth ),
        fBuffer( new char[NavigationState::SizeOf( depth )*size ] ),
        fGPUPointer( NULL )
    {

#ifdef HAVENORMALNAMESPACE
// #pragma message "in namespace vecgeom"
#ifdef VECGEOM_CUDA
        vecgeom::CudaMalloc( &fGPUPointer, NavigationState::SizeOf(depth)*size );
#endif
#endif
        // now create the states
        for(int i=0;i<fCapacity;++i){
            NavigationState::MakeInstanceAt( depth, fBuffer + NavigationState::SizeOf(depth)*i );
        }
    }

    ~NavStatePool(){
        delete [] fBuffer;
    }
#ifdef HAVENORMALNAMESPACE
#ifdef VECGEOM_CUDA
    void CopyToGpu();
    void CopyFromGpu();
#endif
#endif

    VECGEOM_CUDA_HEADER_BOTH
    NavigationState * operator[]( int i ){
        return reinterpret_cast<NavigationState*> ( fBuffer + NavigationState::SizeOf(fDepth)*i );
    }

    VECGEOM_CUDA_HEADER_BOTH
    NavigationState const* operator[]( int i ) const{
        return reinterpret_cast<NavigationState const *> ( fBuffer + NavigationState::SizeOf(fDepth)*i );
    }

    void Print() const {
        for(int i=0;i<fCapacity;++i)
            (*this)[i]->Print();
    }

    void* GetGPUPointer() const {
       return fGPUPointer;
    }

private: // protected methods
#ifdef VECGEOM_CUDA
    // This constructor used to build NavStatePool at the GPU.  BufferGPU
    VECGEOM_CUDA_HEADER_DEVICE
    NavStatePool( int size, int depth, char* fBufferGPU ) :
        fCapacity( size ),
        fDepth( depth ),
        fBuffer( fBufferGPU ),
        fGPUPointer( NULL ) { }
#endif

private: // members
    int fCapacity; // the number of states in the pool
    int fDepth; // depth of the navigation objects to cover
    char* fBuffer; // the memory buffer in which we place states

    // assume it keeps a GPU pointer directly
    // the target of the copy operation
    void* fGPUPointer;

}; // end class


// an implementation of the CopyOperation could be as follows
#ifdef HAVENORMALNAMESPACE
#ifdef VECGEOM_CUDA
inline
void NavStatePool::CopyToGpu() {

    // modify content temporarily to convert CPU pointers to GPU pointers
    NavigationState * state;
    for(int i=0;i<fCapacity;++i){
        state = operator[]( i );
        state->ConvertToGPUPointers();
    }

    // we also have to fix the fPath pointers

    // copy
    vecgeom::CopyToGpu((void*)fBuffer, fGPUPointer, fCapacity*NavigationState::SizeOf(fDepth));
    //CudaAssertError( cudaMemcpy(fGPUPointer, (void*)fBuffer, fCapacity*NavigationState::SizeOf(fDepth), cudaMemcpyHostToDevice) );

    // modify back pointers
    for(int i=0;i<fCapacity;++i){
        state = operator[]( i );
        state->ConvertToCPUPointers();
    }

    // now some kernel can be launched on GPU side
} // end CopyFunction


inline
void NavStatePool::CopyFromGpu() {
    // this does not work
    // modify content temporarily to convert CPU pointers to GPU pointers

    std::cerr << "Starting to COPY" << std::endl;
    std::cerr << "GPU pointer " << fGPUPointer << std::endl;
    vecgeom::CopyFromGpu(fGPUPointer, (void*) fBuffer, fCapacity*NavigationState::SizeOf(fDepth));

    NavigationState * state;
    for(int i=0;i<fCapacity;++i){
        state = operator[]( i );
    state->ConvertToCPUPointers();
    }
} // end CopyFunction

#endif
#endif

} } // end Global namespace

#endif /* NAVSTATEPOOL_H_ */
