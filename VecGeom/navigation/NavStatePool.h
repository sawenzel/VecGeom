/*
 * NavStatePool.h
 *
 *  Created on: 14.11.2014
 *      Author: swenzel
 */

#ifndef NAVSTATEPOOL_H_
#define NAVSTATEPOOL_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Assert.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/management/Logger.h"
#include <stdexcept>

#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/management/CudaManager.h"
#endif
#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Interface.h"
#endif

// a fixed (runtime) size  "array" or contiguous
// memory pool of navigation states
// testing some ideas to copy to gpu
// it is supposed to be long-lived ( it has some initialization time overhead because it allocates the
// GPU pointer at startup

#include <iostream>
#include <fstream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(template <typename Type> class SOA3D;);
VECGEOM_HOST_FORWARD_DECLARE(class NavStatePool;);
VECGEOM_DEVICE_FORWARD_DECLARE(class NavStatePool;);
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
VECCORE_FORCE_INLINE
NavigationState const *GetNavigationState(int i, const char *buffer, int depth)
{
  return reinterpret_cast<NavigationState const *>(buffer + NavigationState::SizeOfInstanceAlignAware(depth) * i);
}

VECCORE_ATT_HOST_DEVICE
VECCORE_FORCE_INLINE
NavigationState *GetNavigationState(int i, char *buffer, int depth)
{
  return reinterpret_cast<NavigationState *>(buffer + NavigationState::SizeOfInstanceAlignAware(depth) * i);
}

class NavStatePoolView {
public:
  VECCORE_ATT_HOST_DEVICE
  NavStatePoolView(char *buffer, int depth, int capacity) : fCapacity(capacity), fDepth(depth), fBuffer(buffer) {}

  VECCORE_ATT_HOST_DEVICE
  NavigationState *operator[](int i)
  {
    VECGEOM_ASSERT(i < fCapacity);
    return GetNavigationState(i, fBuffer, fDepth);
  }

  VECCORE_ATT_HOST_DEVICE
  NavigationState const *operator[](int i) const
  {
    VECGEOM_ASSERT(i < fCapacity);
    return GetNavigationState(i, fBuffer, fDepth);
  }

  VECCORE_ATT_HOST_DEVICE
  int Capacity() const { return fCapacity; }

  VECCORE_ATT_HOST_DEVICE
  int Depth() const { return fDepth; }

  VECCORE_ATT_HOST_DEVICE
  int IsValid() const { return fBuffer && fCapacity > 0 && fDepth > 0; }

private:         // members
  int fCapacity; // Allocated size of the container.
  int fDepth;    // depth of the navigation objects to cover
  char *fBuffer; // the memory buffer in which we place states
}; // end class

class NavStatePool {

public:
  NavStatePool(int size, int depth)
      : fCapacity(size), fDepth(depth), fBuffer(new char[NavigationState::SizeOfInstanceAlignAware(depth) * size]),
        fGPUPointer(NULL)
  {

#ifdef VECGEOM_CUDA_INTERFACE
    VECGEOM_DEVICE_API_CALL(Malloc(&fGPUPointer, NavigationState::SizeOfInstanceAlignAware(depth) * size));
#endif
    // now create the states
    for (int i = 0; i < (int)fCapacity; ++i) {
      NavigationState::MakeInstanceAt(depth, fBuffer + NavigationState::SizeOfInstanceAlignAware(depth) * i);
    }
  }

  ~NavStatePool()
  {
#ifdef VECGEOM_CUDA_INTERFACE
    try {
      VECGEOM_DEVICE_API_CALL(Free(fGPUPointer));
    } catch (std::runtime_error const &e) {
      VECGEOM_LOG(error) << e.what();
    }
#endif
    delete[] fBuffer;
  }
#if !defined(VECCORE_CUDA) && defined(VECGEOM_ENABLE_CUDA)
  void CopyToGpu();
  void CopyFromGpu();
#endif

  // quick and dirty serialization and deserialization
  void ToFile(std::string filename) const
  {
#ifdef VECGEOM_USE_INDEXEDNAVSTATES
    std::ofstream outfile(filename, std::ios::binary);
    outfile.write(reinterpret_cast<const char *>(&fCapacity), sizeof(fCapacity));
    outfile.write(reinterpret_cast<const char *>(&fDepth), sizeof(fDepth));
    outfile.write(reinterpret_cast<char *>(fBuffer), fCapacity * NavigationState::SizeOfInstanceAlignAware(fDepth));
#else
    std::cerr << "serializing pointer based navstates not supported \n";
#endif
  }

  static void ReadDepthAndCapacityFromFile(std::string filename, int &cap, int &dep)
  {
    std::ifstream fin(filename, std::ios::binary);
    fin.read(reinterpret_cast<char *>(&cap), sizeof(cap));
    fin.read(reinterpret_cast<char *>(&dep), sizeof(dep));
  }

  // return number of elements read or -1 if failure
  int FromFile(std::string filename)
  {
#ifdef VECGEOM_USE_INDEXEDNAVSTATES
    // assumes existing NavStatePool object
    decltype(fCapacity) cap;
    decltype(fDepth) dep;
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) return -1;
    fin.read(reinterpret_cast<char *>(&cap), sizeof(cap));
    if (!fin) return -2;
    fin.read(reinterpret_cast<char *>(&dep), sizeof(dep));
    if (!fin) return -2;
    if (cap != fCapacity || dep != fDepth) std::cerr << " warning: reading from navstate with different size\n";
    fin.read(reinterpret_cast<char *>(fBuffer), fCapacity * NavigationState::SizeOfInstanceAlignAware(fDepth));
    if (!fin) return -3;
#else
    std::cerr << "serializing pointer based navstates not supported \n";
#endif
    return fCapacity;
  }

  VECCORE_ATT_HOST_DEVICE
  NavigationState *operator[](int i) { return GetNavigationState(i, fBuffer, fDepth); }

  VECCORE_ATT_HOST_DEVICE
  NavigationState const *operator[](int i) const { return GetNavigationState(i, fBuffer, fDepth); }

  // convert/init this to a plain NavigationState** array
  // so that array[0] points to the first state in the NavStatePool, etc
  // this method also allocates memory; array should be a nullptr initially
  // this is a convenience function
  VECCORE_ATT_HOST_DEVICE
  void ToPlainPointerArray(NavigationState const **&array) const
  {
    array = new NavigationState const *[fCapacity];
    for (int i = 0; i < fCapacity; ++i) {
      array[i] = (*this)[i];
    }
  }

  // dito for the non-const version
  VECCORE_ATT_HOST_DEVICE
  void ToPlainPointerArray(NavigationState **&array)
  {
    array = new NavigationState *[fCapacity];
    for (int i = 0; i < fCapacity; ++i) {
      array[i] = (*this)[i];
    }
  }

  void Print() const
  {
    for (int i = 0; i < fCapacity; ++i)
      (*this)[i]->Print();
  }

  void *GetGPUPointer() const { return fGPUPointer; }

  int capacity() const { return fCapacity; }

private: // protected methods
#ifdef VECGEOM_ENABLE_CUDA
         // This constructor used to build NavStatePool at the GPU.  BufferGPU
  VECCORE_ATT_DEVICE
  NavStatePool(int size, int depth, char *fBufferGPU)
      : fCapacity(size), fDepth(depth), fBuffer(fBufferGPU), fGPUPointer(NULL)
  {
  }
#endif

private:         // members
  int fCapacity; // the number of states in the pool
  int fDepth;    // depth of the navigation objects to cover
  char *fBuffer; // the memory buffer in which we place states

  // assume it keeps a GPU pointer directly
  // the target of the copy operation
  void *fGPUPointer;

}; // end class

// an implementation of the CopyOperation could be as follows
#if !defined(VECCORE_CUDA) && defined(VECGEOM_ENABLE_CUDA)
inline void NavStatePool::CopyToGpu()
{

  // modify content temporarily to convert CPU pointers to GPU pointers
  NavigationState *state;
  for (int i = 0; i < fCapacity; ++i) {
    state = operator[](i);
    state->ConvertToGPUPointers();
  }

  // we also have to fix the fPath pointers

  // copy
  vecgeom::CopyToGpu((void *)fBuffer, fGPUPointer, fCapacity * NavigationState::SizeOfInstanceAlignAware(fDepth));
  // CudaAssertError( cudaMemcpy(fGPUPointer, (void*)fBuffer, fCapacity*NavigationState::SizeOf(fDepth),
  // VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)) );

  // modify back pointers
  for (int i = 0; i < fCapacity; ++i) {
    state = operator[](i);
    state->ConvertToCPUPointers();
  }

  // now some kernel can be launched on GPU side
} // end CopyFunction

inline void NavStatePool::CopyFromGpu()
{
  // this does not work
  // modify content temporarily to convert CPU pointers to GPU pointers

  // std::cerr << "Starting to COPY" << std::endl;
  // std::cerr << "GPU pointer " << fGPUPointer << std::endl;
  vecgeom::CopyFromGpu(fGPUPointer, (void *)fBuffer, fCapacity * NavigationState::SizeOfInstanceAlignAware(fDepth));

  NavigationState *state;
  for (int i = 0; i < fCapacity; ++i) {
    state = operator[](i);
    state->ConvertToCPUPointers();
  }
} // end CopyFunction
#endif
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVSTATEPOOL_H_ */
