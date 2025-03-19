/// \file NavStatePath.h
/// \author Sandro Wenzel (sandro.wenzel@cern.ch)
/// \date 12.03.2014

#ifndef VECGEOM_NAVIGATION_NAVSTATEPATH_H_
#define VECGEOM_NAVIGATION_NAVSTATEPATH_H_

#include "VecGeom/base/Config.h"

#include "VecGeom/base/VariableSizeObj.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/management/GeoManager.h"
#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/management/CudaManager.h"
#endif
#include "VecGeom/base/Global.h"

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

// the NavStateIndex type determines is used
// to calculate addresses of PlacedVolumes
// a short type should be used in case the number of PlacedVolumes can
// be counted with 16bits
// TODO: consider putting uint16 + uint32 types
#ifdef VECGEOM_USE_INDEXEDNAVSTATES
// typedef unsigned short NavStateIndex_t;
typedef unsigned long NavStateIndex_t;
#else
typedef VPlacedVolume const *NavStateIndex_t;
#endif

// helper functionality to convert from NavStateIndex_t to *PlacedVolumes and back
// the template abstraction also allows to go back to pointers as NavStateIndex_t
// via a template specialization
// T stands for NavStateIndex_t
template <typename T>
struct Index2PVolumeConverter {
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToPlacedVolume(T index)
  {
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
    // checking here for NVCC_DEVICE since the global variable globaldevicegeomgata::gCompact...
    // is marked __device__ and can only be compiled within device compiler passes
    assert(vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer != nullptr);
    return &vecgeom::globaldevicegeomdata::gCompactPlacedVolBuffer[index];
#else
#ifndef VECCORE_CUDA
    assert(vecgeom::GeoManager::gCompactPlacedVolBuffer == nullptr ||
           vecgeom::GeoManager::gCompactPlacedVolBuffer[index].id() == index);
    return &vecgeom::GeoManager::gCompactPlacedVolBuffer[index];
#else
    // this is the case when we compile with nvcc for host side
    // (failed previously due to undefined symbol vecgeom::cuda::GeoManager::gCompactPlacedVolBuffer)
    assert(false && "reached unimplement code");
    (void)index; // avoid unused parameter warning.
    return nullptr;
#endif
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static T ToIndex(VPlacedVolume const *pvol) { return pvol->id(); }
};

// template specialization when we directly save VPlacedVolume pointers into the NavStates
template <>
struct Index2PVolumeConverter<VPlacedVolume const *> {
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToPlacedVolume(VPlacedVolume const *pvol) { return pvol; }
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static VPlacedVolume const *ToIndex(VPlacedVolume const *pvol) { return pvol; }
};

/**
 * A class describing a current geometry state
 * likely there will be such an object for each particle/track currently treated.
 */
class NavStatePath : protected VariableSizeObjectInterface<NavStatePath, NavStateIndex_t>,
                     private Index2PVolumeConverter<NavStateIndex_t> {
public:
  using Value_t        = NavStateIndex_t;
  using Base_t         = VariableSizeObjectInterface<NavStatePath, Value_t>;
  using VariableData_t = VariableSizeObj<Value_t>;

private:
  friend Base_t;

  // Required by VariableSizeObjectInterface
  VECCORE_ATT_HOST_DEVICE
  VariableData_t &GetVariableData() { return fPath; }
  VECCORE_ATT_HOST_DEVICE
  const VariableData_t &GetVariableData() const { return fPath; }

  unsigned char
      fCurrentLevel; // value indicating the next free slot in the fPath array ( ergo the current geometry depth )
  // we choose unsigned char in order to save memory, thus supporting geometry depths up to 255 which seems large enough

  // a member to cache some state information across state usages
  // one particular example could be a calculated index for this state
  // if fCache == -1 it means the we have to recalculate it; otherwise we don't
  short fCache;

  bool fOnBoundary; // flag indicating whether track is on boundary of the "Top()" placed volume

#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
  // Cached transformation for top level
  bool fCacheM = true; // flag indicating whether the global matrix for the state is cached
  Transformation3D fTopTrans;
#endif

  // pointer data follows; has to be last
  VariableSizeObj<Value_t> fPath;

  // constructors and assignment operators are private
  // states have to be constructed using MakeInstance() function
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavStatePath(size_t nvalues);

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavStatePath(size_t new_size, NavStatePath &other)
      : fCurrentLevel(other.fCurrentLevel), fCache(-1), fOnBoundary(other.fOnBoundary),
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
        fCacheM(other.fCacheM), fTopTrans(other.fTopTrans),
#endif
        fPath(new_size, other.fPath)
  {
    // Raw memcpy of the content to another existing state.
    //
    // in case NavStatePath was a virtual class: change to
    // std::memcpy(other->DataStart(), DataStart(), DataSize());

    if (new_size > other.fPath.fN) {
      memset(fPath.GetValues() + other.fPath.fN, 0, new_size - other.fPath.fN);
    }
  }

  // some private management methods
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void InitInternalStorage();

private:
  // The data start should point to the address of the first data member,
  // after the virtual table
  // the purpose is probably for the Copy function
  const void *DataStart() const { return (const void *)&fCurrentLevel; }
  const void *ObjectStart() const { return (const void *)this; }
  void *DataStart() { return (void *)&fCurrentLevel; }
  void *ObjectStart() { return (void *)this; }

  // The actual size of the data for an instance, excluding the virtual table
  size_t DataSize() const { return SizeOf() + (size_t)ObjectStart() - (size_t)DataStart(); }

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
  using Base_t::SizeOfAlignAware;

  // Enumerate functions from converter which we want to use
  // ( without retyping of the struct name )
  using Index2PVolumeConverter<NavStateIndex_t>::ToIndex;
  using Index2PVolumeConverter<NavStateIndex_t>::ToPlacedVolume;

  // produces a compact navigation state object of a certain depth
  // the caller can give a memory address where the object will
  // be placed
  // the caller has to make sure that the size of the external memory
  // is >= sizeof(NavStatePath) + sizeof(VPlacedVolume*)*maxlevel
  //
  // Methods MakeInstance(), MakeInstanceAt(), MakeCopy() and MakeCopyAt() are provided by
  // VariableSizeObjectInterface

  VECCORE_ATT_HOST_DEVICE
  static NavStatePath *MakeInstance(int maxlevel)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return Base_t::MakeInstance(maxlevel + 1);
  }

  VECCORE_ATT_HOST_DEVICE
  static NavStatePath *MakeInstanceAt(int maxlevel, void *addr)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return Base_t::MakeInstanceAt(maxlevel + 1, addr);
  }

  // returns the size in bytes of a NavStatePath object with internal
  // path depth maxlevel
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstance(int maxlevel)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return VariableSizeObjectInterface::SizeOf(maxlevel + 1);
  }

  // returns the size in bytes of a NavStatePath object with internal
  // path depth maxlevel -- including space needed for padding to next aligned object
  // of same kind
  VECCORE_ATT_HOST_DEVICE
  static size_t SizeOfInstanceAlignAware(int maxlevel)
  {
    // MaxLevel is 'zero' based (i.e. maxlevel==0 requires one value)
    return VariableSizeObjectInterface::SizeOfAlignAware(maxlevel + 1);
  }

  VECCORE_ATT_HOST_DEVICE
  int GetObjectSize() const { return SizeOf(GetMaxLevel()); }

  VECCORE_ATT_HOST_DEVICE
  int SizeOf() const { return NavStatePath::SizeOfInstance(GetMaxLevel()); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  NavStatePath &operator=(NavStatePath const &rhs);

  // functions useful to "serialize" navigationstate
  // the Vector-of-Indices basically describes the path on the tree taken from top to bottom
  // an index corresponds to a daughter
  void GetPathAsListOfIndices(std::list<uint> &indices) const;
  void ResetPathFromListOfIndices(VPlacedVolume const *world, std::list<uint> const &indices);

  VECCORE_ATT_HOST_DEVICE
  void CopyTo(NavStatePath *other) const
  {
    // Raw memcpy of the content to another existing state.
    //
    // in case NavStatePath was a virtual class: change to
    // std::memcpy(other->DataStart(), DataStart(), DataSize());
    bool alloc = other->fPath.fSelfAlloc;
    // std::memcpy(other, this, this->SizeOf());
    // we only need to copy to relevant depth
    // GetCurrentLevel indicates the 'next' level, i.e. currentLevel==0 is empty
    // fCurrentLevel = maxlevel+1 is full
    // SizeOfInstance expect [0,maxlevel] and add +1 to its params
    std::memcpy(static_cast<void *>(other), this, NavStatePath::SizeOfInstance(this->GetCurrentLevel()));

    other->fPath.fSelfAlloc = alloc;
  }

  // copies a fixed and predetermined number of bytes
  // might be useful for specialized navigators which know the depth + SizeOf in advance
  // N is number of bytes to be copied and can be obtained by a prior call to constexpr NavStatePath::SizeOf( ... );
  template <size_t N>
  void CopyToFixedSize(NavStatePath *other) const
  {
    bool alloc = other->fPath.fSelfAlloc;
    for (size_t i = 0; i < N; ++i) {
      ((char *)other)[i] = ((char *)this)[i];
    }
    other->fPath.fSelfAlloc = alloc;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  ~NavStatePath();

  // what else: operator new etc...

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetMaxLevel() const { return fPath.fN - 1; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetCurrentLevel() const { return fCurrentLevel; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *GetLastExited() const
  { /*one beyond current*/
    // A.G this implementation was bugged, because it relies on the fact that this state object actually exited a deeper
    // path which may not be true - the path may be even set from outside. This implementation cannot provide this
    // functionality unless a specific data member is added and set via SetLastExited return (fCurrentLevel <
    // GetMaxLevel() - 1) ? ToPlacedVolume(fPath[fCurrentLevel]) : nullptr;
    return nullptr;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetLastExited() {}

  // better to use pop and push
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(VPlacedVolume const *);

  // Push a given daughter to the state
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Push(unsigned short);

  // a push version operating on IndexTypes
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void PushIndexType(NavStateIndex_t);

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *Top() const;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *At(int level) const { return ToPlacedVolume(fPath[level]); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Value_t ValueAt(int level) const { return fPath[level]; }

  // direct write access to the path
  // (no one should ever call this function unless you know what you are doing)
  // TODO: consider making this private + friend or so
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetValueAt(int level, Value_t v) { fPath[level] = v; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(Transformation3D &) const;

#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
  /* @brief Update the cached top matrix */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void UpdateTopMatrix(Transformation3D *top_matrix = nullptr);
#endif

  // returning a "delta" transformation that can transform
  // coordinates given in reference frame of this->Top() to the reference frame of other->Top()
  // simply with otherlocalcoordinate = delta.Transform( thislocalcoordinate )
  VECCORE_ATT_HOST_DEVICE
  void DeltaTransformation(NavStatePath const &other, Transformation3D & /* delta */) const;

  // VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &) const;

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GlobalToLocal(Vector3D<Precision> const &, int tolevel) const;

  VECCORE_ATT_HOST_DEVICE
  void TopMatrix(int tolevel, Transformation3D &) const;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Pop();

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  int Distance(NavStatePath const &) const;

  // returns a string representation of a (relative) sequence of operations/moves
  // that transforms this navigation state into the other navigation state
  // example:
  // state1 = /0/1/1/
  // state2 = /0/2/2/3
  // results in string
  // "/up/horiz/1/down/2/down/3" with 4 operations "up", "horiz", "down", "down"
  // the sequence of moves is the following
  // up: /0/1/1 --> /0/1/
  // horiz/1 : 0/1 --> /0/2 ( == /0/(1+1) )   "we are hopping from daughter 1 to 2 (which corresponds to a step of 1)"
  // down/2 : /0/2 --> /0/2/2   "going further down 2nd daughter"
  // down/3 : /0/2/2/3 --> /0/2/2/3  "going further down 2nd daughter"
  std::string RelativePath(NavStatePath const & /*other*/) const;

  // clear all information
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Clear();

  VECCORE_ATT_HOST_DEVICE
  void Print() const;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void Dump() const;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool HasSamePathAsOther(NavStatePath const &other) const
  {
    if (other.fCurrentLevel != fCurrentLevel) return false;
    for (int i = fCurrentLevel - 1; i >= 0; --i) {
      if (fPath[i] != other.fPath[i]) return false;
    }
    return true;
  }

  void printValueSequence(std::ostream & = std::cerr) const;

  // calculates a checksum along the path
  // can be used (as a quick criterion) to see whether 2 states are same
  unsigned long getCheckSum() const
  {
    unsigned long s = 0;
    for (int i = 0; i < fCurrentLevel; ++i) {
      s += (unsigned long)(ValueAt(i) + 1); // + 1 as offset otherwise may not distinguish TOP level and OUTSIDE level
    }
    return s;
  }

  /**
   * returns the number of FILLED LEVELS such that
   * state.GetNode( state.GetLevel() ) == state.Top()
   */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  unsigned char GetLevel() const { return fCurrentLevel - 1; }

  /**
    function returning whether the point (current navigation state) is outside the detector setup
  */
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOutside() const { return !(fCurrentLevel > 0); }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsOnBoundary() const { return fOnBoundary; }

#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool IsMatrixCached() const { return fCacheM; }
#endif

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetBoundaryState(bool b) { fOnBoundary = b; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  short GetCacheValue() const { return fCache; }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetCacheValue(short v) { fCache = v; }

}; // end of class

VECCORE_ATT_HOST_DEVICE
NavStatePath &NavStatePath::operator=(NavStatePath const &rhs)
{
  if (this != &rhs) {
    fCurrentLevel = rhs.fCurrentLevel;
    fCache        = rhs.fCache;
    fOnBoundary   = rhs.fOnBoundary;
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
    fCacheM   = rhs.fCacheM;
    fTopTrans = rhs.fTopTrans;
#endif
    // Use memcpy.  Potential truncation if this is smaller than rhs.
    fPath = rhs.fPath;
  }
  return *this;
}

// private implementation of standard constructor
VECCORE_ATT_HOST_DEVICE
NavStatePath::NavStatePath(size_t nvalues) : fCurrentLevel(0), fCache(-1), fOnBoundary(false), fPath(nvalues)
{
  // clear the buffer
  std::memset(fPath.GetValues(), 0, nvalues * sizeof(NavStateIndex_t));
}

VECCORE_ATT_HOST_DEVICE
NavStatePath::~NavStatePath() {}

VECCORE_ATT_HOST_DEVICE
void NavStatePath::Pop()
{
  if (fCurrentLevel > 0) {
    fCurrentLevel--;
    // note that we are not invalidating the "popped volume" here
    // in order to be able to query the last "exited volume" later
    fCache = -1;
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
    fCacheM = false;
#endif
  }
}

VECCORE_ATT_HOST_DEVICE
void NavStatePath::Clear()
{
  fCurrentLevel = 0;
  fOnBoundary   = false;
  fCache        = -1;
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
  fCacheM = true;
#endif
}

VECCORE_ATT_HOST_DEVICE
void NavStatePath::Push(VPlacedVolume const *v)
{
#ifdef DEBUG
  assert(fCurrentLevel < GetMaxLevel());
#endif
  fPath[fCurrentLevel++] = ToIndex(v);
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
  fCacheM = false;
#endif
}

// Allow pushing by child index
VECCORE_ATT_HOST_DEVICE
void NavStatePath::Push(unsigned short child)
{
  if (fCurrentLevel > 0) {
    auto top = ToPlacedVolume(fPath[fCurrentLevel - 1]);
#ifdef DEBUG
    assert(child < top->GetDaughters().size());
#endif
    fPath[fCurrentLevel++] = ToIndex(top->GetDaughters().operator[](child));
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
    fCacheM = false;
#endif
  }
}

VECCORE_ATT_HOST_DEVICE
void NavStatePath::PushIndexType(NavStateIndex_t v)
{
#ifdef DEBUG
  assert(fCurrentLevel < GetMaxLevel());
#endif
  fPath[fCurrentLevel++] = v;
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
  fCacheM = false;
#endif
}

VECCORE_ATT_HOST_DEVICE
VPlacedVolume const *NavStatePath::Top() const
{
  return (fCurrentLevel > 0) ? ToPlacedVolume(fPath[fCurrentLevel - 1]) : nullptr;
}

// calculates the global matrix to transform from global coordinates
// to the frame of the top volume in the state
// input: a reference to a transformation object ( which should be initialized to identity )
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void NavStatePath::TopMatrix(Transformation3D &global_matrix) const
{
  // this could be actually cached in case the path does not change ( particle stays inside a volume )
#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
  if (fCacheM) {
    global_matrix = fTopTrans;
    return;
  }
#endif
  for (int i = fCurrentLevel - 1; i > 0; --i) {
    global_matrix *= *(ToPlacedVolume(fPath[i])->GetTransformation());
  }
}

#ifdef VECGEOM_USE_CACHED_TRANSFORMATIONS
// Update the cached top matrix
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void NavStatePath::UpdateTopMatrix(Transformation3D *top_matrix)
{
  // Update cached top transformation
  if (fCacheM) return;
  if (top_matrix) {
    fTopTrans = *top_matrix;
  } else {
    // need to recompute
    fTopTrans.Clear();
    TopMatrix(fTopTrans);
  }
  // Flag matrix as cached
  fCacheM = true;
}
#endif

VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
void NavStatePath::Dump() const
{
  const unsigned int *ptr = (const unsigned int *)this;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
  printf("NavState::Dump(): data: %p(%zu) : %p(%zu) : %p(%zu)\n", (void *)&fCurrentLevel, sizeof(fCurrentLevel),
         (void *)&fOnBoundary, sizeof(fOnBoundary), (void *)&fPath, sizeof(fPath));
  for (unsigned int i = 0; i < 20; ++i) {
    printf("%p: ", (void *)ptr);
    for (unsigned int j = 0; j < 8; ++j) {
      printf(" %08x ", *ptr);
      ptr++;
    }
    printf("\n");
  }
#pragma GCC diagnostic pop
}

/**
 * encodes the geometry path as a concatenated string of ( Value_t ) present in fPath
 */
inline void NavStatePath::printValueSequence(std::ostream &stream) const
{
  for (int i = 0; i < fCurrentLevel; ++i) {
    stream << "/" << fPath[i] << "(" << At(i)->GetLabel() << ")";
  }
}

/**
 * calculates if other navigation state takes a different branch in geometry path or is on same branch
 * ( two states are on same branch if one can connect the states just by going upwards or downwards ( or do nothing ))
 */
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
int NavStatePath::Distance(NavStatePath const &other) const
{
  int lastcommonlevel = -1;
  int maxlevel        = Min(GetCurrentLevel(), other.GetCurrentLevel());

  //  algorithm: start on top and go down until paths split
  for (int i = 0; i < maxlevel; i++) {
    if (this->At(i) == other.At(i)) {
      lastcommonlevel = i;
    } else {
      break;
    }
  }

  return (GetCurrentLevel() - lastcommonlevel) + (other.GetCurrentLevel() - lastcommonlevel) - 2;
}

inline void NavStatePath::ConvertToGPUPointers()
{
#if !defined(VECCORE_CUDA) && defined(VECGEOM_ENABLE_CUDA) && !defined(VECGEOM_USE_INDEXEDNAVSTATES)
  for (int i = 0; i < fCurrentLevel; ++i) {
    auto *pvol = vecgeom::CudaManager::Instance().LookupPlaced(ToPlacedVolume(fPath[i])).GetPtr();
    fPath[i]   = ToIndex(pvol);
  }
#endif
}

inline void NavStatePath::ConvertToCPUPointers()
{
#if !defined(VECCORE_CUDA) && defined(VECGEOM_ENABLE_CUDA) && !defined(VECGEOM_USE_INDEXEDNAVSTATES)
  for (int i = 0; i < fCurrentLevel; ++i)
    fPath[i] = ToIndex(vecgeom::CudaManager::Instance().LookupPlacedCPUPtr((const void *)ToPlacedVolume(fPath[i])));
#endif
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#if defined(GCC_DIAG_POP_NEEDED)
#pragma GCC diagnostic pop
#undef GCC_DIAG_POP_NEEDED
#endif

#endif // VECGEOM_NAVIGATION_NAVSTATEPATH_H_
