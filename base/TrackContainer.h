/// \file track_container.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_TRACKCONTAINER_H_
#define VECGEOM_BASE_TRACKCONTAINER_H_

#include "base/Global.h"

namespace VECGEOM_NAMESPACE {

/**
 * @brief Base class for containers storing sets of three coordinates in a
 *        memory scheme determined by the deriving class.
 *        Type can both ordinary PODs as well as more complex types such SIMD
 *        vectors.
 */
template <typename Type>
class TrackContainer {

protected:

  int memory_size_; // Maximum number of elements
  int size_; // Filled number of elements
  bool allocated_;

  VECGEOM_CUDA_HEADER_BOTH
  TrackContainer(const int memsize, const bool allocated)
      : memory_size_(memsize), size_(0), allocated_(allocated) {}

  virtual ~TrackContainer() {}

public:

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const { return size_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int memory_size() const { return memory_size_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void set_size(const int s) { size_ = s; }

};

} // End global namespace

#endif // VECGEOM_BASE_TRACKCONTAINER_H_
