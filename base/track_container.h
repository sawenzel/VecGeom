/**
 * @file track_container.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_TRACKCONTAINER_H_
#define VECGEOM_BASE_TRACKCONTAINER_H_

#include "base/global.h"

namespace VECGEOM_NAMESPACE {

/**
 * @brief Base class for containers storing sets of three coordinates in a
 *        memory scheme determined by the deriving class.
 *		  Type can both ordinary pods as well as more complex types such SIMD vectors
 */
template <typename Type>
class TrackContainer {

private:

  int fillsize_; // number of "useful" data elements
  int size_; // total allocated size

protected:

  bool allocated_;

  TrackContainer(const unsigned size, const bool allocated)
      : size_(size), allocated_(allocated), fillsize_(0) {}

  virtual ~TrackContainer() {}

public:
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int size() const { return size_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  int fillsize() const { return fillsize_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void setfillsize(int s ) { fillsize_ = s; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void push_back( Vector3D<Type> const & vec )  { Set(fillsize_, vec); fillsize_++; }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Vector3D<Type> operator[](const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type& x(const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type const& x(const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type& y(const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type const& y(const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type& z(const int index) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Type const& z(const int index) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Set(const int index, const Type x, const Type y,
                   const Type z) =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Set(const int index, Vector3D<Type> const &vec) =0;

};

} // End global namespace

#endif // VECGEOM_BASE_TRACKCONTAINER_H_
