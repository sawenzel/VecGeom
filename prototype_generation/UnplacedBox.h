#ifndef VECGEOM_UNPLACEDBOX_H_
#define VECGEOM_UNPLACEDBOX_H_

#include "Global.h"
#include "UnplacedVolume.h"

class UnplacedBox : public UnplacedVolume {

private:

  double dimensions_[3];

public:

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(const double x, const double y, const double z)
      : dimensions_{x, y, z} {}

  ~UnplacedBox() {}

  VECGEOM_CUDA_HEADER_BOTH
  double const* dimensions() const { return dimensions_; }

  VECGEOM_CUDA_HEADER_BOTH
  double x() const { return dimensions_[0]; }

  VECGEOM_CUDA_HEADER_BOTH
  double y() const { return dimensions_[1]; }

  VECGEOM_CUDA_HEADER_BOTH
  double z() const { return dimensions_[2]; }

};

#endif // VECGEOM_UNPLACEDBOX_H_