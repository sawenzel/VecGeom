#ifndef VECGEOM_VOLUMES_UNPLACEDVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACEDVOLUME_H_

#include <string>
#include "base/utilities.h"

namespace vecgeom {

class VUnplacedVolume {

public:

  VECGEOM_CUDA_HEADER_HOST
  friend std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol);

private:

  VECGEOM_CUDA_HEADER_HOST
  virtual void print(std::ostream &os) const =0;

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDVOLUME_H_