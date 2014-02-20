#ifndef VECGEOM_VOLUMES_UNPLACEDVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACEDVOLUME_H_

#include <string>
#include "base/utilities.h"

namespace vecgeom {

class VUnplacedVolume {

public:

  friend std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol);

  virtual int byte_size() const =0;

private:

  virtual void print(std::ostream &os) const =0;

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDVOLUME_H_