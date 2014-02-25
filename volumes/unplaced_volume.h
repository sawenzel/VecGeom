#ifndef VECGEOM_VOLUMES_UNPLACEDVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACEDVOLUME_H_

#include <string>
#include "base/utilities.h"

namespace vecgeom {

class VUnplacedVolume {

private:

  friend class CudaManager;

public:

  /**
   * Uses the virtual print method.
   * \sa print(std::ostream &ps)
   */
  friend std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol);

  /**
   * Should return the size of bytes of the deriving class. Necessary for
   * copying to the GPU.
   */
  virtual int byte_size() const =0;

  /**
   * Copies the deriving class to the specified preallocated memory on the GPU.
   * \param target Allocated memory on the GPU.
   */
  #ifdef VECGEOM_NVCC
  virtual void CopyToGpu(VUnplacedVolume *const target) const =0;
  #endif

private:

  /**
   * Print information about the deriving class.
   * \param os Outstream to stream information into.
   */
  virtual void print(std::ostream &os) const =0;

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDVOLUME_H_