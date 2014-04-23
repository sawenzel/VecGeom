<% require "./options" %>

#ifndef VECGEOM_VOLUMES_UNPLACED<%=SHAPE%>_H_
#define VECGEOM_VOLUMES_UNPLACED<%=SHAPE%>_H_

#include <iostream>
#include "base/global.h"
#include "base/vector3d.h"
#include "volumes/unplaced_volume.h"

namespace VECGEOM_NAMESPACE {

class Unplaced<%=Shape%> : public VUnplacedVolume {

private:
  <%Member_definitions.each { |x| %> <%=x%>; 
  <%}%>
public:
  Unplaced<%=Shape%>(
    <%= Constructor_arguments.join(", ") %>
  ) : <%= Initialization_list.join(", ") %> {}

  VECGEOM_CUDA_HEADER_BOTH
  Unplaced<%=Shape%>(Unplaced<%=Shape%> const &other) : <%= Copy_constructor.join(", ") %> {}

  virtual int memory_size() const { return sizeof(*this); }

  #ifdef VECGEOM_CUDA
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
  #endif

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision volume() const {
    #warning "volume not implemented"
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               TransformationMatrix const *const matrix);

private:
  virtual VPlacedVolume* SpecializedVolume(
    LogicalVolume const *const volume,
    TransformationMatrix const *const matrix,
    const TranslationCode trans_code, const RotationCode rot_code) const;

  virtual void Print(std::ostream &os) const {
    os << "<%=Shape%> {" <<
                       <%= Print_arguments.join(" << , << ") %>  
                  << "}";
  }

};

} // namespace VEC_GEOM

#endif // VECGEOM_VOLUMES_UNPLACED<%=SHAPE%>_H_
