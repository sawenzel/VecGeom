#ifndef LAUNCHERVC_H
#define LAUNCHERVC_H

#include "LibraryVc.h"

class LauncherVc {

public:

  static void Contains(Vector3D<VcScalar> const& /*box_pos*/,
                       Vector3D<VcScalar> const& /*box_dim*/,
                       SOA3D_Vc_Float const& /*points*/,
                       VcBool* /*output*/);

private:

  LauncherVc() {}

};

#endif /* LAUNCHERVC_H */