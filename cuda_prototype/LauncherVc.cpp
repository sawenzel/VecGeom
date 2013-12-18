#include "LauncherVc.h"
#include "Kernel.h"

void LauncherVc::Contains(
    Vector3D<VcScalar> const &box_pos,
    Vector3D<VcScalar> const &box_dim,
    SOA3D_Vc_Float const &points,
    VcBool *output) {
  const int size = points.vectors();
  for (int i = 0; i < size; ++i) {
    output[i] = kernel::Contains<kVc>(box_pos, box_dim, points[i]);
  }
}