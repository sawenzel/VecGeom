#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#ifdef VECGEOM_ENABLE_CUDA
#include "VecGeom/backend/cuda/Interface.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using Vec3D  = Vector3D<Precision>;
using Real_v = vecgeom::VectorBackend::Real_v;

// generic implementation for SamplePointOnSurface
Vector3D<Precision> VUnplacedVolume::SamplePointOnSurface() const
{
  Vector3D<Precision> surfacepoint;
  SOA3D<Precision> points(1);
  volumeUtilities::FillRandomPoints(*this, points);

  Vector3D<Precision> dir = volumeUtilities::SampleDirection();
  surfacepoint            = points[0] + DistanceToOut(points[0], dir) * dir;

  // VECGEOM_ASSERT( Inside(surfacepoint) == vecgeom::kSurface );
  return surfacepoint;
}

// trivial implementations for the interface functions
// (since we are moving to these interfaces only gradually)

// ----------------- Extent --------------------------------------------------------------------
// VECCORE_ATT_HOST_DEVICE
// void VUnplacedVolume::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
// {
// #ifndef VECCORE_CUDA
//   throw std::runtime_error("unimplemented function called");
// #endif
// }

// estimating the surface area by sampling
// based on the six-point method of G4
Precision VUnplacedVolume::EstimateSurfaceArea(int nStat) const
{
  static const Precision s2 = 1. / Sqrt(2.);
  static const Precision s3 = 1. / Sqrt(3.);

  // Predefined directions
  //
  static const Vec3D directions[64] = {
      Vec3D(0, 0, 0),      Vec3D(-1, 0, 0),    // (  ,  ,  ) ( -,  ,  )
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),    // ( +,  ,  ) (-+,  ,  )
      Vec3D(0, -1, 0),     Vec3D(-s2, -s2, 0), // (  , -,  ) ( -, -,  )
      Vec3D(s2, -s2, 0),   Vec3D(0, -1, 0),    // ( +, -,  ) (-+, -,  )

      Vec3D(0, 1, 0),      Vec3D(-s2, s2, 0), // (  , +,  ) ( -, +,  )
      Vec3D(s2, s2, 0),    Vec3D(0, 1, 0),    // ( +, +,  ) (-+, +,  )
      Vec3D(0, -1, 0),     Vec3D(-1, 0, 0),   // (  ,-+,  ) ( -,-+,  )
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),   // ( +,-+,  ) (-+,-+,  )

      Vec3D(0, 0, -1),     Vec3D(-s2, 0, -s2),   // (  ,  , -) ( -,  , -)
      Vec3D(s2, 0, -s2),   Vec3D(0, 0, -1),      // ( +,  , -) (-+,  , -)
      Vec3D(0, -s2, -s2),  Vec3D(-s3, -s3, -s3), // (  , -, -) ( -, -, -)
      Vec3D(s3, -s3, -s3), Vec3D(0, -s2, -s2),   // ( +, -, -) (-+, -, -)

      Vec3D(0, s2, -s2),   Vec3D(-s3, s3, -s3), // (  , +, -) ( -, +, -)
      Vec3D(s3, s3, -s3),  Vec3D(0, s2, -s2),   // ( +, +, -) (-+, +, -)
      Vec3D(0, 0, -1),     Vec3D(-s2, 0, -s2),  // (  ,-+, -) ( -,-+, -)
      Vec3D(s2, 0, -s2),   Vec3D(0, 0, -1),     // ( +,-+, -) (-+,-+, -)

      Vec3D(0, 0, 1),      Vec3D(-s2, 0, s2),   // (  ,  , +) ( -,  , +)
      Vec3D(s2, 0, s2),    Vec3D(0, 0, 1),      // ( +,  , +) (-+,  , +)
      Vec3D(0, -s2, s2),   Vec3D(-s3, -s3, s3), // (  , -, +) ( -, -, +)
      Vec3D(s3, -s3, s3),  Vec3D(0, -s2, s2),   // ( +, -, +) (-+, -, +)

      Vec3D(0, s2, s2),    Vec3D(-s3, s3, s3), // (  , +, +) ( -, +, +)
      Vec3D(s3, s3, s3),   Vec3D(0, s2, s2),   // ( +, +, +) (-+, +, +)
      Vec3D(0, 0, 1),      Vec3D(-s2, 0, s2),  // (  ,-+, +) ( -,-+, +)
      Vec3D(s2, 0, s2),    Vec3D(0, 0, 1),     // ( +,-+, +) (-+,-+, +)

      Vec3D(0, 0, -1),     Vec3D(-1, 0, 0),    // (  ,  ,-+) ( -,  ,-+)
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),    // ( +,  ,-+) (-+,  ,-+)
      Vec3D(0, -1, 0),     Vec3D(-s2, -s2, 0), // (  , -,-+) ( -, -,-+)
      Vec3D(s2, -s2, 0),   Vec3D(0, -1, 0),    // ( +, -,-+) (-+, -,-+)

      Vec3D(0, 1, 0),      Vec3D(-s2, s2, 0), // (  , +,-+) ( -, +,-+)
      Vec3D(s2, s2, 0),    Vec3D(0, 1, 0),    // ( +, +,-+) (-+, +,-+)
      Vec3D(0, -1, 0),     Vec3D(-1, 0, 0),   // (  ,-+,-+) ( -,-+,-+)
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),   // ( +,-+,-+) (-+,-+,-+)
  };

  // Get bounding box
  //
  Vec3D bmin, bmax;
  this->Extent(bmin, bmax);
  Vec3D bdim = bmax - bmin;

  // Define statistics and shell thickness
  //
  int npoints      = (nStat < 1000) ? 1000 : nStat;
  Precision coeff  = 0.5 / Cbrt(Precision(npoints));
  Precision eps    = coeff * bdim.Min(); // half thickness
  Precision twoeps = 2. * eps;
  Precision del    = 1.8 * eps; // six-point offset - should be more than sqrt(3.)

  // Enlarge bounding box by eps
  //
  bmin -= Vec3D(eps);
  bdim += Vec3D(twoeps);

  // Calculate surface area
  //
  int icount = 0;
  for (int i = 0; i < npoints; ++i) {
    Precision px = bmin.x() + bdim.x() * RNG::Instance().uniform();
    Precision py = bmin.y() + bdim.y() * RNG::Instance().uniform();
    Precision pz = bmin.z() + bdim.z() * RNG::Instance().uniform();
    Vec3D p(px, py, pz);
    EnumInside in  = this->Inside(p);
    Precision dist = 0;
    if (in == EInside::kInside) {
      if (this->SafetyToOut(p) >= eps) continue;
      int icase = 0;
      if (this->Inside(Vec3D(px - del, py, pz)) != EInside::kInside) icase += 1;
      if (this->Inside(Vec3D(px + del, py, pz)) != EInside::kInside) icase += 2;
      if (this->Inside(Vec3D(px, py - del, pz)) != EInside::kInside) icase += 4;
      if (this->Inside(Vec3D(px, py + del, pz)) != EInside::kInside) icase += 8;
      if (this->Inside(Vec3D(px, py, pz - del)) != EInside::kInside) icase += 16;
      if (this->Inside(Vec3D(px, py, pz + del)) != EInside::kInside) icase += 32;
      if (icase == 0) continue;
      Vec3D v = directions[icase];
      dist    = this->DistanceToOut(p, v);
      Vec3D n;
      this->Normal(p + v * dist, n);
      dist *= v.Dot(n);
    } else if (in == EInside::kOutside) {
      if (this->SafetyToIn(p) >= eps) continue;
      int icase = 0;
      if (this->Inside(Vec3D(px - del, py, pz)) != EInside::kOutside) icase += 1;
      if (this->Inside(Vec3D(px + del, py, pz)) != EInside::kOutside) icase += 2;
      if (this->Inside(Vec3D(px, py - del, pz)) != EInside::kOutside) icase += 4;
      if (this->Inside(Vec3D(px, py + del, pz)) != EInside::kOutside) icase += 8;
      if (this->Inside(Vec3D(px, py, pz - del)) != EInside::kOutside) icase += 16;
      if (this->Inside(Vec3D(px, py, pz + del)) != EInside::kOutside) icase += 32;
      if (icase == 0) continue;
      Vec3D v = directions[icase];
      dist    = this->DistanceToIn(p, v);
      if (dist == kInfLength) continue;
      Vec3D n;
      this->Normal(p + v * dist, n);
      dist *= -(v.Dot(n));
    }
    if (dist < eps) icount++;
  }
  return bdim.x() * bdim.y() * bdim.z() * icount / npoints / twoeps;
}

// estimating the cubic volume by sampling
// based on the method of G4
Precision VUnplacedVolume::EstimateCapacity(int nStat) const
{
  Precision epsilon = 1E-4;

  // limits
  if (nStat < 100) nStat = 100;

  Vector3D<Precision> lower, upper, offset;
  this->Extent(lower, upper);
  offset                        = 0.5 * (upper + lower);
  const Vector3D<Precision> dim = 0.5 * (upper - lower);

  int insidecounter = 0;
  for (int i = 0; i < nStat; i++) {
    auto p = offset + volumeUtilities::SamplePoint(dim);
    if (this->Contains(p)) insidecounter++;
  }
  return 8. * (dim[0] + epsilon) * (dim[1] + epsilon) * (dim[2] + epsilon) * insidecounter / nStat;
}

std::ostream &operator<<(std::ostream &os, VUnplacedVolume const &vol)
{
  vol.Print(os);
  return os;
}

#ifndef VECCORE_CUDA

VPlacedVolume *VUnplacedVolume::PlaceVolume(LogicalVolume const *const volume,
                                            Transformation3D const *const transformation,
                                            VPlacedVolume *const placement) const
{
  return SpecializedVolume(volume, transformation, placement);
}

VPlacedVolume *VUnplacedVolume::PlaceVolume(char const *const label, LogicalVolume const *const volume,
                                            Transformation3D const *const transformation,
                                            VPlacedVolume *const placement) const
{
  VPlacedVolume *const placed = PlaceVolume(volume, transformation, placement);
  placed->set_label(label);
  return placed;
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE
void VUnplacedVolume::CopyBBoxesToGpu(const std::vector<VUnplacedVolume const *> &volumes,
                                      const std::vector<DevicePtr<cuda::VUnplacedVolume>> &gpu_ptrs)
{
  VECGEOM_VALIDATE(volumes.size() == gpu_ptrs.size(), << "Unequal CPU/GPU vectors for copying bounding boxes.");
  // Copy boxes data in a contiguous array, box icrt starting at index 6*icrt
  std::vector<Precision> boxesData(6 * gpu_ptrs.size());
  int icrt = 0;
  for (auto vol : volumes) {
    Vector3D<Precision> amin, amax;
    vol->GetBBox(amin, amax);
    VECGEOM_ASSERT((amax - amin).Mag() > 0);
    for (unsigned int i = 0; i < 3; ++i) {
      boxesData[6 * icrt + i]     = amin[i];
      boxesData[6 * icrt + i + 3] = amax[i];
    }
    icrt++;
  }
  // Dispatch to the GPU interface helper
  CopyBBoxesToGpuImpl<cuda::VUnplacedVolume, DevicePtr<cuda::VUnplacedVolume>>(gpu_ptrs.size(), gpu_ptrs.data(),
                                                                               boxesData.data());
}
#endif

} // namespace VECGEOM_IMPL_NAMESPACE
#ifdef VECCORE_CUDA

namespace cxx {

template void CopyBBoxesToGpuImpl<cuda::VUnplacedVolume, DevicePtr<cuda::VUnplacedVolume>>(
    std::size_t, DevicePtr<cuda::VUnplacedVolume> const *, cuda::Precision *);

} // namespace cxx

#endif // VECCORE_CUDA

} // namespace vecgeom
