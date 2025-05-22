/*
 * UnplacedPolycone.cpp
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedPolycone.h"
#include "VecGeom/volumes/UnplacedCone.h"
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/volumes/PlacedPolycone.h"
#include "VecGeom/volumes/PlacedCone.h"
#include "VecGeom/volumes/SpecializedPolycone.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#ifndef VECCORE_CUDA
#include "VecGeom/base/RNG.h"
#include "VecGeom/volumes/UnplacedImplAs.h"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoPcon.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Polycone.hh"
#endif

#include <iostream>
#include <cstdio>
#include <vector>
#include "VecGeom/base/Vector.h"
#include "VecGeom/management/GeoManager.h"

#include "VecGeom/volumes/ReducedPolycone.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <>
UnplacedPolycone *Maker<UnplacedPolycone>::MakeInstance(Precision phistart, Precision deltaphi, int Nz,
                                                        Precision const *z, Precision const *rmin,
                                                        Precision const *rmax)
{
/*
 * NOTE : Since Polycone is internally using Cone, So for specializaton
 *        we are using ConeTypes as PolyconeTypes
 */
#if !defined(VECGEOM_NO_SPECIALIZATION) && !defined(VECCORE_CUDA)
  if (Nz == 2) {
    Precision dz = std::fabs(z[1] - z[0]) / 2.;
    std::cerr << "*****************************************************************************" << std::endl
              << "****** WARNING :: Trying to create Polycone with only one section... ********" << std::endl
              << "***************** This would Result in reduced performance ******************" << std::endl
              << "@@@@@@@@@@@@@ Suggestion : Create CONE instead of Polycone @@@@@@@@@@@@@@@@@@" << std::endl
              << "*****************************************************************************" << std::endl;
    return new SUnplacedImplAs<SUnplacedPolycone<ConeTypes::UniversalCone>, SUnplacedCone<ConeTypes::UniversalCone>>(
        rmin[0], rmax[0], rmin[1], rmax[1], dz, phistart, deltaphi);
  }

  /* Considering following cases for specialization of Polycon  */

  // 1) When the polycone is Completely NonHollow Polycone
  bool isCompletelyNonHollow = false;
  for (int i = 0; i < Nz; i++) {
    if (i == 0) {
      isCompletelyNonHollow = rmin[i] <= 0.;
    } else {
      isCompletelyNonHollow &= rmin[i] <= 0.;
    }
  }
  if (isCompletelyNonHollow) {
    if (deltaphi >= 2 * M_PI) {
      return new SUnplacedPolycone<ConeTypes::NonHollowCone>(phistart, 2 * M_PI, Nz, z, rmin, rmax);
    }
    if (deltaphi == M_PI) {
      return new SUnplacedPolycone<ConeTypes::NonHollowConeWithPiSector>(phistart, deltaphi, Nz, z, rmin, rmax);
    }
  }

  // 2) When the polycone is Completely Hollow Polycone
  bool isCompletelyHollow = false;
  for (int i = 0; i < Nz; i++) {
    if (i == 0) {
      isCompletelyHollow = rmin[i] > 0.;
    } else {
      isCompletelyHollow &= rmin[i] > 0.;
    }
  }
  if (isCompletelyHollow) {
    if (deltaphi >= 2 * M_PI) {
      return new SUnplacedPolycone<ConeTypes::HollowCone>(phistart, 2 * M_PI, Nz, z, rmin, rmax);
    }
    if (deltaphi == M_PI) {
      return new SUnplacedPolycone<ConeTypes::HollowConeWithPiSector>(phistart, deltaphi, Nz, z, rmin, rmax);
    }
  }

  /*
   * In case the polycone is the combination of Hollow and NonHollow Sections, then
   * we will pass it on to most general case ie. Universal.
   */
  return new SUnplacedPolycone<ConeTypes::UniversalCone>(phistart, deltaphi, Nz, z, rmin, rmax);

#else
  return new SUnplacedPolycone<ConeTypes::UniversalCone>(phistart, deltaphi, Nz, z, rmin, rmax);
#endif
}

template <>
UnplacedPolycone *Maker<UnplacedPolycone>::MakeInstance(Precision phistart, Precision deltaphi, int Nz,
                                                        Precision const *r, Precision const *z)
{
  return new SUnplacedPolycone<ConeTypes::UniversalCone>(phistart, deltaphi, Nz, r, z);
}

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedPolycone::ConvertToRoot(char const *label) const
{
  //	  UnplacedPolycone const *unplaced = GetUnplacedVolume();

  std::vector<Precision> rmin;
  std::vector<Precision> rmax;
  std::vector<Precision> z;
  auto original_param = fPolycone->fOriginal_parameters;
  if (original_param) {
    for (unsigned int i = 0; i < fPolycone->fNz; ++i) {
      z.push_back(original_param->fHZ_values[i]);
      rmin.push_back(original_param->fHRmin[i]);
      rmax.push_back(original_param->fHRmax[i]);
    }
  } else {
    fPolycone->ReconstructSectionArrays(z, rmin, rmax);
  }

  TGeoPcon *rootshape = new TGeoPcon(fPolycone->fStartPhi * kRadToDeg, fPolycone->fDeltaPhi * kRadToDeg, z.size());

  if (fPolycone->fNz != z.size()) std::cerr << "WARNING: Inconsistency in number of polycone sections\n";

  for (unsigned int i = 0; i < fPolycone->fNz; ++i)
    rootshape->DefineSection(i, z[i], rmin[i], rmax[i]);

  return rootshape;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedPolycone::ConvertToGeant4(char const *label) const
{
  // UnplacedPolycone const *unplaced = GetUnplacedVolume();

  std::vector<Precision> rmin;
  std::vector<Precision> rmax;
  std::vector<Precision> z;
  // unplaced->
  fPolycone->ReconstructSectionArrays(z, rmin, rmax);

  G4Polycone *g4shape =
      new G4Polycone("", fPolycone->fStartPhi, fPolycone->fDeltaPhi, z.size(), &z[0], &rmin[0], &rmax[0]);

  return g4shape;
}
#endif
#endif

VECCORE_ATT_HOST_DEVICE
void UnplacedPolycone::Reset()
{
  Precision phiStart = fPolycone->fOriginal_parameters->fHStart_angle;
  Precision *Z, *R1, *R2;
  int num = fPolycone->fOriginal_parameters->fHNum_z_planes; // fOriginalParameters-> NumZPlanes;
  Z       = new Precision[num];
  R1      = new Precision[num];
  R2      = new Precision[num];
  for (int i = 0; i < num; i++) {
    Z[i]  = fPolycone->fOriginal_parameters->fHZ_values[i]; // fOriginalParameters->fZValues[i];
    R1[i] = fPolycone->fOriginal_parameters->fHRmin[i];     // fOriginalParameters->Rmin[i];
    R2[i] = fPolycone->fOriginal_parameters->fHRmax[i];     // fOriginalParameters->Rmax[i];
  }

  fPolycone->Init(phiStart, fPolycone->fOriginal_parameters->fHOpening_angle, num, Z, R1, R2);
  delete[] R1;
  delete[] Z;
  delete[] R2;
}

// Alternative constructor, required for integration with Geant4.
/*
 * The modified definition make use of ReducedPolycone class, and provide more flexibility to
 * user to create Polycone by specifying RZ corners.
 *
 * User has to provide a valid contour formed by RZ corners.
 */
VECCORE_ATT_HOST_DEVICE
UnplacedPolycone::UnplacedPolycone(Precision phiStart, // initial phi starting angle
                                   Precision phiTotal, // total phi angle
                                   int numRZ,          // number corners in r,z space (must be an even number)
                                   Precision const *r, // r coordinate of these corners
                                   Precision const *z) // z coordinate of these corners
    : fPolycone(new PolyconeStruct<Precision>())

{
  fPolycone->fStartPhi = phiStart;
  fPolycone->fDeltaPhi = phiStart + phiTotal;

  Vector<Vector2D<Precision>> rzVect;
  Vector<Precision> rMinVect;
  Vector<Precision> rMaxVect;
  Vector<Precision> zVect;

  // Some safety check before trying to create standard polycone
  if (r[0] == r[numRZ - 1] && z[0] == z[numRZ - 1]) {
    numRZ--;
  }
  for (int i = 0; i < numRZ; i++) {
    if (i > 0) {
      if (r[i - 1] == r[i] && z[i - 1] == z[i]) {
        continue;
      }
    }
    rzVect.push_back(Vector2D<Precision>(r[i], z[i]));
  }

  ReducedPolycone p(rzVect);
  p.GetPolyconeParameters(rMinVect, rMaxVect, zVect);
  fPolycone->fNz  = zVect.size();
  Precision *rmin = new Precision[rMinVect.size()];
  Precision *rmax = new Precision[rMaxVect.size()];
  Precision *zarg = new Precision[zVect.size()];

  for (unsigned int i = 0; i < rMinVect.size(); i++) {
    rmin[i] = rMinVect[i];
    rmax[i] = rMaxVect[i];
    zarg[i] = zVect[i];
  }
  fPolycone->Init(phiStart, phiTotal, fPolycone->fNz, zarg, rmin, rmax);
  ComputeBBox();
  delete[] rmin;
  delete[] rmax;
  delete[] zarg;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedPolycone::Print() const
{
  printf("UnplacedPolycone { stPhi: %.2f, delPhi: %.2f, Nz: %d}\n", fPolycone->fStartPhi, fPolycone->fDeltaPhi,
         fPolycone->fNz);
  printf("\t------- %zu z planes follow ---------\n", fPolycone->fZs.size());
  for (size_t p = 0; p < fPolycone->fZs.size(); ++p) {
    printf("\t plane #%zu at z pos %lf\n", p, fPolycone->fZs[p]);
  }

  printf("\t------ %zu sections follow ----------\n", fPolycone->fSections.size());
  for (int s = 0; s < fPolycone->GetNSections(); ++s) {
    printf("\t section #%d, shift %lf\n\t ", s, fPolycone->fSections[s].fShift);
    fPolycone->fSections[s].fSolid.Print();
    printf("\n");
  }
  printf("\t-------------------------------------");
}
// VECCORE_ATT_HOST_DEVICE
void UnplacedPolycone::Print(std::ostream &os) const
{
  os << "UnplacedPolycone output to string not implemented -- calling Print() instead:\n";
  Print();
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedPolycone::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  typedef Vector3D<Precision> Vec_t;
  SolidMesh *sm = new SolidMesh();

  size_t nPlanes = GetNz();

  // copy each plane to revert values to their originals
  Precision *z    = new Precision[nPlanes];
  Precision *rmin = new Precision[nPlanes];
  Precision *rmax = new Precision[nPlanes];

  for (size_t i = 0; i < nPlanes; i++) {
    z[i]    = GetZAtPlane(i);
    rmin[i] = GetRminAtPlane(i);
    rmax[i] = GetRmaxAtPlane(i) - GetRminAtPlane(i) <= kConeTolerance
                  ? GetRminAtPlane(i)
                  : GetRmaxAtPlane(i); // rmin==rmax results in rmax+=1e-7, revert
  }

  sm->ResetMesh(2 * nPlanes * (nSegments + 1), 2 * (nPlanes - 1) * nSegments + 2 * nSegments + 2 * (nPlanes - 1));

  // fill vertex array
  Vec_t *vertices    = new Vec_t[2 * nPlanes * (nSegments + 1)];
  Precision phi      = GetStartPhi();
  Precision phi_step = GetDeltaPhi() / nSegments;
  size_t idx         = 0;
  size_t idx2        = nPlanes * (nSegments + 1);
  for (size_t i = 0; i < nPlanes; i++, phi = GetStartPhi()) {
    for (size_t j = 0; j <= nSegments; j++, phi += phi_step) {
      vertices[idx++]  = Vec_t(rmin[i] * std::cos(phi), rmin[i] * std::sin(phi), z[i]);
      vertices[idx2++] = Vec_t(rmax[i] * std::cos(phi), rmax[i] * std::sin(phi), z[i]);
    }
  }
  delete[] z, delete[] rmin, delete[] rmax;

  sm->SetVertices(vertices, 2 * nPlanes * (nSegments + 1));
  delete[] vertices;
  sm->TransformVertices(trans);

  // add polygons
  for (size_t i = 0, k = 0, l = k + nSegments + 1; i < nPlanes - 1; i++, k++, l++) {
    for (size_t j = 0; j < nSegments; j++, k++, l++) {
      sm->AddPolygon(4, {k, l, l + 1, k + 1}, true); // inner
    }
  }

  size_t offset = nPlanes * (nSegments + 1);
  for (size_t i = 0, k = offset, l = k + nSegments + 1; i < nPlanes - 1; i++, k++, l++) {
    for (size_t j = 0; j < nSegments; j++, k++, l++) {
      sm->AddPolygon(4, {k, k + 1, l + 1, l}, true); // outer
    }
  }

  for (size_t j = 0, k = j + offset; j < nSegments; j++, k++) {
    sm->AddPolygon(4, {j, j + 1, k + 1, k}, true); // lower
  }

  for (size_t j = 0, k = (nPlanes - 1) * (nSegments + 1), l = k + offset; j < nSegments; j++, k++, l++) {
    sm->AddPolygon(4, {k, l, l + 1, k + 1}, true); // upper
  }

  if (GetDeltaPhi() != kTwoPi) {
    for (size_t j = 0, k = 0, l = k + nSegments + 1; j < nPlanes - 1; j++, k += nSegments + 1, l += nSegments + 1) {
      sm->AddPolygon(4, {k, k + offset, l + offset, l}, true); // lat at sPhi
    }

    for (size_t j = 0, k = nSegments, l = k + nSegments + 1; j < nPlanes - 1;
         j++, k += nSegments + 1, l += nSegments + 1) {
      sm->AddPolygon(4, {k, l, l + offset, k + offset}, true); // lat at sPhi + dPhi
    }
  }

  return sm;
}
#endif

std::ostream &UnplacedPolycone::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Polycone\n"
     << " Parameters: \n"
     << "     N = number of Z-sections: " << fPolycone->fSections.size() << ", # Z-coords=" << fPolycone->fZs.size()
     << "\n"
     << "     z-coordinates:\n";

  uint nz = fPolycone->fZs.size();
  for (uint j = 0; j < (nz - 1) / 5 + 1; ++j) {
    os << "       [ ";
    for (uint i = 0; i < 5; ++i) {
      uint ind = 5 * j + i;
      if (ind < fPolycone->fNz) os << fPolycone->fZs[ind] << "; ";
    }
    os << " ]\n";
  }
  if (fPolycone->fDeltaPhi < kTwoPi) {
    os << "     Wedge starting angles: fSphi=" << fPolycone->fStartPhi * kRadToDeg << "deg, "
       << ", fDphi=" << fPolycone->fDeltaPhi * kRadToDeg << "deg\n";
  }

  size_t nsections = fPolycone->fSections.size();
  os << "\n    # cone sections: " << nsections << "\n";
  for (size_t i = 0; i < nsections; ++i) {
    auto const &subcone = fPolycone->fSections[i].fSolid;
    os << "     cone #" << i << " Rmin1=" << subcone.fRmin1 << " Rmax1=" << subcone.fRmax1
       << " Rmin2=" << subcone.fRmin2 << " Rmax2=" << subcone.fRmax2 << " HalfZ=" << subcone.fDz
       << " from z=" << fPolycone->fZs[i] << " to z=" << fPolycone->fZs[i + 1] << "mm\n";
  }
  os << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu() const
{
  return CopyToGpuImpl<SUnplacedPolycone<ConeTypes::UniversalCone>>();
}

DevicePtr<cuda::VUnplacedVolume> UnplacedPolycone::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const
{

  // idea: reconstruct defining arrays: copy them to GPU; then construct the UnplacedPolycon object from scratch
  // on the GPU
  std::vector<Precision> rmin, z, rmax;
  fPolycone->ReconstructSectionArrays(z, rmin, rmax);

  Precision *z_gpu_ptr    = AllocateOnGpu<Precision>(z.size() * sizeof(Precision));
  Precision *rmin_gpu_ptr = AllocateOnGpu<Precision>(rmin.size() * sizeof(Precision));
  Precision *rmax_gpu_ptr = AllocateOnGpu<Precision>(rmax.size() * sizeof(Precision));

  vecgeom::CopyToGpu(&z[0], z_gpu_ptr, sizeof(Precision) * z.size());
  vecgeom::CopyToGpu(&rmin[0], rmin_gpu_ptr, sizeof(Precision) * rmin.size());
  vecgeom::CopyToGpu(&rmax[0], rmax_gpu_ptr, sizeof(Precision) * rmax.size());

  int s = z.size();

  // attention here z.size() might be different than fNz due to compactification during Reconstruction
  DevicePtr<cuda::VUnplacedVolume> gpupolycon = CopyToGpuImpl<SUnplacedPolycone<ConeTypes::UniversalCone>>(
      gpu_ptr, fPolycone->fStartPhi, fPolycone->fDeltaPhi, s, z_gpu_ptr, rmin_gpu_ptr, rmax_gpu_ptr);

  // remove temporary space from GPU
  FreeFromGpu(z_gpu_ptr);
  FreeFromGpu(rmin_gpu_ptr);
  FreeFromGpu(rmax_gpu_ptr);

  return gpupolycon;
}

void UnplacedPolycone::CopyToGpu(std::vector<VUnplacedVolume const *> const &volumes,
                                 std::vector<DevicePtr<cuda::VUnplacedVolume>> const &devicePointers)
{
  // UnplacedPolycone(bool continuityOverAll, bool convexityPossible, bool equalRmax, Precision phistart,
  //                  Precision deltaphi, int nsections, int nz, Precision const *z, Precision const *rmin,
  //                  Precision const *rmax, size_t const buff_size, void const *buffer)
  const auto size = volumes.size();
  size_t size_buff{0};
  std::vector<size_t> sizes_buff(size, 0);
  std::vector<size_t> offsets_buff(size, 0);
  std::vector<Precision> startPhi, deltaPhi;
  std::vector<char> continuityOverAll, convexityPossible, equalRmax;
  std::vector<int> nZs, nSections;
  startPhi.reserve(size);
  deltaPhi.reserve(size);
  nZs.reserve(size);
  nSections.reserve(size);

  struct VarLengthData {
    std::vector<std::size_t> offsets;
    std::vector<Precision> z, rmin, rmax;
  } vld;
  struct RAIIDevPtr {
    DevicePtr<Precision> devPtr;
    RAIIDevPtr(std::size_t size) : devPtr() { devPtr.Allocate(size); }
    RAIIDevPtr(const RAIIDevPtr &) = delete;
    ~RAIIDevPtr() { devPtr.Deallocate(); }
  };

  for (unsigned int i = 0; i < size; ++i) {
    UnplacedPolycone const &polycone = static_cast<UnplacedPolycone const &>(*volumes[i]);
    assert(dynamic_cast<UnplacedPolycone const *>(volumes[i]));

    startPhi.push_back(polycone.fPolycone->fStartPhi);
    deltaPhi.push_back(polycone.fPolycone->fDeltaPhi);
    continuityOverAll.push_back(polycone.fPolycone->fContinuityOverAll);
    convexityPossible.push_back(polycone.fPolycone->fConvexityPossible);
    equalRmax.push_back(polycone.fPolycone->fEqualRmax);

    vld.offsets.push_back(vld.z.size());
    polycone.fPolycone->ReconstructSectionArrays(vld.z, vld.rmin, vld.rmax);
    nZs.push_back(vld.z.size() - vld.offsets.back());
    nSections.push_back(polycone.GetNSections());
    sizes_buff[i]   = AlignedAllocator::aligned_sizeof<PolyconeStruct<Precision>>(1, 0, polycone.GetNSections());
    offsets_buff[i] = size_buff;
    size_buff += sizes_buff[i];
  }
  // Assert that it's correct to have only one offset array for all variable-length arguments
  assert(vld.z.size() == vld.rmin.size() && vld.z.size() == vld.rmax.size() && vld.offsets.back() < vld.z.size());
  assert(startPhi.size() == size && deltaPhi.size() == size && nZs.size() == size);

  RAIIDevPtr zGPU{vld.z.size()}, rminGPU{vld.rmin.size()}, rmaxGPU{vld.rmax.size()};
  DevicePtr<char> bufferGPU;
  bufferGPU.Allocate(size_buff); // not freed, will need to register to CudaManager for cleanup

  zGPU.devPtr.ToDevice(vld.z.data(), vld.z.size());
  rminGPU.devPtr.ToDevice(vld.rmin.data(), vld.rmin.size());
  rmaxGPU.devPtr.ToDevice(vld.rmax.data(), vld.rmax.size());

  std::vector<Precision const *> zGPUPtrs{size}, rminGPUPtrs{size}, rmaxGPUPtrs{size};
  std::vector<char const *> bufferGPUPtrs{size};
  std::transform(vld.offsets.begin(), vld.offsets.end(), zGPUPtrs.begin(),
                 [&zGPU](std::size_t offset) { return zGPU.devPtr.GetPtr() + offset; });
  std::transform(vld.offsets.begin(), vld.offsets.end(), rminGPUPtrs.begin(),
                 [&rminGPU](std::size_t offset) { return rminGPU.devPtr.GetPtr() + offset; });
  std::transform(vld.offsets.begin(), vld.offsets.end(), rmaxGPUPtrs.begin(),
                 [&rmaxGPU](std::size_t offset) { return rmaxGPU.devPtr.GetPtr() + offset; });
  std::transform(offsets_buff.begin(), offsets_buff.end(), bufferGPUPtrs.begin(),
                 [&bufferGPU](std::size_t offset) { return bufferGPU.GetPtr() + offset; });

  /* Constructor we are targeting:
    UnplacedPolycone(bool continuityOverAll, bool convexityPossible, bool equalRmax, Precision phistart,
                     Precision deltaphi, int nsections, int nz, Precision const *z, Precision const *rmin,
                     Precision const *rmax, size_t const buff_size, void const *buffer)
   */
  ConstructManyOnGpu<cuda::SUnplacedPolycone<cuda::ConeTypes::UniversalCone>>(
      size, devicePointers.data(), reinterpret_cast<bool *>(continuityOverAll.data()),
      reinterpret_cast<bool *>(convexityPossible.data()), reinterpret_cast<bool *>(equalRmax.data()), startPhi.data(),
      deltaPhi.data(), nSections.data(), nZs.data(), zGPUPtrs.data(), rminGPUPtrs.data(), rmaxGPUPtrs.data(),
      sizes_buff.data(), (void const *const *)bufferGPUPtrs.data());
}

#endif // VECGEOM_CUDA_INTERFACE

Precision UnplacedPolycone::SurfaceArea() const
{
  const int numPlanes = GetNSections();

  PolyconeSection const &sec0 = GetSection(0);
  Precision totArea = (kPi * (sec0.fSolid.fRmax1 * sec0.fSolid.fRmax1 - sec0.fSolid.fRmin1 * sec0.fSolid.fRmin1));

  for (int i = 0; i < numPlanes; i++) {
    PolyconeSection const &sec = GetSection(i);

    Precision sectionArea =
        (sec.fSolid.fRmin1 + sec.fSolid.fRmin2) *
        std::sqrt((sec.fSolid.fRmin1 - sec.fSolid.fRmin2) * (sec.fSolid.fRmin1 - sec.fSolid.fRmin2) +
                  4. * sec.fSolid.fDz * sec.fSolid.fDz);

    sectionArea += (sec.fSolid.fRmax1 + sec.fSolid.fRmax2) *
                   std::sqrt((sec.fSolid.fRmax1 - sec.fSolid.fRmax2) * (sec.fSolid.fRmax1 - sec.fSolid.fRmax2) +
                             4. * sec.fSolid.fDz * sec.fSolid.fDz);

    sectionArea *= 0.5 * GetDeltaPhi();

    if (GetDeltaPhi() < kTwoPi) {
      sectionArea += std::fabs(2 * sec.fSolid.fDz) *
                     (sec.fSolid.fRmax1 + sec.fSolid.fRmax2 - sec.fSolid.fRmin1 - sec.fSolid.fRmin2);
    }
    totArea += sectionArea;
  }

  PolyconeSection const &secn = GetSection(numPlanes - 1);
  const auto last = kPi * (secn.fSolid.fRmax2 * secn.fSolid.fRmax2 - secn.fSolid.fRmin2 * secn.fSolid.fRmin2);
  totArea += last;

  return totArea;
}

bool UnplacedPolycone::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  bool valid = true;
  int index  = GetSectionIndex(point.z() - kTolerance);

  if (index < 0) {
    if (index == -1) norm = Vector3D<Precision>(0., 0., -1.);
    if (index == -2) norm = Vector3D<Precision>(0., 0., 1.);
    return valid;
  }
  PolyconeSection const &sec = GetSection(index);
  valid                      = sec.fSolid.Normal(point - Vector3D<Precision>(0, 0, sec.fShift), norm);

  // if point is within tolerance of a Z-plane between 2 sections, get normal from other section too
  if (size_t(index + 1) < fPolycone->fSections.size() && std::abs(point.z() - fPolycone->fZs[index + 1]) < kTolerance) {
    PolyconeSection const &sec2 = GetSection(index + 1);
    bool valid2                 = false;
    Vector3D<Precision> norm2;
    valid2 = sec2.fSolid.Normal(point - Vector3D<Precision>(0, 0, sec2.fShift), norm2);

    if (!valid && valid2) {
      norm  = norm2;
      valid = valid2;
    }

    // if both valid && valid2 true, norm and norm2 should be added...
    if (valid && valid2) {

      // discover exiting direction by moving point a bit (it would be good to have track direction here)
      // if(sec.fSolid.Contains(point + kTolerance*10*norm - Vector3D<Precision>(0, 0, sec.fShift))){

      //}
      bool c2;
      using CI = ConeImplementation<ConeTypes::UniversalCone>;
      // CI::Contains<Precision,false>(sec2.fSolid,
      //                            point + kTolerance * 10 * norm2 - Vector3D<Precision>(0, 0, sec2.fShift), c2);
      CI::Contains(sec2.fSolid, point + kTolerance * 10 * norm2 - Vector3D<Precision>(0, 0, sec2.fShift), c2);
      if (c2) {
        norm = norm2;
      } else {
        norm = norm + norm2;
        // but we might be in the interior of the polycone, and norm2=(0,0,-1) and norm1=(0,0,1) --> norm=(0,0,0)
        // quick fix:  set a normal pointing to the input point, but setting its z=0 (radial)
        if (norm.Mag2() < kTolerance) norm = Vector3D<Precision>(point.x(), point.y(), 0);
      }
    }
  }
  if (valid) norm /= norm.Mag();
  return valid;
}

#ifndef VECCORE_CUDA
/////////////////////////////////////////////////////////////////////////
//
// SamplePointOnSurface
//
// GetPointOnCone
//
// Auxiliary method for Get Point On Surface
//

Vector3D<Precision> UnplacedPolycone::GetPointOnCone(Precision fRmin1, Precision fRmax1, Precision fRmin2,
                                                     Precision fRmax2, Precision zOne, Precision zTwo,
                                                     Precision &totArea) const
{
#if (1)
  // declare working variables
  //
  Precision Aone, Atwo, Afive, phi, zRand, fDPhi, cosu, sinu;
  Precision rRand1, rmin, rmax, chose, rone, rtwo, qone, qtwo;
  Precision fDz = (zTwo - zOne) / 2., afDz = std::fabs(fDz);
  Vector3D<Precision> point, offset        = Vector3D<Precision>(0., 0., 0.5 * (zTwo + zOne));
  fDPhi = GetDeltaPhi();
  rone  = (fRmax1 - fRmax2) / (2. * fDz);
  rtwo  = (fRmin1 - fRmin2) / (2. * fDz);
  if (fRmax1 == fRmax2) {
    qone = 0.;
  } else {
    qone = fDz * (fRmax1 + fRmax2) / (fRmax1 - fRmax2);
  }
  if (fRmin1 == fRmin2) {
    qtwo = 0.;
  } else {
    qtwo = fDz * (fRmin1 + fRmin2) / (fRmin1 - fRmin2);
  }
  Aone    = 0.5 * fDPhi * (fRmax2 + fRmax1) * ((fRmin1 - fRmin2) * (fRmin1 - fRmin2) + (zTwo - zOne) * (zTwo - zOne));
  Atwo    = 0.5 * fDPhi * (fRmin2 + fRmin1) * ((fRmax1 - fRmax2) * (fRmax1 - fRmax2) + (zTwo - zOne) * (zTwo - zOne));
  Afive   = fDz * (fRmax1 - fRmin1 + fRmax2 - fRmin2);
  totArea = Aone + Atwo + 2. * Afive;

  phi  = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosu = std::cos(phi);
  sinu = std::sin(phi);

  if (GetDeltaPhi() >= kTwoPi) {
    Afive = 0;
  }
  chose = RNG::Instance().uniform(0., Aone + Atwo + 2. * Afive);
  if ((chose >= 0) && (chose < Aone)) {
    if (fRmax1 != fRmax2) {
      zRand = RNG::Instance().uniform(-1. * afDz, afDz);
      point = Vector3D<Precision>(rone * cosu * (qone - zRand), rone * sinu * (qone - zRand), zRand);
    } else {
      point = Vector3D<Precision>(fRmax1 * cosu, fRmax1 * sinu, RNG::Instance().uniform(-1. * afDz, afDz));
    }
  } else if (chose >= Aone && chose < Aone + Atwo) {
    if (fRmin1 != fRmin2) {
      zRand = RNG::Instance().uniform(-1. * afDz, afDz);
      point = Vector3D<Precision>(rtwo * cosu * (qtwo - zRand), rtwo * sinu * (qtwo - zRand), zRand);

    } else {
      point = Vector3D<Precision>(fRmin1 * cosu, fRmin1 * sinu, RNG::Instance().uniform(-1. * afDz, afDz));
    }
  } else if ((chose >= Aone + Atwo + Afive) && (chose < Aone + Atwo + 2. * Afive)) {
    zRand  = RNG::Instance().uniform(-afDz, afDz);
    rmin   = fRmin2 - ((zRand - fDz) / (2. * fDz)) * (fRmin1 - fRmin2);
    rmax   = fRmax2 - ((zRand - fDz) / (2. * fDz)) * (fRmax1 - fRmax2);
    rRand1 = std::sqrt(RNG::Instance().uniform(0., 1.) * (rmax * rmax - rmin * rmin) + rmin * rmin);
    point  = Vector3D<Precision>(rRand1 * std::cos(GetStartPhi()), rRand1 * std::sin(GetStartPhi()), zRand);
  } else {
    zRand  = RNG::Instance().uniform(-1. * afDz, afDz);
    rmin   = fRmin2 - ((zRand - fDz) / (2. * fDz)) * (fRmin1 - fRmin2);
    rmax   = fRmax2 - ((zRand - fDz) / (2. * fDz)) * (fRmax1 - fRmax2);
    rRand1 = std::sqrt(RNG::Instance().uniform(0., 1.) * (rmax * rmax - rmin * rmin) + rmin * rmin);
    point  = Vector3D<Precision>(rRand1 * std::cos(GetEndPhi()), rRand1 * std::sin(GetEndPhi()), zRand);
  }

  return point + offset;
#endif
}

//
// GetPointOnTubs
//
// Auxiliary method for GetPoint On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnTubs(Precision fRMin, Precision fRMax, Precision zOne, Precision zTwo,
                                                     Precision &totArea) const
{

  Precision xRand, yRand, zRand, phi, cosphi, sinphi, chose, aOne, aTwo, aFou, rRand, fDz, fSPhi, fDPhi;
  fDz   = std::fabs(0.5 * (zTwo - zOne));
  fSPhi = GetStartPhi();
  fDPhi = GetDeltaPhi();

  aOne    = 2. * fDz * fDPhi * fRMax;
  aTwo    = 2. * fDz * fDPhi * fRMin;
  aFou    = 2. * fDz * (fRMax - fRMin);
  totArea = aOne + aTwo + 2. * aFou;
  phi     = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi  = std::cos(phi);
  sinphi  = std::sin(phi);
  rRand   = fRMin + (fRMax - fRMin) * std::sqrt(RNG::Instance().uniform(0., 1.));

  if (GetDeltaPhi() >= 2 * kPi) aFou = 0;

  chose = RNG::Instance().uniform(0., aOne + aTwo + 2. * aFou);
  if ((chose >= 0) && (chose < aOne)) {
    xRand = fRMax * cosphi;
    yRand = fRMax * sinphi;
    zRand = RNG::Instance().uniform(-1. * fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  } else if ((chose >= aOne) && (chose < aOne + aTwo)) {
    xRand = fRMin * cosphi;
    yRand = fRMin * sinphi;
    zRand = RNG::Instance().uniform(-1. * fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  } else if ((chose >= aOne + aTwo) && (chose < aOne + aTwo + aFou)) {
    xRand = rRand * std::cos(fSPhi + fDPhi);
    yRand = rRand * std::sin(fSPhi + fDPhi);
    zRand = RNG::Instance().uniform(-1. * fDz, fDz);
    return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
  }

  // else

  xRand = rRand * std::cos(fSPhi + fDPhi);
  yRand = rRand * std::sin(fSPhi + fDPhi);
  zRand = RNG::Instance().uniform(-1. * fDz, fDz);
  return Vector3D<Precision>(xRand, yRand, zRand + 0.5 * (zTwo + zOne));
}

//
// GetPointOnRing
//
// Auxiliary method for GetPoint On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnRing(Precision fRMin1, Precision fRMax1, Precision fRMin2,
                                                     Precision fRMax2, Precision zOne) const
{

  Precision xRand, yRand, phi, cosphi, sinphi, rRand1, rRand2, A1, Atot, rCh;
  phi    = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);

  if (fRMin1 == fRMin2) {
    rRand1 = fRMin1;
    A1     = 0.;
  } else {
    rRand1 = RNG::Instance().uniform(fRMin1, fRMin2);
    A1     = std::fabs(fRMin2 * fRMin2 - fRMin1 * fRMin1);
  }
  if (fRMax1 == fRMax2) {
    rRand2 = fRMax1;
    Atot   = A1;
  } else {
    rRand2 = RNG::Instance().uniform(fRMax1, fRMax2);
    Atot   = A1 + std::fabs(fRMax2 * fRMax2 - fRMax1 * fRMax1);
  }
  rCh = RNG::Instance().uniform(0., Atot);

  if (rCh > A1) {
    rRand1 = rRand2;
  }

  xRand = rRand1 * cosphi;
  yRand = rRand1 * sinphi;

  return Vector3D<Precision>(xRand, yRand, zOne);
}

//
// GetPointOnCut
//
// Auxiliary method for Get Point On Surface
//
Vector3D<Precision> UnplacedPolycone::GetPointOnCut(Precision fRMin1, Precision fRMax1, Precision fRMin2,
                                                    Precision fRMax2, Precision zOne, Precision zTwo,
                                                    Precision &totArea) const
{

  if (zOne == zTwo) {
    return GetPointOnRing(fRMin1, fRMax1, fRMin2, fRMax2, zOne);
  }
  if ((fRMin1 == fRMin2) && (fRMax1 == fRMax2)) {
    return GetPointOnTubs(fRMin1, fRMax1, zOne, zTwo, totArea);
  }
  return GetPointOnCone(fRMin1, fRMax1, fRMin2, fRMax2, zOne, zTwo, totArea);
}

//
// SamplePointOnSurface
//
Vector3D<Precision> UnplacedPolycone::SamplePointOnSurface() const
{

  Precision Area = 0, totArea = 0, Achose1 = 0, Achose2 = 0, phi, cosphi, sinphi, rRand;
  int i         = 0;
  int numPlanes = GetNSections();

  phi    = RNG::Instance().uniform(GetStartPhi(), GetEndPhi());
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  std::vector<Precision> areas;
  PolyconeSection const &sec0 = GetSection(0);
  areas.push_back(kPi * (sec0.fSolid.fRmax1 * sec0.fSolid.fRmax1 - sec0.fSolid.fRmin1 * sec0.fSolid.fRmin1));
  rRand = sec0.fSolid.fRmin1 + ((sec0.fSolid.fRmax1 - sec0.fSolid.fRmin1) * std::sqrt(RNG::Instance().uniform(0., 1.)));

  areas.push_back(kPi * (sec0.fSolid.fRmax1 * sec0.fSolid.fRmax1 - sec0.fSolid.fRmin1 * sec0.fSolid.fRmin1));

  for (i = 0; i < numPlanes; i++) {
    PolyconeSection const &sec = GetSection(i);
    Area                       = (sec.fSolid.fRmin1 + sec.fSolid.fRmin2) *
           std::sqrt((sec.fSolid.fRmin1 - sec.fSolid.fRmin2) * (sec.fSolid.fRmin1 - sec.fSolid.fRmin2) +
                     4. * sec.fSolid.fDz * sec.fSolid.fDz);

    Area += (sec.fSolid.fRmax1 + sec.fSolid.fRmax2) *
            std::sqrt((sec.fSolid.fRmax1 - sec.fSolid.fRmax2) * (sec.fSolid.fRmax1 - sec.fSolid.fRmax2) +
                      4. * sec.fSolid.fDz * sec.fSolid.fDz);

    Area *= 0.5 * GetDeltaPhi();

    if (GetDeltaPhi() < kTwoPi) {
      Area += std::fabs(2 * sec.fSolid.fDz) *
              (sec.fSolid.fRmax1 + sec.fSolid.fRmax2 - sec.fSolid.fRmin1 - sec.fSolid.fRmin2);
    }

    areas.push_back(Area);
    totArea += Area;
  }
  PolyconeSection const &secn = GetSection(numPlanes - 1);
  areas.push_back(kPi * (secn.fSolid.fRmax2 * secn.fSolid.fRmax2 - secn.fSolid.fRmin2 * secn.fSolid.fRmin2));

  totArea += (areas[0] + areas[numPlanes + 1]);
  Precision chose = RNG::Instance().uniform(0., totArea);

  if ((chose >= 0.) && (chose < areas[0])) {
    return Vector3D<Precision>(rRand * cosphi, rRand * sinphi, fPolycone->fZs[0]);
  }

  for (i = 0; i < numPlanes; i++) {
    Achose1 += areas[i];
    Achose2 = (Achose1 + areas[i + 1]);
    if (chose >= Achose1 && chose < Achose2) {
      PolyconeSection const &sec = GetSection(i);
      return GetPointOnCut(sec.fSolid.fRmin1, sec.fSolid.fRmax1, sec.fSolid.fRmin2, sec.fSolid.fRmax2,
                           fPolycone->fZs[i], fPolycone->fZs[i + 1], Area);
    }
  }

  rRand = secn.fSolid.fRmin2 + ((secn.fSolid.fRmax2 - secn.fSolid.fRmin2) * std::sqrt(RNG::Instance().uniform(0., 1.)));

  return Vector3D<Precision>(rRand * cosphi, rRand * sinphi, fPolycone->fZs[numPlanes]);
}

#if (0)
// Simplest Extent defintion that does not take PHI into consideration
void UnplacedPolycone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{

  int i          = 0;
  Precision maxR = 0;

  for (i = 0; i < GetNSections(); i++) {
    PolyconeSection const &sec = GetSection(i);
    if (maxR < sec.fSolid.fRmax1) maxR = sec.fSolid.fRmax1;
    if (maxR < sec.fSolid.fRmax2) maxR = sec.fSolid.fRmax2;
  }

  aMin.x() = -maxR;
  aMin.y() = -maxR;
  //  aMin.z() = fZs[0];
  aMax.x() = maxR;
  aMax.y() = maxR;
  if (fZs[0] > fZs[GetNSections()]) {
    aMax.z() = fZs[0];
    aMin.z() = fZs[GetNSections()];
  } else {
    aMin.z() = fZs[0];
    aMax.z() = fZs[GetNSections()];
  }
}
#endif

#if (1)
// Improved Extent definition that also takes PHI into consideration
void UnplacedPolycone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  /* Algorithm:
   * 1. Get the Extent in Z direction and set
   * 	aMax.z and aMin.z
   *
   * 2. For X and Y direction use Extent of Cone
   * 	and set aMax.x aMax.y, aMin.x and aMin.y
   *
   */
  Precision maxR = 0, minR = kInfLength;
  Precision fSPhi = fPolycone->fStartPhi;
  Precision fDPhi = fPolycone->fDeltaPhi;

  for (int i = 0; i < GetNSections(); i++) {
    PolyconeSection const &sec = GetSection(i);
    maxR                       = Max(maxR, Max(sec.fSolid._frmax1, sec.fSolid._frmax2));
    minR                       = Min(minR, Min(sec.fSolid._frmin1, sec.fSolid._frmin2));
  }

  if (fPolycone->fZs[0] > fPolycone->fZs[GetNSections()]) {
    aMax.z() = fPolycone->fZs[0];
    aMin.z() = fPolycone->fZs[GetNSections()];
  } else {
    aMin.z() = fPolycone->fZs[0];
    aMax.z() = fPolycone->fZs[GetNSections()];
  }

  // Using Cone to get Extent in X and Y Direction
  Precision minz = aMin.z();
  Precision maxz = aMax.z();
  SUnplacedCone<ConeTypes::UniversalCone> tempCone(minR, maxR, minR, maxR, 1, fSPhi, fDPhi);
  tempCone.BaseType_t::Extent(aMin, aMax);
  aMin.z() = minz;
  aMax.z() = maxz;

  return;
}
#endif

#endif // !VECCORE_CUDA

#if (0)

bool UnplacedPolycone::CheckContinuityInRmax(const Vector<Precision> &rOuter)
{

  bool continuous  = true;
  unsigned int len = rOuter.size();
  if (len > 2) {
    for (unsigned int j = 1; j < len;) {
      if (j != (len - 1)) continuous &= (rOuter[j] == rOuter[j + 1]);
      j = j + 2;
    }
  }
  return continuous;
}

bool UnplacedPolycone::CheckContinuity(const Precision rOuter[], const Precision rInner[], const Precision zPlane[],
                                       Vector<Precision> &newROuter, Vector<Precision> &newRInner,
                                       Vector<Precision> &newZPlane)
{

  Vector<Precision> rOut, rIn;
  Vector<Precision> zPl;
  rOut.push_back(rOuter[0]);
  rIn.push_back(rInner[0]);
  zPl.push_back(zPlane[0]);
  for (unsigned int j = 1; j < fNz; j++) {

    if (j == fNz - 1) {
      rOut.push_back(rOuter[j]);
      rIn.push_back(rInner[j]);
      zPl.push_back(zPlane[j]);
    } else {
      if ((zPlane[j] != zPlane[j + 1]) || (rOuter[j] != rOuter[j + 1])) {
        rOut.push_back(rOuter[j]);
        rOut.push_back(rOuter[j]);

        zPl.push_back(zPlane[j]);
        zPl.push_back(zPlane[j]);

        rIn.push_back(rInner[j]);
        rIn.push_back(rInner[j]);

      } else {
        rOut.push_back(rOuter[j]);
        zPl.push_back(zPlane[j]);
        rIn.push_back(rInner[j]);
      }
    }
  }

  if (rOut.size() % 2 != 0) {
    // fNz is odd, the adding of the last item did not happen in the loop.
    rOut.push_back(rOut[rOut.size() - 1]);
    rIn.push_back(rIn[rIn.size() - 1]);
    zPl.push_back(zPl[zPl.size() - 1]);
  }

  /* Creating a new temporary Reduced polycone with desired data elements,
   *  which makes sure that denominator will never be zero (hence avoiding FPE(division by zero)),
   *  while calculating slope.
   *
   *  This will be the minimum polycone,i.e. no extra section which
   *  affect its shape
   */

  for (size_t j = 0; j < rOut.size();) {

    if (zPl[j] != zPl[j + 1]) {
      newZPlane.push_back(zPl[j]);
      newZPlane.push_back(zPl[j + 1]);
      newROuter.push_back(rOut[j]);
      newROuter.push_back(rOut[j + 1]);
      newRInner.push_back(rIn[j]);
      newRInner.push_back(rIn[j + 1]);
    }

    j = j + 2;
  }
  // Minimum polycone construction over

  // Checking Slope continuity and Rmax Continuity
  bool contRmax  = CheckContinuityInRmax(newROuter);
  bool contSlope = CheckContinuityInSlope(newROuter, newZPlane);

  // If both are true then the polycone can be convex
  // but still final convexity depends on Inner Radius also.
  return (contRmax && contSlope);
}

/* Cleaner CheckContinuityInSlope.
 * Because of new design, it will not get the case of FPE exception
 * (division by zero)
 */
bool UnplacedPolycone::CheckContinuityInSlope(const Vector<Precision> &rOuter, const Vector<Precision> &zPlane)
{

  bool continuous      = true;
  Precision startSlope = kInfLength;

  // Doing the actual slope calculation here, and checking continuity,
  for (size_t j = 0; j < rOuter.size(); j = j + 2) {
    Precision currentSlope = (rOuter[j + 1] - rOuter[j]) / (zPlane[j + 1] - zPlane[j]);
    continuous &= (currentSlope <= startSlope);
    startSlope = currentSlope;
  }
  return continuous;
}
#endif

VECCORE_ATT_HOST_DEVICE
void UnplacedPolycone::DetectConvexity()
{
  // Default safe convexity value
  fGlobalConvexity = false;

  if (fPolycone->fConvexityPossible) {
    if (fPolycone->fEqualRmax && (fPolycone->fDeltaPhi <= kPi || fPolycone->fDeltaPhi == kTwoPi))
      // In this case, Polycone become solid Cylinder, No need to check anything else, 100% convex
      fGlobalConvexity = true;
    else {
      if (fPolycone->fDeltaPhi <= kPi || fPolycone->fDeltaPhi == kTwoPi) {
        fGlobalConvexity = fPolycone->fContinuityOverAll;
      }
    }
  }

  // return convexity;
}

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::SUnplacedPolycone<ConeTypes::UniversalCone>>::SizeOf();
template void DevicePtr<cuda::SUnplacedPolycone<ConeTypes::UniversalCone>>::Construct(Precision, Precision, int,
                                                                                      Precision *, Precision *,
                                                                                      Precision *) const;

// Constructor we are targeting:
// UnplacedPolycone(bool continuityOverAll, bool convexityPossible, bool equalRmax, Precision phistart,
//                  Precision deltaphi, int nsections, int nz, Precision const *z, Precision const *rmin,
//                  Precision const *rmax, size_t const buff_size, void const *buffer)
template void vecgeom::cxx::ConstructManyOnGpu<SUnplacedPolycone<cuda::ConeTypes::UniversalCone>>(
    std::size_t nElement, DevicePtr<cuda::VUnplacedVolume> const *gpu_ptrs, bool const *continuityOverAll_arr,
    bool const *convexityPossible_arr, bool const *equalRmax_arr, Precision const *startPhi_arr,
    Precision const *deltaPhi_arr, int const *nsections_arr, int const *nz_arr, Precision const *const *zGPUPtrs,
    Precision const *const *rminGPUPtrs, Precision const *const *rmaxGPUPtrs, size_t const *buff_size_arr,
    void const *const *buffer_arr);

} // namespace cxx

#endif

} // namespace vecgeom
