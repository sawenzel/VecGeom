/// @file UnplacedGenericPolycone.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "VecGeom/volumes/Cone.h"
#include "VecGeom/volumes/UnplacedCoaxialCones.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedGenericPolycone.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/SpecializedGenericPolycone.h"
#include "VecGeom/base/RNG.h"
#include <stdio.h>
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/*
 * All the required Parametric Constructor
 */

VECCORE_ATT_HOST_DEVICE
UnplacedGenericPolycone::UnplacedGenericPolycone(Precision phiStart, // initial phi starting angle
                                                 Precision phiTotal, // total phi angle
                                                 int numRZ, // number corners in r,z space (must be an even number)
                                                 Precision const *r, // r coordinate of these corners
                                                 Precision const *z)
{
  fGlobalConvexity = false;
  fSPhi            = phiStart;
  fDPhi            = phiTotal;
  fNumRZ           = numRZ;
  for (int i = 0; i < numRZ; i++) {
    fR.push_back(r[i]);
    fZ.push_back(z[i]);
  }

  Vector<Vector2D<Precision>> rzVect;
  for (int i = 0; i < numRZ; i++) {
    rzVect.push_back(Vector2D<Precision>(r[i], z[i]));
  }
  ReducedPolycone rd(rzVect);
  Vector<Precision> zS;
  Vector<Vector<Precision>> vectOfRmin1Vect;
  Vector<Vector<Precision>> vectOfRmax1Vect;
  Vector<Vector<Precision>> vectOfRmin2Vect;
  Vector<Vector<Precision>> vectOfRmax2Vect;
  rd.GetPolyconeParameters(vectOfRmin1Vect, vectOfRmax1Vect, vectOfRmin2Vect, vectOfRmax2Vect, zS, fAMin, fAMax);

  fGenericPolycone.Set(vectOfRmin1Vect, vectOfRmax1Vect, vectOfRmin2Vect, vectOfRmax2Vect, zS, fSPhi, fDPhi);
  ComputeBBox();
}

VECCORE_ATT_HOST_DEVICE
void UnplacedGenericPolycone::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  /*  Algo : Extent along Z direction will be Z[0] and Z[max]
   *         Extent along X and Y will be from the Extent of
   *         cone/tube formed with Maximum outer radius
   */
  Precision rmax = 0.;
  for (unsigned int i = 0; i < fR.size(); i++) {
    if (fR[i] > rmax) {
      rmax = fR[i];
    }
  }
  /* Using the simplest extent for the time being, because
   * currently it's not possible to generate a Cone using
   * factory on GPU. Once it becomes possible then the code
   * written below can be used.
   */
  aMin.Set(-rmax, -rmax, fAMin.z());
  aMax.Set(rmax, rmax, fAMax.z());
  /*auto coneUnplaced = GeoManager::MakeInstance<UnplacedCone>(0., rmax, 0., rmax, 1., fSPhi, fDPhi);
  coneUnplaced->Extent(aMin, aMax);
  aMin.z() = fAMin.z();
  aMax.z() = fAMax.z();*/
}

/* Borrowed the definition from Polycone and made the required modifications*/
VECCORE_ATT_HOST_DEVICE
bool UnplacedGenericPolycone::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  bool valid = true;
  int index  = fGenericPolycone.GetSectionIndex(point.z() - kTolerance);

  if (index < 0) {
    // Match Polycone: global z end-cap normals are valid surface normals.
    if (index == -1) norm = Vector3D<Precision>(0., 0., -1.);
    if (index == -2) norm = Vector3D<Precision>(0., 0., 1.);
    return valid;
  }
  GenericPolyconeSection const &sec = fGenericPolycone.GetSection(index);
  valid                             = sec.fCoaxialCones->Normal(point - Vector3D<Precision>(0, 0, sec.fShift), norm);

  // if point is within tolerance of a Z-plane between 2 sections, get normal from other section too
  if (size_t(index + 1) < fGenericPolycone.fSections.size() &&
      vecCore::math::Abs(point.z() - fGenericPolycone.fZs[index + 1]) < kTolerance) {
    GenericPolyconeSection const &sec2 = fGenericPolycone.GetSection(index + 1);
    bool valid2                        = false;
    Vector3D<Precision> norm2;
    valid2 = sec2.fCoaxialCones->Normal(point - Vector3D<Precision>(0, 0, sec2.fShift), norm2);

    if (!valid && valid2) {
      norm  = norm2;
      valid = valid2;
    }

    // if both valid && valid2 true, norm and norm2 should be added...
    if (valid && valid2) {

      // discover exiting direction by moving point a bit (it would be good to have track direction here)
      // if(sec.fSolid->Contains(point + kTolerance*10*norm - Vector3D<Precision>(0, 0, sec.fShift))){

      //}
      bool c2;
      using CI = CoaxialConesImplementation; // ConeImplementation<ConeTypes::UniversalCone>;
      CI::Contains(*sec2.fCoaxialCones, point + kTolerance * 10 * norm2 - Vector3D<Precision>(0, 0, sec2.fShift), c2);
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
  if (valid) norm /= vecCore::math::Sqrt(norm.Mag2());
  return valid;
}

Vector3D<Precision> UnplacedGenericPolycone::SamplePointOnSurface() const
{
  /* Algo : First select the section
   *        From the selected section select the cone
   *        Sample the point from selected cone and return the sampled point
   */
  for (int attempt = 0; attempt < 1024; ++attempt) {
    int sectionSelection                        = (int)RNG::Instance().uniform(0., fGenericPolycone.GetNSections());
    const GenericPolyconeSection &section       = fGenericPolycone.GetSection(sectionSelection);
    CoaxialConesStruct<Precision> *coaxialCones = section.fCoaxialCones;
    Vector<ConeStruct<Precision> *> coneStructVector = coaxialCones->fConeStructVector;
    int coneSelection                                = (int)RNG::Instance().uniform(0., coneStructVector.size());
    ConeStruct<Precision> *coneStruct                = coneStructVector[coneSelection];
    SUnplacedCone<ConeTypes::UniversalCone> coneUnplaced(coneStruct->fRmin1, coneStruct->fRmax1, coneStruct->fRmin2,
                                                         coneStruct->fRmax2, coneStruct->fDz, coneStruct->fSPhi,
                                                         coneStruct->fDPhi);
    auto sample = coneUnplaced.SamplePointOnSurface();
    sample.z() += section.fShift;

    // Section-local cone surfaces may be internal shared z planes; the public
    // sampler contract requires points on the full generic-polycone surface.
    if (Inside(sample) == vecgeom::EInside::kSurface) return sample;
  }

  return VUnplacedVolume::SamplePointOnSurface();
}

VECCORE_ATT_HOST_DEVICE
void UnplacedGenericPolycone::Print() const
{
  // Provided Elliptical Cone Parameters as done for Tube below
  // printf("GenericPolycone {%.2f, %.2f, %.2f}", fGenericPolycone.fDx, fGenericPolycone.fDy, fGenericPolycone.fDz);
  printf("UnplacedGenericPolycone:  {StartPhi: %f   DeltaPhi: %f   N_RZpoints: %d}\n", fSPhi, fDPhi, fNumRZ);
  printf("        -------------------------------------------\n");
  for (int i = 0; i < fNumRZ; i++) {
    printf("        RZpoint #%i:  %f, %f\n", i, fR[i], fZ[i]);
  }
  printf("        -------------------------------------------\n");
}

void UnplacedGenericPolycone::Print(std::ostream &os) const
{
  // Provided Elliptical Cone Parameters as done for Tube below
  // os << "GenericPolycone {" << fEllipticalTube.fDx << ", " << fEllipticalTube.fDy << ", " << fEllipticalTube.fDz <<
  // "}";
  os << "---------------------------------------------------\n"
     << " Solid type: GenericPolycone\n"
     << " Parameters: \n"
     << "    starting phi angle : " << fSPhi << " radians \n"
     << "    ending phi angle   : " << fDPhi << " radians \n";

  os << "    number of RZ points: " << fNumRZ << "\n"
     << "              RZ values (corners): \n";
  for (int i = 0; i < fNumRZ; i++) {
    os << "                         " << fR[i] << ", " << fZ[i] << "\n";
  }
  os << "-----------------------------------------------------------\n";
}

#ifndef VECCORE_CUDA
VPlacedVolume *UnplacedGenericPolycone::Create(LogicalVolume const *const logical_volume,
                                               Transformation3D const *const transformation,
                                               VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedGenericPolycone(logical_volume, transformation);
    return placement;
  }
  return new SpecializedGenericPolycone(logical_volume, transformation);
}

VPlacedVolume *UnplacedGenericPolycone::SpecializedVolume(LogicalVolume const *const volume,
                                                          Transformation3D const *const transformation,
                                                          VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedGenericPolycone>(volume, transformation, placement);
}
#else

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedGenericPolycone::Create(LogicalVolume const *const logical_volume,
                                               Transformation3D const *const transformation, const int id,
                                               const int copy_no, const int child_id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedGenericPolycone(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedGenericPolycone(logical_volume, transformation, id, copy_no, child_id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedGenericPolycone::SpecializedVolume(LogicalVolume const *const volume,
                                                          Transformation3D const *const transformation, const int id,
                                                          const int copy_no, const int child_id,
                                                          VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedGenericPolycone>(volume, transformation, id, copy_no, child_id,
                                                                        placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedGenericPolycone::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedGenericPolycone>();
}

DevicePtr<cuda::VUnplacedVolume> UnplacedGenericPolycone::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  Precision *z_gpu_ptr = AllocateOnGpu<Precision>(fNumRZ * sizeof(Precision));
  Precision *r_gpu_ptr = AllocateOnGpu<Precision>(fNumRZ * sizeof(Precision));

  vecgeom::CopyToGpu(&fZ[0], z_gpu_ptr, sizeof(Precision) * fNumRZ);
  vecgeom::CopyToGpu(&fR[0], r_gpu_ptr, sizeof(Precision) * fNumRZ);

  DevicePtr<cuda::VUnplacedVolume> gpuGenericPolycone =
      CopyToGpuImpl<UnplacedGenericPolycone>(in_gpu_ptr, fSPhi, fDPhi, fNumRZ, r_gpu_ptr, z_gpu_ptr);
  // remove temporary space from GPU
  FreeFromGpu(z_gpu_ptr);
  FreeFromGpu(r_gpu_ptr);

  return gpuGenericPolycone;
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedGenericPolycone>::SizeOf();
template void DevicePtr<cuda::UnplacedGenericPolycone>::Construct(Precision phiStart, Precision phiTotal, int numRZ,
                                                                  Precision *r, Precision *z) const;

} // namespace cxx

#endif
} // namespace vecgeom
