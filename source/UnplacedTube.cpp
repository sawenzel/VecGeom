/// \file UnplacedTube.cpp
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTube.h"
#include "volumes/SpecializedTube.h"
#include "volumes/utilities/GenerationUtilities.h"

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTube::Print() const {
  printf("UnplacedTube {%.2f, %.2f, %.2f, %.2f, %.2f}",
         rmin(), rmax(), z(), sphi(), dphi() );
}

void UnplacedTube::Print(std::ostream &os) const {
  os << "UnplacedTube {" << rmin() << ", " << rmax() << ", " << z()
     << ", " << sphi() << ", " << dphi() << "}";
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTube::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

      using namespace TubeTypes;
      __attribute__((unused)) const UnplacedTube &tube = static_cast<const UnplacedTube&>( *(logical_volume->unplaced_volume()) );

      #ifdef VECGEOM_NVCC
        #define RETURN_SPECIALIZATION(tubeTypeT) return CreateSpecializedWithPlacement< \
            SpecializedTube<transCodeT, rotCodeT, tubeTypeT> >(logical_volume, transformation, id, placement)
      #else
        #define RETURN_SPECIALIZATION(tubeTypeT) return CreateSpecializedWithPlacement< \
            SpecializedTube<transCodeT, rotCodeT, tubeTypeT> >(logical_volume, transformation, placement)
      #endif

#ifdef GENERATE_TUBE_SPECIALIZATIONS
      if(tube.rmin() <= 0) {
        if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(NonHollowTube);
        if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(NonHollowTubeWithPiSector); // == M_PI ???

        if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(NonHollowTubeWithSmallerThanPiSector);
        if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(NonHollowTubeWithBiggerThanPiSector);
      }
      else if(tube.rmin() > 0) {
        if(tube.dphi() >= 2*M_PI)  RETURN_SPECIALIZATION(HollowTube);
        if(tube.dphi() == M_PI)    RETURN_SPECIALIZATION(HollowTubeWithPiSector); // == M_PI ???

        if(tube.dphi() < M_PI)     RETURN_SPECIALIZATION(HollowTubeWithSmallerThanPiSector);
        if(tube.dphi() > M_PI)     RETURN_SPECIALIZATION(HollowTubeWithBiggerThanPiSector);
      }
#endif

      RETURN_SPECIALIZATION(UniversalTube);

      #undef RETURN_SPECIALIZATION
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTube::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTube>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

#ifdef VECGEOM_USOLIDS
  VECGEOM_CUDA_HEADER_BOTH
  Precision UnplacedTube::SurfaceArea () const {
    Precision area = fDphi * (fRmin + fRmax) * (2 * fZ + fRmax - fRmin);
    if (fDphi<kTwoPi) {
      area += 4 * fZ * (fRmax - fRmin);
    }
    return area;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void UnplacedTube::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const {
    // most general case
    aMin = Vector3D<Precision>(-fRmax,-fRmax,-fZ);
    aMax = Vector3D<Precision>( fRmax, fRmax, fZ);

    if(fDphi==kTwoPi) return;

    // check how many of phi=90, 180, 270, 360deg are outside this tube
    auto Rin = 0.5*(fRmax+fRmin);
    bool phi0out   = ! GetWedge().Contains<kScalar>( Vector3D<Precision>( Rin, 0, 0) );
    bool phi90out  = ! GetWedge().Contains<kScalar>( Vector3D<Precision>( 0, Rin, 0) );
    bool phi180out = ! GetWedge().Contains<kScalar>( Vector3D<Precision>(-Rin, 0, 0) );
    bool phi270out = ! GetWedge().Contains<kScalar>( Vector3D<Precision>( 0,-Rin, 0) );

    // if none of those 4 phis is outside, largest box still required
    if( !(phi0out || phi90out || phi180out || phi270out) ) return;

    // some extent(s) of box will be reduced
    // --> think of 4 points A,B,C,D such that A,B are at Rmin, C,D at Rmax
    //     and A,C at startPhi (fSphi), B,D at endPhi (fSphi+fDphi)
    auto Cx = fRmax * cos(fSphi);
    auto Dx = fRmax * cos(fSphi+fDphi);
    auto Cy = fRmax * sin(fSphi);
    auto Dy = fRmax * sin(fSphi+fDphi);

    // then rewrite box sides whenever each one of those phis are not contained in the tube section
    if( phi0out )   aMax.x() = Max( Cx, Dx );
    if( phi90out )  aMax.y() = Max( Cy, Dy );
    if( phi180out ) aMin.x() = Min( Cx, Dx );
    if( phi270out ) aMin.y() = Min( Cy, Dy );

    if(fDphi>=kPi) return;

    auto Ax = fRmin * cos(fSphi);
    auto Bx = fRmin * cos(fSphi+fDphi);
    auto Ay = fRmin * sin(fSphi);
    auto By = fRmin * sin(fSphi+fDphi);

    Precision temp;
    temp = Max(Ax, Bx);
    aMax.x() =  temp > aMax.x() ? temp : aMax.x();

    temp = Max(Ay, By);
    aMax.y() =  temp > aMax.y() ? temp : aMax.y();

    temp = Min(Ax, Bx);
    aMin.x() =  temp < aMin.x() ? temp : aMin.x();

    temp = Min(Ay, By);
    aMin.y() =  temp < aMin.y() ? temp : aMin.y();

    return;
  }
#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTube::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedTube>(in_gpu_ptr, rmin(), rmax(), z(), sphi(), dphi());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTube::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedTube>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTube>::SizeOf();
template void DevicePtr<cuda::UnplacedTube>::Construct(
    const Precision rmin, const Precision rmax, const Precision z, 
    const Precision sphi, const Precision dphi) const;

} // End cxx namespace

#endif

} // End global namespace
