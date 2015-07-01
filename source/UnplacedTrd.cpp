/// @file UnplacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTrd.h"
#include "volumes/SpecializedTrd.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"

#include "management/VolumeFactory.h"


namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTrd::Print() const {
  printf("UnplacedTrd {%.2f, %.2f, %.2f, %.2f, %.2f}",
         dx1(), dx2(), dy1(), dy2(), dz() );
}

void UnplacedTrd::Print(std::ostream &os) const {
  os << "UnplacedTrd {" << dx1() << ", " << dx2() << ", " << dy1()
     << ", " << dy2() << ", " << dz();
}

#ifndef VECGEOM_NVCC
Precision UnplacedTrd::Capacity() const {
   return  2*(fDX1+fDX2)*(fDY1+fDY2)*fDZ
     + (2./3.)*(fDX1-fDX2)*(fDY1-fDY2)*fDZ;
}

Precision UnplacedTrd::SurfaceArea() const {
  Precision dz = 2*fDZ;
  bool xvert = (fDX1 == fDX2) ? true : false;
  Precision SA = 0.0;

  // Sum of area for planes Perp. to +X and -X
  Precision ht = (xvert) ? dz : Sqrt((fDX1-fDX2)*(fDX1-fDX2) + dz*dz);
  SA += 2.0 * 0.5 * (fDY1 + fDY2) * ht;

  // Sum of area for planes Perp. to +Y and -Y
  SA += 2.0 * 0.5 * (fDX1 + fDX2) * ht;    // if xvert then topology forces to become yvert for closing

  // Sum of area for top and bottom planes +Z and -Z
  SA += (fDX1 * fDY1) + (fDX2 * fDY2);

  return SA;
}

/*

void UnplacedTrd::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const {
	aMin.x() = -1.0 * Min(dx1(), dx2());
	aMax.x() = Max(dx1(), dx2());
	aMin.y() = -1.0 * Min(dy1(), dy2());
	aMax.y() = Max(dy1(), dy2());
	aMin.z() = -dz();
	aMax.z() = dz();
}
*/

int UnplacedTrd::ChooseSurface() const {
    int choice = 0;
    int nChoice = 6;
    Precision sumWeight = 0.0;
    Precision PlusXArea = GetPlusXArea();
    Precision MinusXArea = GetMinusXArea();
    Precision PlusYArea = GetPlusYArea();
    Precision MinusYArea = GetMinusYArea();
    Precision PlusZArea = GetPlusZArea();
    Precision MinusZArea = GetMinusZArea();
    Precision totArea = PlusXArea + MinusXArea + PlusYArea + MinusYArea + PlusZArea + MinusZArea;
    Precision prob[6] ;

    prob[0] = PlusXArea  / totArea;
    prob[1] = MinusXArea / totArea;
    prob[2] = PlusYArea  / totArea;
    prob[3] = MinusYArea / totArea;
    prob[4] = PlusZArea  / totArea;
    prob[5] = MinusZArea / totArea;

    for (int i = 0; i < nChoice; i++)
        sumWeight += prob[i];

    Precision rand = RNG::Instance().uniform() * sumWeight;

    while (rand > prob[choice])
        rand -= prob[choice], choice++;

    assert(choice < nChoice);
    return choice;
}

Vector3D<Precision> UnplacedTrd::GetPointOnSurface() const {
    Precision xVal = 0.0;
    Precision yVal = 0.0;
    Precision zVal = 0.0;
    Precision zz2 = -1.0 * fDZ;
    Precision minX = Min(fDX1, fDX2);
    Precision maxX = Max(fDX1, fDX2);
    Precision minY = Min(fDY1, fDY2);
    Precision maxY = Max(fDY1, fDY2);
    Precision dz = 2.*fDZ;
    bool xvert(false);
    bool yvert(false);
    xvert = (fDX1 == fDX2) ? true : false;
    yvert = (fDY1 == fDY2) ? true : false;
    Precision llX = (xvert) ? dz : Sqrt(x2minusx1() * x2minusx1()  + dztimes2() * dztimes2());
    Precision llY = (yvert) ? dz : Sqrt(y2minusy1() * y2minusy1()  + dztimes2() * dztimes2());
    Precision dzstar;
    int choice = ChooseSurface();

    switch (choice) {
        case 0:   // +X plane perpendicular to +X axis
            xVal = (xvert) ? fDX1 : RNG::Instance().uniform() * (Abs(fDX1 - fDX2)) + minX;
            zVal = (xvert) ? RNG::Instance().uniform() * dz + zz2 : (maxX - xVal) / fx();
            dzstar = (xvert) ? zVal : Sqrt((fDX1 - xVal) * (fDX1 - xVal) + (zVal * zVal));
            yVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * fDY1 - fDY1 : RNG::Instance().uniform() * 2.0 * (llX * fDX1 + x2minusx1() * dzstar) + (-fDY1);
        break;
        case 1:  // -X
            xVal = (xvert) ? fDX1 : RNG::Instance().uniform() * (Abs(fDX1 - fDX2)) + minX;
            zVal = (xvert) ? RNG::Instance().uniform() * dz + zz2 : (maxX - xVal) / fx();
            dzstar = (xvert) ? zVal : Sqrt((fDX1 - xVal) * (fDX1 - xVal) + (zVal * zVal));
            yVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * fDY1 - fDY1 : RNG::Instance().uniform() * 2.0 * (llX * fDX1 + x2minusx1() * dzstar) + (-fDY1);
            xVal *= -1.0;
        break;
        case 2:  // + Y
            yVal = (yvert) ? fDY1 : RNG::Instance().uniform() * (Abs(fDY1 - fDY2)) + minY;
            zVal = (yvert) ? RNG::Instance().uniform() * dz + zz2 : (maxY - yVal) / fy();
            dzstar = (yvert) ? zVal : Sqrt((fDY1 - yVal) * (fDY1 - yVal) + (zVal * zVal));
            xVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * fDX1 - fDX1 : RNG::Instance().uniform() * 2.0 * (llY * fDY1 + y2minusy1() * dzstar) + (-fDX1);
        break;
        case 3:  // -Y
            yVal = (yvert) ? fDY1 : RNG::Instance().uniform() * (Abs(fDY1 - fDY2)) + minY;
            zVal = (yvert) ? RNG::Instance().uniform() * dz + zz2 : Abs(minY - yVal) /fy();
            dzstar = (yvert) ? zVal : Sqrt((fDY1 - yVal) * (fDY1 - yVal) + (zVal * zVal));
            xVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * fDX1 - fDX1 : RNG::Instance().uniform() * 2.0 * (llY * fDY1 + y2minusy1() * dzstar) + (-fDX1);
            yVal *= -1.0;
        break;
        case 4: // +Z
            xVal = RNG::Instance().uniform() * (2.0 * fDX2) + (-fDX2);
            yVal = RNG::Instance().uniform() * (2.0 * fDY2) + (-fDY2);
            zVal = fDZ;
        break;
        case 5: // -Z
            xVal = RNG::Instance().uniform() * (2.0 * fDX1) + (-fDX1);
            yVal = RNG::Instance().uniform() * (2.0 * fDY1) + (-fDY1);
            zVal = zz2;
        break;
    }
    return Vector3D<Precision> (xVal, yVal, zVal);
}


bool UnplacedTrd::Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const {
	int nosurface = 0;
	bool onSurf(false);

    Precision xp = point[0];
    Precision yp = point[1];
    Precision zp = point[2];
    Precision xx1 = dx1();
    Precision xx2 = dx2();
    Precision yy1 = dy1();
    Precision yy2 = dy2();
    Precision zz = dz();
    Precision XplusX   = 2.0 * yy1 * 2.0 * zz;
    Precision XminusX  = XplusX;
    Precision XplusY   = 0.0;
    Precision XminusY  = 0.0;
    Precision XplusZ   = 2.0 * yy1 * (xx1 - xx2);
    Precision XminusZ  = 2.0 * yy1 * (-xx1 + xx2);

    Precision YplusX   = 0.0;
    Precision YminusX  = 0.0;
    Precision YplusY   = -2.0 * xx1 * 2.0 * zz;
    Precision YminusY  = YplusY;
    Precision YplusZ   = 2.0 * xx1 * (-yy1 + yy2);
    Precision YminusZ  = 2.0 * xx1 * (yy1 - yy2);
    Precision ZplusX   = 0.0;
    Precision ZminusX  = 0.0;
    Precision ZplusY   = 0.0;
    Precision ZminusY  = 0.0;
    Precision ZplusZ   = -2.0 * xx2 * 2.0 * yy2;
    Precision ZminusZ  = 2.0 * xx2 * 2.0 * yy1;

	// Checking for each plane whether the point is on Surface, if yes transfer normal
    bool FacPlusX  = XplusX * (xp - xx2) + XplusY * (yp - yy2) + XplusZ * (zp - zz);
    bool FacMinusX = XminusX * (xp + xx2) + XminusY * (yp - yy2) + XminusZ * (zp - zz);
    bool FacPlusY  = YplusX * (xp - xx2)  + YplusY * (yp - yy2) + YplusZ * (zp - zz);
    bool FacMinusY = YplusX * (xp - xx2) + YminusY * (yp + yy2) + YminusZ * (zp - zz);
    bool FacPlusZ  = ZplusX * (xp - xx2) + ZplusY * (yp - yy2) + ZplusZ * (zp - zz);
    bool FacMinusZ = ZminusX * (xp - xx2) + ZminusY * (yp + yy2) + ZminusZ * (zp - zz);
    onSurf = FacPlusX || FacMinusX || FacPlusY || FacMinusY || FacPlusZ || FacMinusZ;
    if (onSurf && FacPlusX)  norm[0] = XplusX;   norm[1] = XplusY;  norm[2] = XplusZ;  nosurface++;
    if (onSurf && FacMinusX) norm[0] = XminusX;  norm[1] = XminusY; norm[2] = XminusZ; nosurface++;
    if (onSurf && FacPlusY)  norm[0] = YplusX;   norm[1] = YplusY;  norm[2] = YplusZ;  nosurface++;
    if (onSurf && FacMinusY) norm[0] = YminusX;  norm[1] = YminusY; norm[2] = YminusZ; nosurface++;
    if (onSurf && FacPlusZ)  norm[0] = ZplusX;   norm[1] = ZplusY;  norm[2] = ZplusZ;  nosurface++;
    if (onSurf && FacMinusZ) norm[0] = ZminusX;  norm[1] = ZminusY; norm[2] = ZminusZ; nosurface++;
    return nosurface != 0;
}


#endif

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrd::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

    using namespace TrdTypes;

#ifndef VECGEOM_NO_SPECIALIZATION

     __attribute__((unused)) const UnplacedTrd &trd = static_cast<const UnplacedTrd&>( *(logical_volume->GetUnplacedVolume()) );

    #define GENERATE_TRD_SPECIALIZATIONS
    #ifdef GENERATE_TRD_SPECIALIZATIONS
      if(trd.dy1() == trd.dy2()) {
    //          std::cout << "trd1" << std::endl;
          return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd1> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                 ,id
#endif
                 , placement);
      } else {
    //          std::cout << "trd2" << std::endl;
          return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd2> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                 ,id
#endif
                 , placement);
    }
    #endif

#endif // VECGEOM_NO_SPECIALIZATION

      //    std::cout << "universal trd" << std::endl;
    return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::UniversalTrd> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                ,id
#endif
                , placement);
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrd::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTrd>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedTrd>(in_gpu_ptr, dx1(), dx2(), dy1(), dy2(), dz());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedTrd>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTrd>::SizeOf();
template void DevicePtr<cuda::UnplacedTrd>::Construct(
   const Precision dx1, const Precision dx2, const Precision dy1,
   const Precision dy2, const Precision d) const;

} // End cxx namespace

#endif

} // End global namespace
