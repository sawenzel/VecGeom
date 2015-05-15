/// @file UnplacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTrd.h"
#include "volumes/SpecializedTrd.h"
#include "volumes/utilities/GenerationUtilities.h"

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
	Precision xx1 = fDX1;           // dx1();
	Precision xx2 = fDX2;           // dx2();
	Precision yy1 = fDY1;           // dy1();
	Precision yy2 = fDY2;           // dy2();
	Precision zz1 = fDZ;            // dz();
	Precision zz2 = -1.0 * fDZ;     // dz();
	Precision dz = Abs(zz1 - zz2);
	bool xvert = (xx1 == xx2) ? true : false;
	Precision SA = 0.0;
	
	// Sum of area for planes Perp. to +X and -X
	Precision ht = (xvert) ? dz : Sqrt((xx1-xx2)*(xx1-xx2) + dz*dz);
	SA += 2.0 * 0.5 * (yy1 + yy2) * ht;
	
	// Sum of area for planes Perp. to +Y and -Y
	SA += 2.0 * 0.5 * (xx1 + xx2) * ht;    // if xvert then topology forces to become yvert for closing
	
	// Sum of area for top and bottom planes +Z and -Z
	SA += (xx1 * yy1) + (xx2 * yy2);
	
	return SA;
}


void UnplacedTrd::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const {
	aMin.x() = -1.0 * Min(dx1(), dx2());
	aMax.x() = Max(dx1(), dx2());
	aMin.y() = -1.0 * Min(dy1(), dy2());
	aMax.y() = Max(dy1(), dy2());
	aMin.z() = -dz();
	aMax.z() = dz();
}

int UnplacedTrd::ChooseSurface() const {
	int i, j, nChoice = 6;
	Precision sumWeight = 0.0;
	Precision PlusXArea = GetPlusXArea();
	Precision MinusXArea = GetMinusXArea();
	Precision PlusYArea = GetPlusYArea();
	Precision MinusYArea = GetMinusYArea();
	Precision PlusZArea = GetPlusZArea();
	Precision MinusZArea = GetMinusZArea();
	Precision totArea = PlusXArea + MinusXArea + PlusYArea + MinusYArea + PlusZArea + MinusZArea;
	double prob[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	int iprob[6] = {0, 0, 0, 0, 0, 0};

	prob[0] = PlusXArea / totArea;   iprob[0] = 0; sumWeight += prob[0];
	prob[1] = MinusXArea / totArea;  iprob[1] = 1; sumWeight += prob[1];
	prob[2] = PlusYArea / totArea;   iprob[2] = 2; sumWeight += prob[2];
	prob[3] = MinusYArea / totArea;  iprob[3] = 3; sumWeight += prob[3];
	prob[4] = PlusZArea / totArea;   iprob[4] = 4; sumWeight += prob[4];
	prob[5] = MinusZArea / totArea;  iprob[5] = 5; sumWeight += prob[5];

	// sorting the array
    Precision tmp1, tmp2;
	for (i = 0; i < nChoice - 1; i++) {
		for (j = 0; j < nChoice -1; j++) {
			if (prob[j] > prob[j+1]) {
				tmp1 = prob[j];        tmp2 = iprob[j];
				prob[j] = prob[j+1];   iprob[j] = iprob[j+1];
				prob[j+1] = tmp1;      iprob[j+1] = tmp2;
			}
		}
	}

	Precision firstRnd = RNG::Instance().uniform() * sumWeight;
	for (i = 0; i < nChoice; i++) {
		if (firstRnd < prob[i]) return iprob[i];
		firstRnd -= prob[i];
	}
}

Vector3D<Precision> UnplacedTrd::GetPointOnSurface() const {
	Precision xVal = 0.0;
	Precision yVal = 0.0;
	Precision zVal = 0.0;
	Precision xx1 = fDX1;           // dx1();
	Precision xx2 = fDX2;           // dx2();
	Precision yy1 = fDY1;           // dy1();
	Precision yy2 = fDY2;           // dy2();
	Precision zz1 = fDZ;            // dz();
	Precision zz2 = -1.0 * fDZ;     // dz();
	Precision minX = Min(xx1, xx2);
	Precision maxX = Max(xx1, xx2);
	Precision minY = Min(yy1, yy2);
	Precision maxY = Max(yy1, yy2);
	Precision dx = Abs(xx1 - xx2);
	Precision dy = Abs(yy1 - yy2);
	Precision dz = Abs(zz1 - zz2);
	Precision llX = (dx <= 1.0e-8) ? dz : Sqrt(x2minusx1() * x2minusx1()  + dztimes2() * dztimes2());
	Precision llY = (dy <= 1.0e-8) ? dz : Sqrt(y2minusy1() * y2minusy1()  + dztimes2() * dztimes2());
	Precision dzstar;
	bool xvert(false);
	bool yvert(false);
	xvert = (xx1 == xx2) ? true : false;
	yvert = (yy1 == yy2) ? true : false;
	int choice = ChooseSurface();

	switch (choice) {

		case 0:   // +X plane perpendicular to +X axis
			xVal = (xvert) ? xx1 : RNG::Instance().uniform() * (Abs(xx1 - xx2)) + minX;
			zVal = (xvert) ? RNG::Instance().uniform() * dz + zz2 : (maxX - xVal) / fx();
			dzstar = (xvert) ? zVal : Sqrt((xx1 - xVal) * (xx1 - xVal) + (zVal * zVal));
			yVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * yy1 - yy1 : RNG::Instance().uniform() * 2.0 * (llX * xx1 + x2minusx1() * dzstar) + (-yy1);
		break;
		
		case 1:  // -X
			xVal = (xvert) ? xx1 : RNG::Instance().uniform() * (Abs(xx1 - xx2)) + minX;
			zVal = (xvert) ? RNG::Instance().uniform() * dz + zz2 : (maxX - xVal) / fx();
			dzstar = (xvert) ? zVal : Sqrt((xx1 - xVal) * (xx1 - xVal) + (zVal * zVal));
			yVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * yy1 - yy1 : RNG::Instance().uniform() * 2.0 * (llX * xx1 + x2minusx1() * dzstar) + (-yy1);;
			xVal *= -1.0;
		break;
		
		case 2:  // + Y
			yVal = (yvert) ? yy1 : RNG::Instance().uniform() * (Abs(yy1 - yy2)) + minY;
			zVal = (yvert) ? RNG::Instance().uniform() * dz + zz2 : (maxY - yVal) / fy();
			dzstar = (yvert) ? zVal : Sqrt((yy1 - yVal) * (yy1 - yVal) + (zVal * zVal));
			xVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * xx1 - xx1 : RNG::Instance().uniform() * 2.0 * (llY * yy1 + y2minusy1() * dzstar) + (-xx1);
		break;

		case 3:  // -Y
			yVal = (yvert) ? yy1 : RNG::Instance().uniform() * (Abs(yy1 - yy2)) + minY;
			zVal = (yvert) ? RNG::Instance().uniform() * dz + zz2 : Abs(minY - yVal) /fy();
			dzstar = (yvert) ? zVal : Sqrt((yy1 - yVal) * (yy1 - yVal) + (zVal * zVal));
			xVal = (xvert && yvert) ? RNG::Instance().uniform() * 2.0 * xx1 - xx1 : RNG::Instance().uniform() * 2.0 * (llY * yy1 + y2minusy1() * dzstar) + (-xx1);
			yVal *= -1.0;
		break;

		case 4: // +Z
			xVal = RNG::Instance().uniform() * 2.0 * fDX2 + (-fDX2);
			yVal = RNG::Instance().uniform() * 2.0 * fDY2 + (-fDY2);
			zVal = zz1;
		break;

		case 5: // -Z
			xVal = RNG::Instance().uniform() * 2.0 * fDX1 + (-fDX1);
			yVal = RNG::Instance().uniform() * 2.0 * fDY1 + (-fDY1);
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

     __attribute__((unused)) const UnplacedTrd &trd = static_cast<const UnplacedTrd&>( *(logical_volume->unplaced_volume()) );

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
