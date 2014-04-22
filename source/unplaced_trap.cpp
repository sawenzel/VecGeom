/**
 * @file unplaced_trap.cpp
 * @author Guilherme Lima (Guilherme.Lima@cern.ch)
 *
 * 140407 G.Lima - based on equivalent box code
 */

#include "volumes/unplaced_trap.h"

#include "base/vector3d.h"
#include "management/volume_factory.h"
#include "volumes/specialized_trap.h"
#include "volumes/placed_box.h"
//#include "backend.h"

#include <sstream>
#include <iostream>
using std::endl;
using std::cerr;

namespace VECGEOM_NAMESPACE {

UnplacedTrap::UnplacedTrap(Precision pDz, Precision pTheta, Precision pPhi,
						   Precision pDy1, Precision pDx1, Precision pDx2, Precision pAlpha1,
						   Precision pDy2, Precision pDx3, Precision pDx4, Precision pAlpha2 )  {
	// validity check
	if( pDz <= 0 || pDy1 <= 0 || pDx1 <= 0 ||
		pDx2 <= 0 || pDy2 <= 0 || pDx3 <= 0 || pDx4 <= 0 ) {

		std::ostringstream message;
		message << "Invalid length parameters for Solid: UnplacedTrap\n" // << GetName() << G4endl
				<< "        X - "
				<< pDx1 << ", " << pDx2 << ", " << pDx3 << ", " << pDx4 << endl
				<< "          Y - " << pDy1 << ", " << pDy2 << endl
				<< "          Z - " << pDz;
		//G4Exception("G4Trap::G4Trap()", "GeomSolids0002", FatalException, message);
		cerr << "UnplacedTrap(pDz,...) - GeomSolids0002, Fatal Exception: "<< message << endl;
		std::exit(1);
	}

	_params.set( 0, pDz );

	_params.set( 1, pDy1 );
	_params.set( 2, pDx1 );
	_params.set( 3, pDx2 );
	_params.set( 4, pAlpha1);

	_params.set( 5, pDy2 );
	_params.set( 6, pDx3 );
	_params.set( 7, pDx4 );
	_params.set( 8, pAlpha2 );

	_params.set( 9, tan(pTheta)*sin(pPhi) );
	_params.set(10, tan(pTheta)*cos(pPhi) );

	MakePlanes();
}

// Use corner points pt[8] to initialize
UnplacedTrap::UnplacedTrap( TrapCorners_t const& pt ) {

  bool good;

  // Start with check of centering

  // - the center of gravity trap line should cross the origin of frame
  if (!(   pt[0].z() < 0
        && pt[0].z() == pt[1].z() && pt[0].z() == pt[2].z()
        && pt[0].z() == pt[3].z()
        && pt[4].z() > 0
        && pt[4].z() == pt[5].z() && pt[4].z() == pt[6].z()
        && pt[4].z() == pt[7].z()
        && std::fabs( pt[0].z() + pt[4].z() ) < kTolerance
        && pt[0].y() == pt[1].y() && pt[2].y() == pt[3].y()
        && pt[4].y() == pt[5].y() && pt[6].y() == pt[7].y()
        && std::fabs( pt[0].y() + pt[2].y() + pt[4].y() + pt[6].y() ) < kTolerance
        && std::fabs( pt[0].x() + pt[1].x() + pt[4].x() + pt[5].x() +
           pt[2].x() + pt[3].x() + pt[6].x() + pt[7].x() ) < kTolerance ) )
  {
    std::ostringstream message;
    message << "Invalid vertice coordinates for Solid: trap"; //<< GetName();
    //G4Exception("G4Trap::G4Trap()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap(corners[8]) - GeomSolids0002, fatal exception: "<< message << endl;
    std::exit(1);
  }

  // Bottom side with normal approx. -Y
  good = MakePlane(pt[0],pt[4],pt[5],pt[1],_fPlanes[0]);
  if (!good) {
    //    DumpInfo();
    std::ostringstream message;
    message << "Invalid vertice coordinates for Solid: trap"; //<< GetName();
    //G4Exception("G4Trap::G4Trap()", "GeomSolids0002", FatalException, "Face at ~-Y not planar.");
    cerr << "UnplacedTrap(corners[8]) - GeomSolids0002, fatal exception. " << endl;
    std::exit(1);
  }

  // Top side with normal approx. +Y
  good = MakePlane(pt[2],pt[3],pt[7],pt[6],_fPlanes[1]);
  if (!good) {
    std::ostringstream message;
    message << "Face at ~+Y not planar for Solid: trap"; // << GetName();
    //G4Exception("G4Trap::G4Trap()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap(corners[8]) - GeomSolids0002, fatal exception: " << message << endl;
    std::exit(1);
  }

  // Front side with normal approx. -X
  good = MakePlane(pt[0],pt[2],pt[6],pt[4],_fPlanes[2]);
  if (!good) {
    std::ostringstream message;
    message << "Face at ~-X not planar for Solid: trap"; // << GetName();
    //G4Exception("G4Trap::G4Trap()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap(corners[8]) - GeomSolids0002, fatal exception: " << message << endl;
    std::exit(1);
  }

  // Back side with normal approx. +X

  good = MakePlane(pt[1],pt[5],pt[7],pt[3],_fPlanes[3]);
  if (!good) {
    std::ostringstream message;
    message << "Face at ~+X not planar for Solid: trap"; // << GetName();
    //G4Exception("G4Trap::G4Trap()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap(corners[8]) - GeomSolids0002, fatal exception: " << message << endl;
    std::exit(1);
  }

  this->fromCornersToParameters(pt);
};


#ifndef VECGEOM_NVCC

  template <TranslationCode trans_code, RotationCode rot_code>
  VPlacedVolume* UnplacedTrap::Create(LogicalVolume const *const logical_volume,
	  TransformationMatrix const *const matrix,
	  VPlacedVolume *const placement) {

	if (placement) {
	  new(placement) SpecializedTrap<trans_code, rot_code>(logical_volume, matrix);
      return placement;
    }
    return new SpecializedTrap<trans_code, rot_code>(logical_volume, matrix);
  }

  VPlacedVolume* UnplacedTrap::SpecializedVolume( LogicalVolume const *const volume,
                                                  TransformationMatrix const *const matrix,
                                                  const TranslationCode trans_code, const RotationCode rot_code ) const {
     return VolumeFactory::Instance().CreateByTransformation<UnplacedTrap>(volume, matrix, trans_code, rot_code);
  }

#else

template <TranslationCode trans_code, RotationCode rot_code>
__device__
VPlacedVolume* UnplacedTrap::Create(
    LogicalVolume const *const logical_volume,
    TransformationMatrix const *const matrix,
    const int id, VPlacedVolume *const placement) {
  if (placement) {
    new(placement) SpecializedTrap<trans_code, rot_code>(logical_volume, matrix,
                                                        id);
    return placement;
  }
  return new SpecializedTrap<trans_code, rot_code>(logical_volume, matrix, id);
}

__device__
VPlacedVolume* UnplacedBox::CreateSpecializedVolume( LogicalVolume const *const volume,
		TransformationMatrix const *const matrix,
		const TranslationCode trans_code, const RotationCode rot_code,
		const int id, VPlacedVolume *const placement) {

	return VolumeFactory::CreateByTransformation<UnplacedTrap>( volume,
		matrix, trans_code, rot_code, id, placement );
}

#endif
} // End global namespace

namespace vecgeom {

VECGEOM_CUDA_HEADER_BOTH
void UnplacedTrap::Print() const {
  printf("Trap {%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}\n",
		 _params[0], _params[1], _params[2], _params[3], _params[4], _params[0],
		 _params[6], _params[7], _params[8], _params[9], _params[10]);
}

//////////////////////////////////////////////////////////////////////////
//
// Checking coplanarity of all four side faces
//
bool UnplacedTrap::MakePlanes() {
  TrapCorners_t pt;
  fromParametersToCorners(pt);
  return MakePlanes(pt);
}

bool UnplacedTrap::MakePlanes(TrapCorners_t const & pt) {

  bool good = true;

  // Bottom side with normal approx. -Y
  //
  good = MakePlane(pt[0],pt[4],pt[5],pt[1],_fPlanes[0]) ;
  if (!good) {
    std::ostringstream message;
    message << "Face at ~-Y not planar for Solid: UnplacedTrap"; // << GetName();
    //G4Exception("G4Trap::MakePlanes()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap::MakePlanes() - GeomSolids0002 - "<< message << endl;
    abort();
  }

  // Top side with normal approx. +Y
  //
  good = MakePlane(pt[2],pt[3],pt[7],pt[6],_fPlanes[1]);
  if (!good) {
    std::ostringstream message;
    message << "Face at ~+Y not planar for Solid: UnplacedTrap"; // << GetName();
    //G4Exception("G4Trap::MakePlanes()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap::MakePlanes() - GeomSolids0002 - "<< message << endl;
    std::abort();
  }

  // Front side with normal approx. -X
  //
  good = MakePlane(pt[0],pt[2],pt[6],pt[4],_fPlanes[2]);
  if (!good) {
    std::ostringstream message;
    message << "Face at ~-X not planar for Solid: UnplacedTrap"; // << GetName();
    //G4Exception("G4Trap::MakePlanes()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap::MakePlanes() - GeomSolids0002 - "<< message << endl;
    exit(1);
  }
   
  // Back side iwth normal approx. +X
  //
  good = MakePlane(pt[1],pt[5],pt[7],pt[3],_fPlanes[3]);
  if ( !good ) {
    std::ostringstream message;
    message << "Face at ~+X not planar for Solid: UnplacedTrap"; // << GetName();
    //G4Exception("G4Trap::MakePlanes()", "GeomSolids0002", FatalException, message);
    cerr << "UnplacedTrap::MakePlanes() - GeomSolids0002 - "<< message << endl;
    std::exit(1);
  }

  return good;
}

//////////////////////////////////////////////////////////////////////////////
//
// Calculate the coef's of the plane p1->p2->p3->p4->p1
// where the ThreeVectors 1-4 are in anti-clockwise order when viewed from
// infront of the plane (i.e. from normal direction).
//
// Return true if the ThreeVectors are coplanar + set coef;s
//        false if ThreeVectors are not coplanar

bool UnplacedTrap::MakePlane( const Vector3D<Precision>& p1,
			      const Vector3D<Precision>& p2,
			      const Vector3D<Precision>& p3,
			      const Vector3D<Precision>& p4,
			      TrapSidePlane& plane )
{
  bool good;
  Precision a, b, c, sd;
  Vector3D<Precision> v12, v13, v14, Vcross;

  v12    = p2 - p1;
  v13    = p3 - p1;
  v14    = p4 - p1;
  Vcross = Vcross.Cross(v12,v13);

  // check coplanarity
  if (std::fabs(v14.Dot(Vcross)/(Vcross.Length()*v14.Length())) > kTolerance)  {
	  good = false;
  }
  else {
    // a,b,c correspond to the x/y/z components of the
    // normal vector to the plane
     
    // a  = (p2.y()-p1.y())*(p1.z()+p2.z())+(p3.y()-p2.y())*(p2.z()+p3.z());
    // a += (p4.y()-p3.y())*(p3.z()+p4.z())+(p1.y()-p4.y())*(p4.z()+p1.z()); // ?   
    // b  = (p2.z()-p1.z())*(p1.x()+p2.x())+(p3.z()-p2.z())*(p2.x()+p3.x());
    // b += (p4.z()-p3.z())*(p3.x()+p4.x())+(p1.z()-p4.z())*(p4.x()+p1.x()); // ?      
    // c  = (p2.x()-p1.x())*(p1.y()+p2.y())+(p3.x()-p2.x())*(p2.y()+p3.y());
    // c += (p4.x()-p3.x())*(p3.y()+p4.y())+(p1.x()-p4.x())*(p4.y()+p1.y()); // ?

    // Let create diagonals 4-2 and 3-1 than (4-2)x(3-1) provides
    // vector perpendicular to the plane directed to outside !!!
    // and a,b,c, = f(1,2,3,4) external relative to trap normal

    //??? can these be vectorized?
    a = +(p4.y() - p2.y())*(p3.z() - p1.z())
       - (p3.y() - p1.y())*(p4.z() - p2.z());

    b = -(p4.x() - p2.x())*(p3.z() - p1.z())
       + (p3.x() - p1.x())*(p4.z() - p2.z());
 
    c = +(p4.x() - p2.x())*(p3.y() - p1.y())
       - (p3.x() - p1.x())*(p4.y() - p2.y());

    sd = std::sqrt( a*a + b*b + c*c ); // so now vector plane.(a,b,c) is unit 

    if( sd > 0 ) {
      plane._a = a/sd;
      plane._b = b/sd;
      plane._c = c/sd;
    }
    else {
      std::ostringstream message;
      message << "Invalid parameters: norm.mod() <= 0, for Solid: UnplacedTrap"; // << GetName();
      //G4Exception("G4Trap::MakePlanes()", "GeomSolids0002", FatalException, message) ;
      cerr << "UnplacedTrap::MakePlane() - GeomSolids0002 - "<< message << endl;
      exit(1);
    }

    // Calculate D: p1 in in plane so D=-n.p1.Vect()
    plane._d = -( plane._a*p1.x() + plane._b*p1.y() + plane._c*p1.z() );

    good = true;
  }
  return good;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision UnplacedTrap::volume() const {

	auto dz = _params.par(0);
	auto dy1 = _params.par(1);
	auto dx1 = _params.par(2);
	auto dx2 = _params.par(3);
	auto dy2 = _params.par(5);
	auto dx3 = _params.par(6);
	auto dx4 = _params.par(7);
    // cubic approximation used in Geant4
    Precision cubicVolume = dz*( (dx1+dx2+dx3+dx4)*(dy1 + dy2)
								   + (dx4+dx3-dx2-dx1)*(dy2 - dy1)/3.0 );
    /*
      double const* pt = this->GetCornerPoints();

      // more precise, hopefully correct version (to be checked)
      Precision BmZm = pt[1].x() - pt[0].x();
      Precision BpZm = pt[3].x() - pt[2].x();
      Precision BmZp = pt[5].x() - pt[4].x();
      Precision BpZp = pt[7].x() - pt[6].x();
      Precision xCorr = (BpZp-BpZm + BmZp-BmZm) / (BpZm+BmZm);

      Precision ymZm = pt[0].y();
      Precision ypZm = pt[2].y();
      Precision ymZp = pt[4].y();
      Precision ypZp = pt[6].y();
      Precision yCorr = (ypZp-ypZm - (ymZp-ymZm)) / (ypZm-ymZm);

      Precision volume = 8*dz*dy1*(dx1+dx2) * ( 1.0 + (xCorr + yCorr)/2.0 + xCorr*yCorr/3.0 );
    */

    return cubicVolume;
}

void UnplacedTrap::fromCornersToParameters( TrapCorners_t const& pt) {

	auto Dz = pt[7].z();
	auto DzRecip = 1.0 / Dz;
	_params.set( 0, Dz );

    auto Dy1 = 0.50*( pt[2].y() - pt[1].y() );
	auto Dx1 = 0.50*( pt[1].x() - pt[0].x() );
	auto Dx2 = 0.50*( pt[3].x() - pt[2].x() );
	auto Talpha1 = 0.25*( pt[2].x() + pt[3].x() - pt[1].x() - pt[0].x() ) / Dy1;
	_params.set( 1, Dy1 );
	_params.set( 2, Dx1 );
	_params.set( 3, Dx2 );
	_params.set( 4, Talpha1 );

	auto Dy2 = 0.50*( pt[6].y() - pt[5].y() );
	auto Dx3 = 0.50*( pt[5].x() - pt[4].x() );
	auto Dx4 = 0.50*( pt[7].x() - pt[6].x() );
	auto Talpha2 = 0.25*( pt[6].x() + pt[7].x() - pt[5].x() - pt[4].x() ) / Dy2;
	_params.set( 5, Dy2 );
	_params.set( 6, Dx3 );
	_params.set( 7, Dx4 );
	_params.set( 8, Talpha2 );

    auto tanThetaCosPhi = ( pt[4].x() + Dy2*Talpha2 + Dx3) * DzRecip;
    auto tanThetaSinPhi = ( pt[4].y() + Dy2) * DzRecip;
	_params.set( 9, tanThetaSinPhi );
	_params.set(10, tanThetaCosPhi );
  }

  void UnplacedTrap::fromParametersToCorners( TrapCorners_t& pt ) const {

	  Precision const& Dz  = _params[0];
	  Precision const& Dy1 = _params[1];
	  Precision const& Dx1 = _params[2];
	  Precision const& Dx2 = _params[3];
	  Precision const& Talpha1 = _params[4];
	  Precision const& Dy2 = _params[5];
	  Precision const& Dx3 = _params[6];
	  Precision const& Dx4 = _params[7];
	  Precision const& Talpha2 = _params[8];
	  Precision const& TthetaSphi = _params[9];
	  Precision const& TthetaCphi = _params[10];

	  // hopefully the compiler will optimize the repeated multiplications ... to be checked!
	  pt[0] = Vector3D<Precision>(-Dz*TthetaCphi-Dy1*Talpha1-Dx1, -Dz*TthetaSphi-Dy1, -Dz);
	  pt[1] = Vector3D<Precision>(-Dz*TthetaCphi-Dy1*Talpha1+Dx1, -Dz*TthetaSphi-Dy1, -Dz);
	  pt[2] = Vector3D<Precision>(-Dz*TthetaCphi+Dy1*Talpha1-Dx2, -Dz*TthetaSphi+Dy1, -Dz);
	  pt[3] = Vector3D<Precision>(-Dz*TthetaCphi+Dy1*Talpha1+Dx2, -Dz*TthetaSphi+Dy1, -Dz);
	  pt[4] = Vector3D<Precision>(+Dz*TthetaCphi-Dy2*Talpha2-Dx3, +Dz*TthetaSphi-Dy2, +Dz);
	  pt[5] = Vector3D<Precision>(+Dz*TthetaCphi-Dy2*Talpha2+Dx3, +Dz*TthetaSphi-Dy2, +Dz);
	  pt[6] = Vector3D<Precision>(+Dz*TthetaCphi+Dy2*Talpha2-Dx4, +Dz*TthetaSphi+Dy2, +Dz);
	  pt[7] = Vector3D<Precision>(+Dz*TthetaCphi+Dy2*Talpha2+Dx4, +Dz*TthetaSphi+Dy2, +Dz);
  }


#ifdef VECGEOM_NVCC

__global__
void ConstructOnGpu(const UnplacedTrap trap, VUnplacedVolume *const gpu_ptr) {
	new(gpu_ptr) vecgeom_cuda::UnplacedTrap(trap);
}

void UnplacedTrap_CopyToGpu(const UnplacedTrap trap, VUnplacedVolume *const gpu_ptr) {
  ConstructOnGpu<<<1, 1>>>(*this, gpu_ptr);
}

#endif

} // End global namespace
