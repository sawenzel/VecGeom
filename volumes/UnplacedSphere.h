/// \file UnplacedSphere.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
#define VECGEOM_VOLUMES_UNPLACEDSPHERE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#ifndef VECGEOM_NVCC
  #include "base/RNG.h"
#include <cassert>
#include <cmath>
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedSphere; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedSphere );

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedSphere : public VUnplacedVolume, public AlignedBase {

private:
    // Radial and angular dimensions
    Precision fRmin, fRmax, fSPhi, fDPhi, fSTheta, fDTheta;
    
    //Radial and angular tolerances
    Precision fRminTolerance, mkTolerance, kAngTolerance,
            kRadTolerance, fEpsilon;
    
    // Cached trigonometric values for Phi angle
    Precision sinCPhi, cosCPhi, cosHDPhiOT, cosHDPhiIT,
           sinSPhi, cosSPhi, sinEPhi, cosEPhi, hDPhi, cPhi, ePhi;
    
    // Cached trigonometric values for Theta angle
    Precision sinSTheta, cosSTheta, sinETheta, cosETheta,
           tanSTheta, tanSTheta2, tanETheta, tanETheta2, eTheta;
    
    // Flags for identification of section, shell or full sphere
    bool fFullPhiSphere, fFullThetaSphere, fFullSphere;
    
    // Precomputed values computed from parameters
    Precision fCubicVolume, fSurfaceArea;
    
    //Tolerance compatiable with USolids
    Precision epsilon;// = 2e-11; 
    Precision frTolerance;//=1e-9;     //radial tolerance;
    Precision fgTolerance ;//= 1e-9;  // cartesian tolerance;
    Precision faTolerance ;//= 1e-9;  // angular tolerance;


    // Member variables go here
    // Precision fR,fRTolerance, fRTolI, fRTolO;
 

public:
    
VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
void InitializePhiTrigonometry()
{
  hDPhi = 0.5 * fDPhi;        // half delta phi
  cPhi  = fSPhi + hDPhi;
  ePhi  = fSPhi + fDPhi;

  sinCPhi   = std::sin(cPhi);
  cosCPhi   = std::cos(cPhi);
  cosHDPhiIT = std::cos(hDPhi - 0.5 * kAngTolerance); // inner/outer tol half dphi
  cosHDPhiOT = std::cos(hDPhi + 0.5 * kAngTolerance);
  sinSPhi = std::sin(fSPhi);
  cosSPhi = std::cos(fSPhi);
  sinEPhi = std::sin(ePhi);
  cosEPhi = std::cos(ePhi);
}  
    
VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
void InitializeThetaTrigonometry()
{
  eTheta  = fSTheta + fDTheta;

  sinSTheta = std::sin(fSTheta);
  cosSTheta = std::cos(fSTheta);
  sinETheta = std::sin(eTheta);
  cosETheta = std::cos(eTheta);

  tanSTheta = std::tan(fSTheta);
  tanSTheta2 = tanSTheta * tanSTheta;
  tanETheta = std::tan(eTheta);
  tanETheta2 = tanETheta * tanETheta;
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CheckThetaAngles(Precision sTheta, Precision dTheta)
{
  if ((sTheta < 0) || (sTheta > kPi))
  {
   //std::ostringstream message;
   // message << "sTheta outside 0-PI range." << std::endl
     //       << "Invalid starting Theta angle for solid: " ;//<< GetName();
    //return;
    //UUtils::Exception("USphere::CheckThetaAngles()", "GeomSolids0002",
      //                FatalError, 1, message.str().c_str());
  }
  else
  {
    fSTheta = sTheta;
  }
  if (dTheta + sTheta >= kPi)
  {
    fDTheta = kPi - sTheta;
  }
  else if (dTheta > 0)
  {
    fDTheta = dTheta;
  }
  else
  {
    /*
      std::ostringstream message;
    message << "Invalid dTheta." << std::endl
            << "Negative delta-Theta (" << dTheta << "), for solid: ";
    return;
     */ 
            //<< GetName();
    //UUtils::Exception("USphere::CheckThetaAngles()", "GeomSolids0002",
      //                FatalError, 1, message.str().c_str());
  }
  if (fDTheta - fSTheta < kPi)
  {
    fFullThetaSphere = false;
  }
  else
  {
    fFullThetaSphere = true ;
  }
  fFullSphere = fFullPhiSphere && fFullThetaSphere;

  InitializeThetaTrigonometry();
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CheckSPhiAngle(Precision sPhi)
{
  // Ensure fSphi in 0-2PI or -2PI-0 range if shape crosses 0

  if (sPhi < 0)
  {
    fSPhi = 2 * kPi - std::fmod(std::fabs(sPhi), 2 * kPi);
  }
  else
  {
    fSPhi = std::fmod(sPhi, 2 * kPi) ;
  }
  if (fSPhi + fDPhi > 2 * kPi)
  {
    fSPhi -= 2 * kPi ;
  }
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CheckDPhiAngle(Precision dPhi)
{
  fFullPhiSphere = true;
  if (dPhi >= 2 * kPi - kAngTolerance * 0.5)
  {
    fDPhi = 2 * kPi;
    fSPhi = 0;
  }
  else
  {
    fFullPhiSphere = false;
    if (dPhi > 0)
    {
      fDPhi = dPhi;
    }
    else
    {
        /*
      std::ostringstream message;
      message << "Invalid dphi." << std::endl
              << "Negative delta-Phi (" << dPhi << "), for solid: ";
      return;
         */ 
           
             // << GetName();
      //UUtils::Exception("USphere::CheckDPhiAngle()", "GeomSolids0002",
        //                FatalError, 1, message.str().c_str());
    }
  }
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void CheckPhiAngles(Precision sPhi, Precision dPhi)
{
  CheckDPhiAngle(dPhi);
  //if (!fFullPhiSphere && sPhi) { CheckSPhiAngle(sPhi); }
  if (!fFullPhiSphere)
  {
    CheckSPhiAngle(sPhi);
  }
  fFullSphere = fFullPhiSphere && fFullThetaSphere;

  InitializePhiTrigonometry();
}

  //constructor
  //VECGEOM_CUDA_HEADER_BOTH
  //UnplacedSphere();

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedSphere(Precision pRmin, Precision pRmax,
                 Precision pSPhi, Precision pDPhi,
                 Precision pSTheta, Precision pDTheta);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInsideRadius() const { return fRmin; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInnerRadius() const { return fRmin; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetOuterRadius() const { return fRmax; } 
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStartPhiAngle() const { return fSPhi; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDeltaPhiAngle() const { return fDPhi; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStartThetaAngle() const { return fSTheta; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDeltaThetaAngle() const { return fDTheta; }
  
  //Functions to get Tolerance
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetFRminTolerance() const { return fRminTolerance; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetAngTolerance() const { return kAngTolerance; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsFullSphere() const { return fFullPhiSphere; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsFullPhiSphere() const { return fFullSphere; }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsFullThetaSphere() const { return fFullThetaSphere; }
  
  //All angle related functions
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetHDPhi() const { return hDPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCPhi() const { return cPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEPhi() const { return ePhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinCPhi() const { return sinCPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosCPhi() const { return cosCPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinSPhi() const { return sinSPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosSPhi() const { return cosSPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinEPhi() const { return sinEPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosEPhi() const { return cosEPhi;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetETheta() const { return eTheta;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinSTheta() const { return sinSTheta;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosSTheta() const { return cosSTheta;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetSinETheta() const { return sinETheta;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetCosETheta() const { return cosETheta;}
  
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Initialize(){
    fCubicVolume = 0.;
    fSurfaceArea = 0.;
  }
  
  //VECGEOM_CUDA_HEADER_BOTH
  //VECGEOM_INLINE
  void SetInsideRadius(Precision newRmin)
  {
  fRmin = newRmin;
  fRminTolerance = (fRmin) ? std::max(kRadTolerance, fEpsilon * fRmin) : 0;
  Initialize();
  CalcCapacity();
  CalcSurfaceArea();
  }
  
  //VECGEOM_CUDA_HEADER_BOTH
  //VECGEOM_INLINE
  void SetInnerRadius(Precision newRmin)
  {
    SetInsideRadius(newRmin);
  }
  
  //VECGEOM_CUDA_HEADER_BOTH
  //VECGEOM_INLINE
  void SetOuterRadius(Precision newRmax)
  {
    fRmax = newRmax;
    mkTolerance = std::max(kRadTolerance, fEpsilon * fRmax); //RELOOK at kTolerance, may be we will take directly from base/global.h
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }
  
  //VECGEOM_CUDA_HEADER_BOTH
  //VECGEOM_INLINE
  void SetStartPhiAngle(Precision newSPhi, bool compute)
  {
   // Flag 'compute' can be used to explicitely avoid recomputation of
   // trigonometry in case SetDeltaPhiAngle() is invoked afterwards
 
   CheckSPhiAngle(newSPhi);
   fFullPhiSphere = false;
   if (compute)
   {
    InitializePhiTrigonometry();
   }
   Initialize();
   CalcCapacity();
   CalcSurfaceArea();
  }
  
  //VECGEOM_CUDA_HEADER_BOTH
  //VECGEOM_INLINE
  void SetDeltaPhiAngle(Precision newDPhi)
  {
   CheckPhiAngles(fSPhi, newDPhi);
   Initialize();
   CalcCapacity();
   CalcSurfaceArea();
  }
  
  //VECGEOM_CUDA_HEADER_BOTH  
  //VECGEOM_INLINE
  void SetStartThetaAngle(Precision newSTheta)
  {
    CheckThetaAngles(newSTheta, fDTheta);
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }
  
  //VECGEOM_CUDA_HEADER_BOTH  
  //VECGEOM_INLINE
  void SetDeltaThetaAngle(Precision newDTheta)
  {
    CheckThetaAngles(fSTheta, newDTheta);
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }
  
// Old access functions
VECGEOM_CUDA_HEADER_BOTH  
  VECGEOM_INLINE
Precision GetRmin() const
{
  return GetInsideRadius();
}

VECGEOM_CUDA_HEADER_BOTH 
VECGEOM_INLINE
Precision GetRmax() const
{
  return GetOuterRadius();
}

VECGEOM_CUDA_HEADER_BOTH 
VECGEOM_INLINE
Precision GetSPhi() const
{
  return GetStartPhiAngle();
}

VECGEOM_CUDA_HEADER_BOTH 
VECGEOM_INLINE
Precision GetDPhi() const
{
  return GetDeltaPhiAngle();
}

VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
Precision GetSTheta() const
{
  return GetStartThetaAngle();
}

VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
Precision GetDTheta() const
{
  return GetDeltaThetaAngle();
}

  
  //*****************************************************
  /*
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolO() const { return fRTolO; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolI() const { return fRTolI; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetfRTolerance() const { return fRTolerance; }
  
  VECGEOM_CUDA_HEADER_BOTH
  void SetRadius (const Precision r);
  
  //_____________________________________________________________________________
  */

VECGEOM_CUDA_HEADER_BOTH
void CalcCapacity();

VECGEOM_CUDA_HEADER_BOTH
  void CalcSurfaceArea();

  VECGEOM_CUDA_HEADER_BOTH
  void Extent( Vector3D<Precision> &, Vector3D<Precision> &) const;
   
  VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const;
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea() const;
  
#if !defined(VECGEOM_NVCC)
  Vector3D<Precision> GetPointOnSurface() const;
#endif
 
  VECGEOM_CUDA_HEADER_BOTH
  std::string GetEntityType() const;

  void GetParametersList(int aNumber, Precision *aArray) const;
  
  UnplacedSphere* Clone() const;

  std::ostream& StreamInfo(std::ostream &os) const;
    
  
  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBBox() const; 
  
  //VECGEOM_CUDA_HEADER_BOTH
  //Precision sqr(Precision x) {return x*x;}; 
   
  
public:
  virtual int memory_size() const { return sizeof(*this); }

  
  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;//{}//;
  
  //VECGEOM_CUDA_HEADER_BOTH 
  virtual void Print(std::ostream &os) const;//{}//;

  
  #ifndef VECGEOM_NVCC

  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL);

  #else

  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               const int id,
                               VPlacedVolume *const placement = NULL);

  __device__
  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL);

  #endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedSphere>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:

  

  #ifndef VECGEOM_NVCC

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   placement);
  }

#else

  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   id, placement);
  }

#endif

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
