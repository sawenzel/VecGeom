/// \file UnplacedSphere.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
#define VECGEOM_VOLUMES_UNPLACEDSPHERE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "base/RNG.h"

namespace VECGEOM_NAMESPACE {

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
    Precision epsilon = 2e-11; 
    Precision frTolerance=1e-9;     //radial tolerance;
    Precision fgTolerance = 1e-9;  // cartesian tolerance;
    Precision faTolerance = 1e-9;  // angular tolerance;


    // Member variables go here
    // Precision fR,fRTolerance, fRTolI, fRTolO;
 

public:
    
VECGEOM_CUDA_HEADER_BOTH    
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
  Precision GetInsideRadius() const { return fRmin; }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetInnerRadius() const { return fRmin; }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetOuterRadius() const { return fRmax; } 
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetStartPhiAngle() const { return fSPhi; }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDeltaPhiAngle() const { return fDPhi; }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetStartThetaAngle() const { return fSTheta; }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDeltaThetaAngle() const { return fDTheta; }
  
  //Functions to get Tolerance
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetFRminTolerance() const { return fRminTolerance; }
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAngTolerance() const { return kAngTolerance; }
  
  VECGEOM_CUDA_HEADER_BOTH
  bool IsFullSphere() const { return fFullPhiSphere; }
  
  VECGEOM_CUDA_HEADER_BOTH
  bool IsFullPhiSphere() const { return fFullSphere; }
  
  VECGEOM_CUDA_HEADER_BOTH
  bool IsFullThetaSphere() const { return fFullThetaSphere; }
  
  VECGEOM_CUDA_HEADER_BOTH
  void Initialize(){
    fCubicVolume = 0.;
    fSurfaceArea = 0.;
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  void SetInsideRadius(Precision newRmin)
  {
  fRmin = newRmin;
  fRminTolerance = (fRmin) ? std::max(kRadTolerance, fEpsilon * fRmin) : 0;
  Initialize();
  CalcCapacity();
  CalcSurfaceArea();
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  void SetInnerRadius(Precision newRmin)
  {
    SetInsideRadius(newRmin);
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  void SetOuterRadius(double newRmax)
  {
    fRmax = newRmax;
    mkTolerance = std::max(kRadTolerance, fEpsilon * fRmax); //RELOOK at kTolerance, may be we will take directly from base/global.h
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }
  
  VECGEOM_CUDA_HEADER_BOTH
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
  
  VECGEOM_CUDA_HEADER_BOTH
  void SetDeltaPhiAngle(Precision newDPhi)
  {
   CheckPhiAngles(fSPhi, newDPhi);
   Initialize();
   CalcCapacity();
   CalcSurfaceArea();
  }
  
  VECGEOM_CUDA_HEADER_BOTH  
  void SetStartThetaAngle(Precision newSTheta)
  {
    CheckThetaAngles(newSTheta, fDTheta);
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }
  
  VECGEOM_CUDA_HEADER_BOTH  
  void SetDeltaThetaAngle(Precision newDTheta)
  {
    CheckThetaAngles(fSTheta, newDTheta);
    Initialize();
    CalcCapacity();
    CalcSurfaceArea();
  }
  
// Old access functions
VECGEOM_CUDA_HEADER_BOTH  
Precision GetRmin() const
{
  return GetInsideRadius();
}

VECGEOM_CUDA_HEADER_BOTH  
Precision GetRmax() const
{
  return GetOuterRadius();
}

VECGEOM_CUDA_HEADER_BOTH  
Precision GetSPhi() const
{
  return GetStartPhiAngle();
}

VECGEOM_CUDA_HEADER_BOTH  
Precision GetDPhi() const
{
  return GetDeltaPhiAngle();
}

VECGEOM_CUDA_HEADER_BOTH  
Precision GetSTheta() const
{
  return GetStartThetaAngle();
}

VECGEOM_CUDA_HEADER_BOTH  
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
  
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision>  GetPointOnSurface() const;
 
  
  VECGEOM_CUDA_HEADER_BOTH
  std::string GetEntityType() const;
  
    
  VECGEOM_CUDA_HEADER_BOTH
  void GetParametersList(int aNumber, double *aArray) const; 
  
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedSphere* Clone() const;

  VECGEOM_CUDA_HEADER_BOTH
  std::ostream& StreamInfo(std::ostream &os) const;
    
  
  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBBox() const; 
  
  //VECGEOM_CUDA_HEADER_BOTH
  //Precision sqr(Precision x) {return x*x;}; 
   
  
public:
  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

private:

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const;

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDSPHERE_H_
