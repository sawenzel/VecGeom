/*
 * ConeStruct.h
 *
 *  Created on: May 11, 2017
 *      Author: Raman Sehgal
 */
#ifndef VECGEOM_POLYCONESTRUCT_H_
#define VECGEOM_POLYCONESTRUCT_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include "VecGeom/volumes/ConeStruct.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/volumes/PolyconeHistorical.h"
#include "VecGeom/volumes/PolyconeSection.h"
#include "VecGeom/base/Array.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// a plain and lightweight struct to encapsulate data members of a Polycone
template <typename T = double>
struct PolyconeStruct {

  bool fEqualRmax;
  bool fContinuityOverAll;
  bool fConvexityPossible;

  evolution::Wedge fPhiWedge;
  Precision fStartPhi;
  Precision fDeltaPhi;
  unsigned int fNz;

  Vector<PolyconeSection> fSections;
  Vector<Precision> fZs;
  PolyconeHistorical *fOriginal_parameters;

  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuity(const Precision rOuter[], const Precision rInner[], const Precision zPlane[],
                       Vector<Precision> &newROuter, Vector<Precision> &newRInner, Vector<Precision> &newZPlane)
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

  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuityInRmax(const Vector<Precision> &rOuter)
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

  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuityInSlope(const Vector<Precision> &rOuter, const Vector<Precision> &zPlane)
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

  VECCORE_ATT_HOST_DEVICE
  void Init(Precision phiStart,       // initial phi starting angle
            Precision phiTotal,       // total phi angle
            unsigned int numZPlanes,  // number of z planes
            const Precision zPlane[], // position of z planes
            const Precision rInner[], // tangent distance to inner surface
            const Precision rOuter[])
  {

    SetAndCheckDPhiAngle(phiTotal);
    SetAndCheckSPhiAngle(phiStart);
    fNz                = numZPlanes;
    Precision *zPlaneR = new Precision[numZPlanes];
    Precision *rInnerR = new Precision[numZPlanes];
    Precision *rOuterR = new Precision[numZPlanes];
    for (unsigned int i = 0; i < numZPlanes; i++) {
      zPlaneR[i] = zPlane[i];
      rInnerR[i] = rInner[i];
      rOuterR[i] = rOuter[i];
    }
    if (zPlane[0] > zPlane[numZPlanes - 1]) {
      // Reverse the arrays
      for (unsigned int i = 0; i < numZPlanes; i++) {
        zPlaneR[i] = zPlane[numZPlanes - 1 - i];
        rInnerR[i] = rInner[numZPlanes - 1 - i];
        rOuterR[i] = rOuter[numZPlanes - 1 - i];
      }
    }

    // Conversion for angles
    if (phiTotal <= 0. || phiTotal > kTwoPi - kTolerance) {
      // phiIsOpen=false;
      fStartPhi = 0;
      fDeltaPhi = kTwoPi;
    } else {
      //
      // Convert phi into our convention
      //
      fStartPhi = phiStart;
      while (fStartPhi < 0)
        fStartPhi += kTwoPi;
    }

    // Calculate RMax of Polycone in order to determine convexity of sections
    //
    Precision RMaxextent = rOuterR[0];

    Vector<Precision> newROuter, newZPlane, newRInner;
    fContinuityOverAll &= CheckContinuity(rOuterR, rInnerR, zPlaneR, newROuter, newRInner, newZPlane);
    fConvexityPossible &= (newRInner[0] == 0.);

    Precision startRmax = newROuter[0];
    for (unsigned int j = 1; j < newROuter.size(); j++) {
      fEqualRmax &= (startRmax == newROuter[j]);
      startRmax = newROuter[j];
      fConvexityPossible &= (newRInner[j] == 0.);
    }

    for (unsigned int j = 1; j < numZPlanes; j++) {

      if (rOuterR[j] > RMaxextent) RMaxextent = rOuterR[j];

      if (rInnerR[j] > rOuterR[j]) {
#ifndef VECCORE_CUDA
        std::cerr << "Cannot create Polycone with rInner > rOuter for the same Z"
                  << "\n"
                  << "        rInner > rOuter for the same Z !\n"
                  << "        rMin[" << j << "] = " << rInner[j] << " -- rMax[" << j << "] = " << rOuter[j];
#endif
      }
    }

    Precision prevZ = zPlaneR[0], prevRmax = 0, prevRmin = 0;
    int dirZ = 1;
    if (zPlaneR[1] < zPlaneR[0]) dirZ = -1;

    for (unsigned int i = 0; i < numZPlanes; ++i) {
      if ((i < numZPlanes - 1) && (zPlaneR[i] == zPlaneR[i + 1])) {
        if ((rInnerR[i] > rOuterR[i + 1]) || (rInnerR[i + 1] > rOuterR[i])) {
#ifndef VECCORE_CUDA
          std::cerr << "Cannot create a Polycone with no contiguous segments." << std::endl
                    << "                Segments are not contiguous !" << std::endl
                    << "                rMin[" << i << "] = " << rInnerR[i] << " -- rMax[" << i + 1
                    << "] = " << rOuterR[i + 1] << std::endl
                    << "                rMin[" << i + 1 << "] = " << rInnerR[i + 1] << " -- rMax[" << i
                    << "] = " << rOuterR[i];
#endif
        }
      }

      Precision rMin = rInnerR[i];

      Precision rMax = rOuterR[i];
      Precision z    = zPlaneR[i];

      // i has to be at least one to complete a section
      if (i > 0) {
	// GL: I had to add kTolerance here, otherwise this polycone would get an
	//     extra 0-length ZSection on the GPU -- see VECGEOM-578
        if (((z > prevZ + kTolerance) && (dirZ > 0)) || ((z < prevZ - kTolerance) && (dirZ < 0))) {
          if (dirZ * (z - prevZ) < 0) {
#ifndef VECCORE_CUDA
            std::cerr << "Cannot create a Polycone with different Z directions.Use GenericPolycone." << std::endl
                      << "              ZPlane is changing direction  !" << std::endl
                      << "  zPlane[0] = " << zPlaneR[0] << " -- zPlane[1] = " << zPlaneR[1] << std::endl
                      << "  zPlane[" << i - 1 << "] = " << zPlaneR[i - 1] << " -- rPlane[" << i << "] = " << zPlaneR[i];
#endif
          }

          ConeStruct<Precision> *solid;

          Precision dz = (z - prevZ) / 2;

          solid = new ConeStruct<Precision>(prevRmin, prevRmax, rMin, rMax, dz, phiStart, phiTotal);

          fZs.push_back(z);
          int zi          = fZs.size() - 1;
          Precision shift = fZs[zi - 1] + 0.5 * (fZs[zi] - fZs[zi - 1]);

          PolyconeSection section;
          section.fShift = shift;
          section.fSolid = solid;

          section.fConvex = !((rMax < prevRmax) || (rMax < RMaxextent) || (prevRmax < RMaxextent));

          fSections.push_back(section);
        }
      } else { // for i == 0 just push back first z plane
        fZs.push_back(z);
      }

      prevZ    = z;
      prevRmin = rMin;
      prevRmax = rMax;
    }

    fOriginal_parameters                  = new PolyconeHistorical(numZPlanes);
    fOriginal_parameters->fHStart_angle   = phiStart;
    fOriginal_parameters->fHOpening_angle = phiTotal;
    for (unsigned int i = 0; i < numZPlanes; i++) {
      fOriginal_parameters->fHZ_values[i] = zPlaneR[i];
      fOriginal_parameters->fHRmin[i]     = rInnerR[i];
      fOriginal_parameters->fHRmax[i]     = rOuterR[i];
    }

    delete[] zPlaneR;
    delete[] rInnerR;
    delete[] rOuterR;
  }

  VECCORE_ATT_HOST_DEVICE
  PolyconeHistorical *GetOriginalParameters() const { return fOriginal_parameters; }

  VECCORE_ATT_HOST_DEVICE unsigned int GetNz() const { return fNz; }

  VECCORE_ATT_HOST_DEVICE
  int GetNSections() const { return fSections.size(); }

  VECCORE_ATT_HOST_DEVICE
  int GetSectionIndex(Precision zposition) const
  {
    // TODO: consider binary search
    // TODO: consider making these comparisons tolerant in case we need it
    if (zposition < fZs[0]) return -1;
    for (unsigned int i = 0; i < fSections.size(); ++i) {
      if (zposition >= fZs[i] && zposition <= fZs[i + 1]) return i;
    }
    return -2;
  }

  VECCORE_ATT_HOST_DEVICE
  PolyconeSection const &GetSection(Precision zposition) const
  {
    // TODO: consider binary search
    int i = GetSectionIndex(zposition);
    return fSections[i];
  }

  VECCORE_ATT_HOST_DEVICE
  // GetSection if index is known
  PolyconeSection const &GetSection(int index) const { return fSections[index]; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRminAtPlane(int index) const
  {
    int nsect = fSections.size();
    assert(index >= 0 && index <= nsect);
    if (index == nsect)
      return fSections[index - 1].fSolid->fRmin2; // GetRmin2();
    else
      return fSections[index].fSolid->fRmin1; // GetRmin1();
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmaxAtPlane(int index) const
  {
    int nsect = fSections.size();
    assert(index >= 0 || index <= nsect);
    if (index == nsect)
      return fSections[index - 1].fSolid->fRmax2; // GetRmax2();
    else
      return fSections[index].fSolid->fRmax1; // GetRmax1();
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetZAtPlane(unsigned int index) const
  {
    assert(index <= fSections.size());
    return fZs[index];
  }

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckSPhiAngle(Precision sPhi)
  {
    // Ensure fSphi in 0-2PI or -2PI-0 range if shape crosses 0
    if (sPhi < 0) {
      fStartPhi = kTwoPi - std::fmod(std::fabs(sPhi), kTwoPi);
    } else {
      fStartPhi = std::fmod(sPhi, kTwoPi);
    }
    if (fStartPhi + fDeltaPhi > kTwoPi) {
      fStartPhi -= kTwoPi;
    }

    // Update Wedge
    fPhiWedge.SetStartPhi(fStartPhi);
  }

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckDPhiAngle(Precision dPhi)
  {
    if (dPhi >= kTwoPi - 0.5 * kAngTolerance) {
      fDeltaPhi = kTwoPi;
      fStartPhi = 0;
    } else {
      if (dPhi > 0) {
        fDeltaPhi = dPhi;
      } else {
        //        std::ostringstream message;
        //        message << "Invalid dphi.\n"
        //                << "Negative or zero delta-Phi (" << dPhi << ")\n";
        //        std::cerr<<"UnplacedTube::CheckDPhiAngle(): Fatal error: "<< message.str().c_str() <<"\n";
      }
    }
    // Update Wedge
    fPhiWedge.SetDeltaPhi(fDeltaPhi);
  }

  VECCORE_ATT_HOST_DEVICE
  PolyconeStruct() {}
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
