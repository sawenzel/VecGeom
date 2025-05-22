/*
 * PolyconeStruct.h
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

  bool fEqualRmax{false};
  bool fContinuityOverAll{false};
  bool fConvexityPossible{false};

  evolution::Wedge fPhiWedge;
  Precision fStartPhi{0.};
  Precision fDeltaPhi{0.};
  unsigned int fNz{0};

  Vector<PolyconeSection> fSections;
  Vector<Precision> fZs;
  PolyconeHistorical *fOriginal_parameters{nullptr};

  // Data member to hold the line segment that form Polycone boundary
  Vector<Vector3D<Precision>> fRMinTwoDVec;
  Vector<Vector3D<Precision>> fRMaxTwoDVec;
  Vector<Vector3D<Precision>> fTwoDVec;

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static bool ApproxEqual(const Precision &x, const Precision &y) { return vecCore::math::Abs(x - y) < kTolerance; }

  PolyconeStruct() = default;

  VECCORE_ATT_HOST_DEVICE
  PolyconeStruct(bool equalRmax, bool continuityOverAll, bool convexityPossible, Precision phiStart, Precision phiTotal,
                 unsigned int numSections, unsigned int numZPlanes, const Precision zPlaneR[],
                 const Precision rInnerR[], const Precision rOuterR[], AlignedAllocator &a)
      : fEqualRmax(equalRmax), fContinuityOverAll(continuityOverAll), fConvexityPossible(convexityPossible),
        fNz(numZPlanes), fSections(numSections, a), fZs(numSections + 1, a), fRMinTwoDVec(2 * numSections, a),
        fRMaxTwoDVec(2 * numSections, a), fTwoDVec(4 * numSections, a)
  {
    // These will set fStartPhi, fDeltaPhi and fPhiWedge
    SetAndCheckStartAndDeltaPhi(phiStart, phiTotal);

    // Create the sections

    // Calculate RMax of Polycone in order to determine convexity of sections
    Precision RMaxextent = rOuterR[0];
    for (unsigned int j = 1; j < numZPlanes; j++)
      if (rOuterR[j] > RMaxextent) RMaxextent = rOuterR[j];

    Precision prevZ = zPlaneR[0], prevRmax = 0., prevRmin = 0.;
    int dirZ = 1.;
    if (zPlaneR[1] < zPlaneR[0]) dirZ = -1.;

    size_t isection = 0, iz = 0;
    for (unsigned int i = 0; i < numZPlanes; ++i) {
      Precision rMin = rInnerR[i];
      Precision rMax = rOuterR[i];
      Precision z    = zPlaneR[i];

      // i has to be at least one to complete a section
      if (i > 0) {
        if (((z > prevZ + kTolerance) && (dirZ > 0)) || ((z < prevZ - kTolerance) && (dirZ < 0))) {
          Precision dz    = 0.5 * (z - prevZ);
          fZs[iz++]       = z;
          Precision shift = prevZ + dz;

          PolyconeSection &section = fSections[isection];
          section.fShift           = shift;
          section.fSolid.Init(prevRmin, prevRmax, rMin, rMax, dz, fStartPhi, fDeltaPhi);

          section.fConvex = !((rMax < prevRmax) || (rMax < RMaxextent) || (prevRmax < RMaxextent));

          fRMinTwoDVec[2 * isection + 0].Set(prevRmin, prevZ, 0.);
          fRMinTwoDVec[2 * isection + 1].Set(rMin, z, 0);
          fRMaxTwoDVec[2 * isection + 0].Set(prevRmax, prevZ, 0);
          fRMaxTwoDVec[2 * isection + 1].Set(rMax, z, 0.);
          isection++;
        }
      } else {
        fZs[iz++] = z;
      }
      prevZ    = z;
      prevRmin = rMin;
      prevRmax = rMax;
    }
    size_t counter = 0;
    for (auto val : fRMaxTwoDVec)
      fTwoDVec[counter++] = val;

    for (int k = fRMinTwoDVec.size() - 1; k >= 0; k--) {
      fTwoDVec[counter++] = fRMinTwoDVec[k];
    }
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  static size_t aligned_sizeof_data(size_t numSections)
  {
    // Sections data
    size_t aligned_size = AlignedAllocator::aligned_sizeof<PolyconeSection>(numSections, 0);
    // fZs
    aligned_size += AlignedAllocator::aligned_sizeof<Precision>(numSections + 1, 0);
    // fRMinTwoDVec, fRMaxTwoDVec
    aligned_size += 2 * AlignedAllocator::aligned_sizeof<Vector3D<Precision>>(2 * numSections, 0);
    // fTwoDVec
    aligned_size += AlignedAllocator::aligned_sizeof<Vector3D<Precision>>(4 * numSections, 0);
    return aligned_size;
  }

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
  void Dump()
  {
    printf("== PolyconeStruct at: %p", (void *)this);
    printf("   numZplanes: %u numSections: %lu, phiStart: %g phiDelta: %g\n", fNz, fSections.size(), fStartPhi,
           fDeltaPhi);
    printf("   fEqualRmax: %d fContinuityOverAll: %d fConvexityPossible: %d\n", fEqualRmax, fContinuityOverAll,
           fConvexityPossible);
    printf("   fZs: {");
    for (auto z : fZs)
      printf("  %g", z);
    printf("  }\n");
    auto dump_section = [](int isection, PolyconeSection const &section) {
      printf("   section %d  fShift: %g  fTubular: %d  fConvex: %d\n", isection, section.fShift, section.fTubular,
             section.fConvex);
      printf("      ");
      section.fSolid.Print();
      printf("\n");
    };
    int isection = 0;
    for (auto const &section : fSections)
      dump_section(isection++, section);
  }

  // a method to reconstruct "plane" section arrays for z, rmin and rmax
  template <typename PushableContainer>
  void ReconstructSectionArrays(PushableContainer &z_values, PushableContainer &rmin_values,
                                PushableContainer &rmax_values) const
  {
    // loop sections
    Precision prevZ = 0., prevRmax = 0., prevRmin = 0.;
    int iplane = 0;
    for (auto const &section : fSections) {
      Precision zmin = section.fShift - section.fSolid.fDz;
      Precision zmax = section.fShift + section.fSolid.fDz;
      Precision rmin = section.fSolid.fRmin1;
      Precision rmax = section.fSolid.fRmax1;
      if (iplane == 0) {
        prevZ    = zmin;
        prevRmin = rmin;
        prevRmax = rmax;
        z_values.push_back(zmin);
        rmin_values.push_back(rmin);
        rmax_values.push_back(rmax);
        iplane++;
      }
      // Add bottom Z plane if there is a radial discontinuity
      if (!ApproxEqual(rmin, prevRmin) || !ApproxEqual(rmax, prevRmax)) {
        z_values.push_back(zmin);
        rmin_values.push_back(rmin);
        rmax_values.push_back(rmax);
        iplane++;
      }
      // Add top Z plane
      prevZ    = zmax;
      prevRmin = section.fSolid.fRmin2;
      prevRmax = section.fSolid.fRmax2;
      z_values.push_back(prevZ);
      rmin_values.push_back(prevRmin);
      rmax_values.push_back(prevRmax);
      iplane++;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  void Init(Precision phiStart,       // initial phi starting angle
            Precision phiTotal,       // total phi angle
            unsigned int numZPlanes,  // number of z planes
            const Precision zPlane[], // position of z planes
            const Precision rInner[], // tangent distance to inner surface
            const Precision rOuter[])
  {

    SetAndCheckStartAndDeltaPhi(phiStart, phiTotal);
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

          Precision dz = (z - prevZ) / 2;
          fZs.push_back(z);
          int zi          = fZs.size() - 1;
          Precision shift = fZs[zi - 1] + 0.5 * (fZs[zi] - fZs[zi - 1]);

          PolyconeSection section;
          section.fShift = shift;
          section.fSolid.Init(prevRmin, prevRmax, rMin, rMax, dz, fStartPhi, fDeltaPhi);

          section.fConvex = !((rMax < prevRmax) || (rMax < RMaxextent) || (prevRmax < RMaxextent));

          fSections.push_back(section);
          fRMinTwoDVec.push_back(Vector3D<Precision>(prevRmin, prevZ, 0));
          fRMinTwoDVec.push_back(Vector3D<Precision>(rMin, z, 0));
          fRMaxTwoDVec.push_back(Vector3D<Precision>(prevRmax, prevZ, 0));
          fRMaxTwoDVec.push_back(Vector3D<Precision>(rMax, z, 0));
        }
      } else { // for i == 0 just push back first z plane
        fZs.push_back(z);
      }

      prevZ    = z;
      prevRmin = rMin;
      prevRmax = rMax;
    }
    for (auto val : fRMaxTwoDVec) {
      fTwoDVec.push_back(val);
    }

    for (int k = fRMinTwoDVec.size() - 1; k >= 0; k--) {
      fTwoDVec.push_back(fRMinTwoDVec[k]);
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
    // i is negative if zposition is out of range
    if (i == -1)
      i = 0;
    else if (i == -2)
      i = fSections.size() - 1;
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
      return fSections[index - 1].fSolid.fRmin2; // GetRmin2();
    else
      return fSections[index].fSolid.fRmin1; // GetRmin1();
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmaxAtPlane(int index) const
  {
    int nsect = fSections.size();
    assert(index >= 0 || index <= nsect);
    if (index == nsect)
      return fSections[index - 1].fSolid.fRmax2; // GetRmax2();
    else
      return fSections[index].fSolid.fRmax1; // GetRmax1();
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetZAtPlane(unsigned int index) const
  {
    assert(index <= fSections.size());
    return fZs[index];
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmin1AtSection(size_t index) const
  {
    assert(index < fSections.size());
    return fSections[index].fSolid.fRmin1;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmin2AtSection(size_t index) const
  {
    assert(index < fSections.size());
    return fSections[index].fSolid.fRmin2;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmax1AtSection(size_t index) const
  {
    assert(index < fSections.size());
    return fSections[index].fSolid.fRmax1;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmax2AtSection(size_t index) const
  {
    assert(index < fSections.size());
    return fSections[index].fSolid.fRmax2;
  }

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckSPhiAngle(Precision sPhi)
  {
    // Ensure fSphi in [0, 2PI)
    while (sPhi < 0.)
      sPhi += kTwoPi;
    fStartPhi = std::fmod(sPhi, kTwoPi);
    // Update Wedge
    fPhiWedge.SetStartPhi(fStartPhi);
    fPhiWedge.UpdateNormals();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckDPhiAngle(Precision dPhi)
  {
    if (dPhi <= 0. || dPhi > kTwoPi - kTolerance) {
      fStartPhi = 0;
      fDeltaPhi = kTwoPi;
    } else {
      fDeltaPhi = dPhi;
    }
    while (fStartPhi < 0)
      fStartPhi += kTwoPi;
    // Update Wedge
    fPhiWedge.SetDeltaPhi(fDeltaPhi);
    fPhiWedge.UpdateNormals();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetAndCheckStartAndDeltaPhi(Precision sPhi, Precision dPhi)
  {
    fStartPhi = sPhi;
    while (fStartPhi < 0)
      fStartPhi += kTwoPi;
    fStartPhi = std::fmod(sPhi, kTwoPi);

    fDeltaPhi = dPhi;
    if (dPhi <= 0. || dPhi > kTwoPi - kTolerance) {
      fStartPhi = 0;
      fDeltaPhi = kTwoPi;
    }
    // Update Wedge
    fPhiWedge.Set(fDeltaPhi, fStartPhi);
    fPhiWedge.UpdateNormals();
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
