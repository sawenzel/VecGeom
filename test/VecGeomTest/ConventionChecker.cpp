/// @file ConventionChecker.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

/* This file contains implementation of additional functions added to ShapeTester,
 * to have the shape convention checking feature.
 */

#include "ShapeTester.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TGeoBBox.h"
#include "TGeoParaboloid.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoParaboloid.h"
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGraph2D.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TROOT.h"
#include "TAttMarker.h"
#include "TF1.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TView3D.h"
#include "TVirtualPad.h"
#endif

#undef NDEBUG

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using vecgeom::kInfLength;
using Vec_t = vecgeom::Vector3D<Precision>;

// Function to set the number of Points to be displayed in case of convention not followed
template <typename ImplT>
void ShapeTester<ImplT>::SetNumDisp(int num)
{
  fNumDisp = num;
}

// Helper function taken from ApproxEqual.h
template <typename ImplT>
bool ShapeTester<ImplT>::ApproxEqual(const double &x, const double &y)
{
  if (x == y) {
    return true;
  } else if (x * y == 0.0) {
    double diff = std::fabs(x - y);
    return diff < kApproxEqualTolerance;
  } else {
    double diff  = std::fabs(x - y);
    double abs_x = std::fabs(x), abs_y = std::fabs(y);
    return diff / (abs_x + abs_y) < kApproxEqualTolerance;
  }
}

template <typename ImplT>
bool ShapeTester<ImplT>::ApproxEqual(const float &x, const float &y)
{
  if (x == y) {
    return true;
  } else if (x * y == 0.0) {
    float diff = std::fabs(x - y);
    return diff < kApproxEqualTolerance;
  } else {
    float diff  = std::fabs(x - y);
    float abs_x = std::fabs(x), abs_y = std::fabs(y);
    return diff / (abs_x + abs_y) < kApproxEqualTolerance;
  }
}

// Return true if the 3vector check is approximately equal to target
template <typename ImplT>
template <class Vec_t>
bool ShapeTester<ImplT>::ApproxEqual(const Vec_t &check, const Vec_t &target)
{
  return (ApproxEqual(check.x(), target.x()) && ApproxEqual(check.y(), target.y()) &&
          ApproxEqual(check.z(), target.z()))
             ? true
             : false;
}

/* Function to Setup all the convention messages
 * With this interface it will be easy, if we want to put
 * some more conventions in future
 */
template <typename ImplT>
void ShapeTester<ImplT>::SetupConventionMessages()
{
  // For Surface Points
  fScore = 0;                                                                                    // index
  fConventionMessage.push_back("DistanceToIn()  : For Point On Surface and Entering the Shape"); // 0
  fConventionMessage.push_back("DistanceToIn()  : For Point On Surface and Exiting the Shape");  // 1
  fConventionMessage.push_back("DistanceToOut() : For Point On Surface and Exiting the Shape");  // 2
  fConventionMessage.push_back("DistanceToOut() : For Point On Surface and Entering the Shape"); // 3
  fConventionMessage.push_back("SafetyToIn()    : For Point On Surface ");                       // 4
  fConventionMessage.push_back("SafetyToOut()   : For Point On Surface ");                       // 5

  // For Inside Points
  fConventionMessage.push_back("DistanceToIn()  : For Inside Point"); // 6
  fConventionMessage.push_back("DistanceToOut() : For Inside Point"); // 7
  fConventionMessage.push_back("SafetyToIn()    : For Inside Point"); // 8
  fConventionMessage.push_back("SafetyToOut()   : For Inside Point"); // 9

  // Outside Points
  fConventionMessage.push_back("DistanceToIn()  : For Outside Point"); // 10
  fConventionMessage.push_back("DistanceToOut() : For Outside Point"); // 11
  fConventionMessage.push_back("SafetyToIn()    : For Outside Point"); // 12
  fConventionMessage.push_back("SafetyToOut()   : For Outside Point"); // 13

  fNumDisp = 1;
}

// Funtion to check conventions for Surface Points
template <typename ImplT>
bool ShapeTester<ImplT>::ShapeConventionSurfacePoint()
{
  int nError                     = 0;
  bool surfPointConventionPassed = true;
  for (int i = 0; i < fMaxPointsSurface + fMaxPointsEdge; i++) { // test SamplePointOnSurface()
    Vec_t point     = fPoints[fOffsetSurface + i];
    Vec_t direction = fDirections[fOffsetSurface + i];
    if (fVolume->Inside(point) != vecgeom::EInside::kSurface) {
      ReportError(&nError, point, direction, 0., "For Surface point, Inside says that the Point is not on the Surface");
    }

    // Point on Surface and moving inside
    Vec_t normal(0., 0., 0.);
    bool validNormal = fVolume->Normal(point, normal);

    Precision Dist = fVolume->DistanceToIn(point, direction);
    // if (Dist >= kInfLength) Dist = kInfLength;

    int indx = 0;

    // Conventions Check for DistanceToIn
    // A.G Note that for points on edges (often in single precision mode) the ray
    // may not hit the solid even if `approaching` (i.e direction.Dot(normal) < 0)
    if (direction.Dot(normal) < 0. && Dist < kInfLength) // particle is entering into the shape
    {
      bool ok = fabs(Dist * direction.Dot(normal)) <= fSolidTolerance;
      if (validNormal && !ok) {
        fScore |= (1 << indx);
        surfPointConventionPassed &= false;
        ReportError(
            &nError, point, direction, Dist,
            "DistanceToIn for Surface Point entering into the Shape should be 0 within tolerance (VecGeom convention)");
        Dist = fVolume->DistanceToIn(point, direction);
      }
    }

    indx = 1;
    // Consider all the shapes as "Not convex" even if it is !!.
    bool convexShape = false; // convexShape = IsConvex()
    // Point on Surface and moving outside
    if (direction.Dot(normal) > 0. && Dist == kInfLength) // particle is exiting from the shape
    {
      if (convexShape) {
        if (!ApproxEqual(Dist, kInfLength)) {
          fScore |= (1 << indx);
          surfPointConventionPassed &= false;
        }
      } else {
        // If the shape is not convex then DistanceIn is distance to sNext Intersection
        // It may possible that it will not hit the shape again, in that case, Distance should be infinity
        // So overall distance must be greater than zero.
        if (!(Dist > 0.)) {
          ReportError(&nError, point, direction, Dist,
                      "DistanceToIn for Surface Point exiting the Shape should be > 0.");
          fScore |= (1 << indx);
          surfPointConventionPassed &= false;
        }
      }
    }

    // Conventions check for DistanceToOut
    indx = 2;
    Vec_t norm(0., 0., 0.);
    bool convex = false;
    Dist        = CallDistanceToOut(fVolume, point, direction, norm, convex);
    // if (Dist >= kInfLength) Dist = kInfLength;
    if (direction.Dot(normal) > 0. && Dist == kInfLength) {
      // particle is exiting from the shape
      bool ok = (Dist * direction.Dot(normal)) <= fSolidTolerance;
      if (!ok) {
        fScore |= (1 << indx);
        surfPointConventionPassed &= false;
        ReportError(&nError, point, direction, Dist,
                    "DistanceToOut for Surface Point exiting the shape should be <= tolerance (VecGeom convention)");
      }
    }

    indx = 3;
    if (direction.Dot(normal) < 0.) // particle is entering the shape
    {
      if (!(Dist > 0.)) {
        // validNormal: ignore points on corners (edges are still ok)
        if (validNormal) {
          ReportError(&nError, point, direction, Dist,
                      "DistanceToOut for Surface Point entering into the Shape should be > 0.");

          Dist = CallDistanceToOut(fVolume, point, direction, norm, convex);
          fScore |= (1 << indx);
          surfPointConventionPassed &= false;
        }
      }
    }

    indx = 4;
    // Conventions check for SafetyFromOutside
    Dist = fVolume->SafetyToIn(point);
    // if (Dist >= kInfLength) Dist = kInfLength;
    {
      bool ok = Dist <= fSolidTolerance;
      if (!ok) {
        fScore |= (1 << indx);
        surfPointConventionPassed &= false;
        ReportError(&nError, point, direction, Dist,
                    "SafetyFromOutside for Surface Point should be 0 (VecGeom convention)");
      }
    }

    indx = 5;
    // Conventions check for SafetyToOut
    Dist = fVolume->SafetyToOut(point);
    // if (Dist >= kInfLength) Dist = kInfLength;
    {
      bool ok = Dist <= fSolidTolerance;
      if (!ok) {
        fScore |= (1 << indx);
        surfPointConventionPassed &= false;
        ReportError(&nError, point, direction, Dist,
                    "SafetyToOut for Surface Point should be <= tolerance (VecGeom convention)");
        Dist = fVolume->SafetyToOut(point);
      }
    }
  }

  return surfPointConventionPassed;
}

// Function to check conventions for Inside points
template <typename ImplT>
bool ShapeTester<ImplT>::ShapeConventionInsidePoint()
{

  int nError = 0;
  Precision Dist;

  bool insidePointConventionPassed = true;

  for (int i = 0; i < fMaxPointsInside; i++) { // test SamplePointOnSurface()
    Vec_t point     = fPoints[fOffsetInside + i];
    Vec_t direction = fDirections[fOffsetInside + i];
    if (fVolume->Inside(point) != vecgeom::EInside::kInside) {
      ReportError(&nError, point, direction, 0., "For Inside point, Inside function says that the Point is not inside");
    }

    // Convention Check for DistanceToIn
    int indx = 6;
    Dist     = fVolume->DistanceToIn(point, direction);
    // if (Dist >= kInfLength) Dist = kInfLength;
    {
      bool ok = Dist < 0.;
      if (!ok) {
        fVolume->Inside(point);
        Dist = fVolume->DistanceToIn(point, direction);
        ReportError(&nError, point, direction, Dist,
                    "DistanceToIn for Inside Point should be Negative (-1.) (Wrong side)");
        fScore |= (1 << indx);
        insidePointConventionPassed &= false;
      }
    }

    indx = 7;
    // Convention Check for DistanceToOut
    Vec_t norm(0., 0., 0.);
    bool convex = false;
    Dist        = CallDistanceToOut(fVolume, point, direction, norm, convex);
    // if (Dist >= kInfLength) Dist = kInfLength;
    if (Dist == kInfLength) {
      ReportError(&nError, point, direction, Dist, "DistanceToOut for Inside Point can never be Infinity");
      fScore |= (1 << indx);
      insidePointConventionPassed &= false;
    }

    indx = 8;
    // Conventions Check for SafetyFromOutside
    Dist = fVolume->SafetyToIn(point);
    // if (Dist >= kInfLength) Dist = kInfLength;
    {
      bool ok = (Dist < 0.);
      if (!ok) {
        ReportError(&nError, point, direction, Dist,
                    "SafetyFromOutside for Inside Point should be Negative (-1.) (Wrong side, VecGeom conv)");
        fScore |= (1 << indx);
        insidePointConventionPassed &= false;
      }
    }

    indx = 9;
    // Conventions Check for SafetyToOut
    Dist = fVolume->SafetyToOut(point);
    // if (Dist >= kInfLength) Dist = kInfLength;
    if (!(Dist > 0.)) {
      ReportError(&nError, point, direction, Dist, "SafetyToOut for Inside Point should be > 0.");
      Dist = fVolume->SafetyToOut(point);
      fScore |= (1 << indx);
      insidePointConventionPassed &= false;
    }
  }

  return insidePointConventionPassed;
}

// Function to check conventions for outside points
template <typename ImplT>
bool ShapeTester<ImplT>::ShapeConventionOutsidePoint()
{
  int nError = 0;
  Precision Dist, DistBB;

  bool outsidePointConventionPassed = true;

  for (int i = 0; i < fMaxPointsOutside; i++) { // test SamplePointOnSurface()
    Vec_t point = fPoints[fOffsetOutside + i];
    Vec_t pointBB;
    Vec_t direction = fDirections[fOffsetOutside + i];
    if (fVolume->Inside(point) != vecgeom::EInside::kOutside) {
      ReportError(&nError, point, direction, 0.,
                  "For Outside point, Inside function says that the Point is not Outside");
    }

    int indx = 10;
    // Convention Check for DistanceToIn
    DistBB  = fVolume->GetUnplacedVolume()->ApproachSolid(point, 1 / direction);
    pointBB = point + DistBB * direction;
    Dist    = fVolume->DistanceToIn(pointBB, direction) + DistBB;
    // if (Dist >= kInfLength) Dist = kInfLength;
    if (!(Dist > 0.)) {
      ReportError(&nError, point, direction, Dist, "DistanceToIn for Outside Point should be > 0.");
      fScore |= (1 << indx);
      outsidePointConventionPassed &= false;
    }

    indx = 11;
    // Convention Check for DistanceToOut
    Vec_t norm(0., 0., 0.);
    bool convex = false;
    Dist        = CallDistanceToOut(fVolume, point, direction, norm, convex);
    // if (Dist >= kInfLength) Dist = kInfLength;
    {
      bool ok = (Dist < 0.);
      if (!ok) {
        Dist = CallDistanceToOut(fVolume, point, direction, norm, convex);
        ReportError(&nError, point, direction, Dist,
                    "DistanceToOut for Outside Point should be Negative (-1.) (Wrong side).");
        fScore |= (1 << indx);
        outsidePointConventionPassed &= false;
      }
    }

    indx = 12;
    // Conventions Check for SafetyFromOutside
    Dist = fVolume->SafetyToIn(point);
    // if (Dist >= kInfLength) Dist = kInfLength;
    if (!(Dist > 0.)) {
      ReportError(&nError, point, direction, Dist, "SafetyToIn for Outside Point should be > 0.");
      Dist = fVolume->SafetyToIn(point);
      fScore |= (1 << indx);
      outsidePointConventionPassed &= false;
    }

    indx = 13;
    // Conventions Check for SafetyToOut
    Dist = fVolume->SafetyToOut(point);
    // if (Dist >= kInfLength) Dist = kInfLength;
    {
      bool ok = (Dist < 0.);
      if (!ok) {
        ReportError(&nError, point, direction, Dist,
                    "SafetyToOut for Outside Point should be Negative (-1) (Wrong side)");
        Dist = fVolume->SafetyToOut(point);
        fScore |= (1 << indx);
        outsidePointConventionPassed &= false;
      }
    }
  }

  return outsidePointConventionPassed;
}

// Function that will call the above three functions to do the convention check
template <typename ImplT>
bool ShapeTester<ImplT>::ShapeConventionChecker()
{

  // Setting up Convention sMessages
  SetupConventionMessages();

  // Generating Points and direction for
  // Inside, Surface, Outside fPoints
  CreatePointsAndDirections();

  bool surfacePointConventionResult = ShapeConventionSurfacePoint();
  bool insidePointnConventionResult = ShapeConventionInsidePoint();
  bool outsidePointConventionResult = ShapeConventionOutsidePoint();
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "Generated Score : " << fScore << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  if (surfacePointConventionResult && insidePointnConventionResult && outsidePointConventionResult) {
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "---------- Shape Conventions Passed -------------" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
  }

  GenerateConventionReport();

  return true;
}

// Function to print all the conventions messages
template <typename ImplT>
void ShapeTester<ImplT>::PrintConventionMessages()
{

  for (auto i : fConventionMessage)
    std::cout << i << std::endl;
}

// Functions to generate Convention Report at the end
template <typename ImplT>
void ShapeTester<ImplT>::GenerateConventionReport()
{

  int n     = fScore;
  int index = -1;
  if (fScore) {
    std::cout << "\033[1;39m";
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "--------- Following ShapeConventions are Not Followed ---------" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "\033[0m";
    while (n > 0) {
      index++;
      if (n % 2) {
        // std::cout << index << "  ";

        std::cout << "\033[1;31m " << fConventionMessage[index] << "\033[0m" << std::endl;
      }
      n /= 2;
    }
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "--- Please refer to convention document on the repository -----" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "-------------- Continuing Shape Tester tests ------------------" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
  }
}

/* Public interface to run convention checker.
 * This interface is intentionally left public, so as to allow, if one want to call
 * just the convention checker without the ShapeTester's tests.
 */
template <typename ImplT>
bool ShapeTester<ImplT>::RunConventionChecker(ImplT const *testVolume)
{
  fVolume = testVolume;
  ShapeConventionChecker();

  return true;
}
