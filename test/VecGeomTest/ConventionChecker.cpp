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

/* Function to Setup all the convention messages
 * With this interface it will be easy, if we want to put
 * some more conventions in future
 */
template <typename ImplT>
void ShapeTester<ImplT>::SetupConventionMessages()
{
  fScore               = 0;
  auto const &messages = vecgeom::test::ShapeConventionMessages();
  fConventionMessage.assign(messages.begin(), messages.end());
  fNumDisp = 1;
}

// Bridge the legacy ShapeTester entry point to the extracted helper layer.
// ShapeTester still owns sampling, reporting and score presentation, while the
// reusable contract predicates now live in ShapeContractChecks.h.
template <typename ImplT>
bool ShapeTester<ImplT>::ShapeConventionChecker()
{

  // Setting up Convention sMessages
  SetupConventionMessages();
  ClearErrors();

  // Generating Points and direction for
  // Inside, Surface, Outside fPoints
  CreatePointsAndDirections();

  vecgeom::test::ShapeContractSampleView samples;
  samples.points             = &fPoints;
  samples.directions         = &fDirections;
  samples.offset_inside      = fOffsetInside;
  samples.offset_surface     = fOffsetSurface;
  samples.offset_edge        = fOffsetEdge;
  samples.offset_outside     = fOffsetOutside;
  samples.max_points_inside  = fMaxPointsInside;
  samples.max_points_surface = fMaxPointsSurface;
  samples.max_points_edge    = fMaxPointsEdge;
  samples.max_points_outside = fMaxPointsOutside;

  auto distance_to_out = [this](ImplT const *volume, const Vec_t &point, const Vec_t &direction, Vec_t &normal) {
    bool convex = false;
    return CallDistanceToOut(volume, point, direction, normal, convex);
  };

  // Preserve the current console/debug output path while delegating the actual
  // violation bookkeeping to ShapeCheckResult through the helper sink.
  auto make_reporter = [this](int &nError) {
    return [this, &nError](const vecgeom::test::ShapeRecordDecision &decision, const vecgeom::test::ShapeCheckContext &,
                           const std::string &message, const Vec_t &point, const Vec_t &direction, Precision distance) {
      DisplayRecordedError(decision, &nError, point, direction, distance, message);
    };
  };

  vecgeom::test::ShapeContractCheckSummary summary;
  int surface_errors = 0;
  int inside_errors  = 0;
  int outside_errors = 0;

  vecgeom::test::ShapeContractViolationSink surface_sink(fCheckResult, fNumDisp, make_reporter(surface_errors));
  vecgeom::test::ShapeContractViolationSink inside_sink(fCheckResult, fNumDisp, make_reporter(inside_errors));
  vecgeom::test::ShapeContractViolationSink outside_sink(fCheckResult, fNumDisp, make_reporter(outside_errors));

  summary.surface_points_passed = vecgeom::test::CheckSurfaceConventions(fVolume, samples, fSolidTolerance,
                                                                         distance_to_out, surface_sink, summary.score);
  summary.inside_points_passed =
      vecgeom::test::CheckInsideConventions(fVolume, samples, distance_to_out, inside_sink, summary.score);
  summary.outside_points_passed =
      vecgeom::test::CheckOutsideConventions(fVolume, samples, distance_to_out, outside_sink, summary.score);

  fScore = summary.score;
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "Generated Score : " << fScore << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  if (summary.Passed()) {
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "---------- Shape Conventions Passed -------------" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
  }

  GenerateConventionReport();

  return summary.Passed();
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
  return ShapeConventionChecker();
}
