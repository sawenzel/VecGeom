//===-- test/shape_tester/ShapeTester.h ----------------------------*- C++ -*-===//
//
// Definition of the batch solid test.
//
// The new helper-level contract runner is documented in docs/shape_testing.md.
// This legacy wrapper still owns the whole-solid workflow, but many of the
// reusable predicates now live in ShapeContractChecks.h and are shared with
// ShapeContractTest.
//

#ifndef ShapeTester_hh
#define ShapeTester_hh

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/RNG.h"
#include "VecGeomTest/ApproxEqual.h"
#include "VecGeomTest/ShapeCheckResult.h"
#include "VecGeomTest/ShapeContractChecks.h"
#include "VecGeomTest/ShapeSampleSet.h"

#ifdef VECGEOM_ROOT
#include "Visualizer.h"
#endif

using vecgeom::Precision;
using Vec_t = vecgeom::Vector3D<Precision>;

template <typename ImplT>
class ShapeTester {

  // ALL MEMBER FUNCTIONS FIRST
public:
  ShapeTester();
  ~ShapeTester();

  void setStat(bool _stat) { fStat = _stat; }
  void setDebug(bool _debug) { fDebug = _debug; }

  int Run(ImplT const *testVolume);
  void Run(ImplT const *testVolume, const char *type);
  int RunMethod(ImplT const *testVolume, std::string fMethod1);
  inline void SetFilename(const std::string &newFilename) { fFilename = newFilename; }
  inline void SetMaxPoints(const int newMaxPoints) { fMaxPoints = newMaxPoints; }
  inline void SetMethod(const std::string &newMethod) { fMethod = newMethod; }
  inline void SetInsidePercent(const Precision percent) { fInsidePercent = percent; }
  inline void SetGrazingTolerance(const Precision tolerance) { fGrazingTolerance = tolerance; }
  inline void SetErrorOnZeroDoutGrazing(bool flag) { fErrorOnZeroDoutGrazing = flag; }
  inline void SetOutsidePercent(const Precision percent) { fOutsidePercent = percent; }
  inline void SetEdgePercent(const Precision percent) { fEdgePercent = percent; }
  inline void SetOutsideMaxRadiusMultiple(const Precision percent) { fOutsideMaxRadiusMultiple = percent; }
  inline void SetOutsideRandomDirectionPercent(const Precision percent) { fOutsideRandomDirectionPercent = percent; }
  inline void SetSaveAllData(const bool safe) { fIfSaveAllData = safe; }
  inline void SetSampleSeed(const unsigned long seed)
  {
    fSampleSeed    = seed;
    fUseSampleSeed = true;
  }
  inline void SetSolidTolerance(const Precision value) { fSolidTolerance = value; }
  inline void SetSolidFarAway(const Precision value) { fSolidFarAway = value; }
  inline void SetTestBoundaryErrors(bool flag) { fTestBoundaryErrors = flag; }
  void SetFolder(const std::string &newFolder);
  void SetVerbose(int verbose) { fVerbose = verbose; }
  inline int GetMaxPoints() const { return fMaxPoints; }
  inline Vec_t GetPoint(int index) { return fPoints[index]; }
  inline void SetNumberOfScans(int num) { fGNumberOfScans = num; }

  /* Keeping this Function as public to allow, if somebody just want
   * to do the Convention Check
   */
  bool RunConventionChecker(ImplT const *testVolume);
  void EnableDebugger(bool val); // function to enable or disable visualization for debugging

private:
  void SetDefaults();
  int SaveVectorToExternalFile(const std::vector<double> &vector, const std::string &fFilename);
  int SaveVectorToExternalFile(const std::vector<Vec_t> &vector, const std::string &fFilename);
  int SaveLegend(const std::string &fFilename);
  int SaveDifLegend(const std::string &fFilename);
  int SaveDoubleResults(const std::string &fFilename);
  int SaveDifDoubleResults(const std::string &fFilename);
  int SaveVectorResults(const std::string &fFilename);
  int SaveDifVectorResults(const std::string &fFilename);
  int SaveDifVectorResults1(const std::string &fFilename);
  std::string PrintCoordinates(const Vec_t &vec, const std::string &delimiter, int precision = 4);
  std::string PrintCoordinates(const Vec_t &vec, const char *delimiter, int precision = 4);
  void PrintCoordinates(std::stringstream &ss, const Vec_t &vec, const std::string &delimiter, int precision = 4);
  void PrintCoordinates(std::stringstream &ss, const Vec_t &vec, const char *delimiter, int precision = 4);

  template <class T>
  void VectorDifference(const std::vector<T> &first, const std::vector<T> &second, std::vector<T> &result);

  void VectorToDouble(const std::vector<Vec_t> &vectorUVector, std::vector<double> &vectorDouble);
  void BoolToDouble(const std::vector<bool> &vectorBool, std::vector<double> &vectorDouble);
  int CountDoubleDifferences(const std::vector<double> &differences);
  int CountDoubleDifferences(const std::vector<double> &differences, const std::vector<double> &values1,
                             const std::vector<double> &values2);

  void FlushSS(std::stringstream &ss);
  void Flush(const std::string &s);

  Vec_t GetPointOnOrb(Precision r);
  Vec_t GetRandomDirection();
  Vec_t GetRandomDirectionPerp(Vec_t const &normal);

  int TestBoundaryPrecision(int mode);
  int TestConsistencySolids();
  int TestInsidePoint();
  int TestOutsidePoint();
  int TestSurfacePoint();

  int TestNormalSolids();

  int TestSafetyFromInsideSolids();
  int TestSafetyFromOutsideSolids();
  int ShapeSafetyFromInside(int max);
  int ShapeSafetyFromOutside(int max);

  void PropagatedNormal(const Vec_t &point, const Vec_t &direction, Precision distance, Vec_t &normal);
  void PropagatedNormalU(const Vec_t &point, const Vec_t &direction, Precision distance, Vec_t &normal);
  int TestDistanceToInSolids();
  int TestAccuracyDistanceToIn(Precision dist);
  int ShapeDistances();
  int TestFarAwayPoint();
  int TestDistanceToOutSolids();
  int ShapeNormal();
  int TestXRayProfile();
  int XRayProfile(double theta = 45, int nphi = 15, int ngrid = 1000, bool useeps = true);
  int Integration(double theta = 45, double phi = 45, int ngrid = 1000, bool useeps = true, int npercell = 1,
                  bool graphics = true);
  Precision CrossedLength(const Vec_t &point, const Vec_t &dir, bool useeps);
  void CreatePointsAndDirections();

  void CompareAndSaveResults(const std::string &fMethod, double resG, double resR, double resU);
  int SaveResultsToFile(const std::string &fMethod);
  void SavePolyhedra(const std::string &fMethod);
  double MeasureTest(int (ShapeTester::*funcPtr)(int), const std::string &fMethod);
  double NormalizeToNanoseconds(double time);

  int TestMethod(int (ShapeTester::*funcPtr)());
  int TestMethodAll();

  // This was needed because of different signature in VecGeom vs. USolids
  Precision CallDistanceToOut(ImplT const *vol, const Vec_t &point, const Vec_t &dir, Vec_t &normal, bool convex) const;

  template <typename Type>
  inline Type RandomRange(Type min, Type max)
  {
    Type rand = min + (max - min) * fRNG.uniform();
    return rand;
  }

  inline Precision RandomIncrease()
  {
    Precision tolerance = vecgeom::kTolerance;
    Precision rand      = -1 + 2 * fRNG.uniform();
    Precision dif       = tolerance * 0.1 * rand;
    return dif;
  }

  /* Private functions for Convention Checker, These functions never need
   * to be called from Outside the class
   */
  void PrintConventionMessages();  // Function to print convention messages
  void GenerateConventionReport(); // Function to generate Convention Report
  void SetupConventionMessages();  // Function to setup convention messages
  bool ShapeConventionChecker();   // Function that call other core convention checking function
  void SetNumDisp(int);            // Function to set num. of points to be displayed during convention failure

protected:
  Vec_t GetRandomPoint() const;
  double GaussianRandom(const double cutoff) const;
  // Legacy error display path kept separate from structured recording so the
  // extracted contract helpers can reuse the same console/debug output without
  // depending on ShapeTester internals for result storage.
  void DisplayRecordedError(const vecgeom::test::ShapeRecordDecision &decision, int *nError, const Vec_t &p,
                            const Vec_t &v, Precision distance, const std::string &comment);
  void ReportError(int *nError, Vec_t &p, Vec_t &v, Precision distance,
                   std::string comment); //, std::ostream &fLogger );
  void ClearErrors();
  int CountErrors() const;

  // ALL DATA MEMBERS
protected:
  int fMaxPoints;                           // Maximum num. of points to be generated for ShapeTester Tests.
  int fVerbose;                             // Variable to set verbose
  Precision fInsidePercent;                 // Percentage of inside points
  Precision fOutsidePercent;                // Percentage of outside points
  Precision fEdgePercent;                   // Percentage of edge points
  Precision fOutsideMaxRadiusMultiple;      // Range of outside points
  Precision fOutsideRandomDirectionPercent; // Percentage of outside random direction
  Precision fGrazingTolerance; // Tolerance for rays staring on surface to be perpendicular to the surface normal

  // XRay profile statistics
  int fGNumberOfScans;         // data member to store the number of different scan angle used for XRay profile
  double fGCapacitySampled;    // data member to store calculated capacity
  double fGCapacityAnalytical; // data member to store analytical capacity
  double fGCapacityError;      // data member to store error between above two.

  std::string fMethod; // data member to store the name of method to be executed

  vecgeom::test::ShapeCheckResult fCheckResult; // structured list of error categories and occurrences

private:
  std::vector<Vec_t> fPoints;     // STL vector to store the points generated for various tests of ShapeTester
  std::vector<Vec_t> fDirections; // STL vector to store the directions generated for corresponding points.

  ImplT const *fVolume;      // Pointer that owns shape object.
  std::string fVolumeString; // data member to store the name of volume;

  std::vector<Precision> fResultPrecision; // stl vector for storing the double/float results
  std::vector<Vec_t> fResultVector;        // stl vector for storing the vector results

  int fOffsetSurface;    // offset of surface points
  int fOffsetInside;     // offset of inside points
  int fOffsetOutside;    // offset of outside points
  int fOffsetEdge;       // offset of edge points
  int fMaxPointsInside;  // Number of inside points
  int fMaxPointsOutside; // Number of outside points
  int fMaxPointsSurface; // Number of surface points
  int fMaxPointsEdge;    // Number of edge points

  std::ostream *fLog; // Pointer to the directory storing all the log file for different tests

  std::string fFolder;   // Name of the log folder
  std::string fFilename; // name of the file name depending on the method under test

  // Save only differences
  bool fIfSaveAllData; // save alldata, big files
  // take more time, but not affect performance measures
  bool fDefinedNormal;          // bool variable to skip normal calculation if it does not exist in the shape
  bool fIfException;            // data memeber to abort ShapeTester if any error found
  bool fTestBoundaryErrors;     // Enable testing boundary errors
  bool fErrorOnZeroDoutGrazing; // If this is set, generate errors for grazing rays returning zero distance to out

  // Added data member required for convention checker
  std::vector<std::string> fConventionMessage; // STL vector for convention error messages.
  int fScore;      // an error code generate if conventions not followed, 0 mean convenetion followed.
  int fNumDisp;    // number of points to be displayed in case a shape is not following conventions.
  bool fVisualize; // Flag to be set or unset by EnableDebugger() function that user will
  // call with true parameter if want to see visualization in case of some mismatch
  Precision fSolidTolerance; // Tolerance on boundary declared by solid (default kTolerance)
  Precision fSolidFarAway;   // Distance to shoot points at from solid in TestFarAwayPoints
  unsigned long fSampleSeed; // Optional deterministic sampling seed for extracted ShapeSampleSet generation
  bool fUseSampleSeed;       // Whether to reseed both local and global RNGs before sampling
#ifdef VECGEOM_ROOT
  vecgeom::Visualizer fVisualizer; // Visualizer object to visualize the geometry if fVisualize is set.
#endif
  vecgeom::RNG fRNG;
  bool fStat;  // data member to show the statistic visualtion if set to true
  bool fDebug; // data member to visualized the shape and first mismatched point with directions
};

#endif
