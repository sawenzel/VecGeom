//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// SBTperformance.hh
//
// Definition of the batch solid test
//

#ifndef SBTperformance_hh
#define SBTperformance_hh

#include <iostream>
#include "G4VSolid.hh"
#include "G4Orb.hh"
#include "G4ThreeVector.hh"
#include "VUSolid.hh"
#include "TGeoShape.h"

#include "UMultiUnion.hh"
#include "G4UnionSolid.hh"

#include "Randomize.hh"

class SBTVisManager;

class SBTperformance {

public:
	SBTperformance();
	~SBTperformance();
	void SetDefaults();

	void Run(G4VSolid *testVolume, std::ofstream &logger);

	G4int DrawError( const G4VSolid *testVolume, std::istream &logger, const G4int errorIndex,
		SBTVisManager *visManager ) const;

	inline void SetFilename( const std::string &newFilename ) { filename = newFilename; }
	inline void SetMaxPoints( const G4int newMaxPoints ) { maxPoints = newMaxPoints; }
	inline void SetRepeat( const G4int newRepeat ) { repeat = newRepeat; }
	inline void SetMethod( const std::string &newMethod ) { method = newMethod; }
	inline void SetInsidePercent( const G4double percent ) { insidePercent = percent; }
	inline void SetOutsidePercent( const G4double percent ) { outsidePercent = percent; }

	inline void SetOutsideMaxRadiusMultiple( const G4double percent ) { outsideMaxRadiusMultiple = percent; }
	inline void SetOutsideRandomDirectionPercent( const G4double percent ) { outsideRandomDirectionPercent = percent; }
	inline void SetDifferenceTolerance( const G4double tolerance ) { differenceTolerance = tolerance; }
	void SetFolder( const std::string &newFolder );

	inline G4int GetMaxPoints() const { return maxPoints; }
	inline G4int GetRepeat() const { return repeat; }

	int SaveVectorToMatlabFile(const std::vector<double> &vector, const std::string &filename);
	int SaveVectorToMatlabFile(const std::vector<UVector3> &vector, const std::string &filename);
	int SaveLegend(const std::string &filename);
	int SaveDoubleResults(const std::string &filename);
	int SaveVectorResults(const std::string &filename);

	std::string PrintCoordinates (const UVector3 &vec, const std::string &delimiter, int precision=4);
	std::string PrintCoordinates (const UVector3 &vec, const char *delimiter, int precision=4);
	void PrintCoordinates (std::stringstream &ss, const UVector3 &vec, const std::string &delimiter, int precision=4);
	void PrintCoordinates (std::stringstream &ss, const UVector3 &vec, const char *delimiter, int precision=4);

	template <class T> void VectorDifference(const std::vector<T> &first, const std::vector<T> &second, std::vector<T> &result);
	
	void VectorToDouble(const std::vector<UVector3> &vectorUVector, std::vector<double> &vectorDouble);

  void BoolToDouble(const std::vector<bool> &vectorBool, std::vector<double> &vectorDouble);
	
	int CountDoubleDifferences(const std::vector<double> &differences);
	int CountDoubleDifferences(const std::vector<double> &differences, const std::vector<double> &values1, const std::vector<double> &values2);

//	int CompareVectorDifference(std::string filename);

protected:
	G4ThreeVector	GetRandomPoint() const;
	G4double	GaussianRandom(const G4double cutoff) const;

	void	ReportError( G4int *nError, const G4ThreeVector p, 
		const G4ThreeVector v, G4double distance,
		const G4String comment, std::ostream &logger );
	void 	ClearErrors();		
	G4int 	CountErrors() const;		

protected:

	int		maxPoints, repeat;
	G4double	insidePercent, outsidePercent, outsideMaxRadiusMultiple, outsideRandomDirectionPercent, differenceTolerance;
	std::string method;


	typedef struct sSBTperformanceErrorList {
		G4String	message;
		G4int		nUsed;
		struct sSBTperformanceErrorList *next;
	} SBTperformanceErrorList;

	SBTperformanceErrorList *errorList;

private:
	int numCheckPoints;

	int compositeCounter;

	void FlushSS(std::stringstream &ss);
	void Flush(const std::string &s);

	G4VSolid *volumeGeant4;
	VUSolid *volumeUSolids;
	TGeoShape *volumeROOT;
	std::string volumeString;

	void SetupSolids(G4VSolid *testVolume);

	void ConvertMultiUnionFromGeant4(UMultiUnion &multiUnion, G4UnionSolid &solid, std::string &rootComposite);

	std::vector<UVector3> points, directions;
	std::vector<UVector3> resultVectorGeant4;
	std::vector<UVector3> resultVectorRoot;
	std::vector<UVector3> resultVectorUSolids, resultVectorDifference;
	std::vector<double> resultDoubleGeant4, resultDoubleRoot, resultDoubleUSolids, resultDoubleDifference;
  std::vector<bool> resultBoolGeant4, resultBoolUSolids, resultBoolDifference;

	int offsetSurface, offsetInside, offsetOutside;
	int maxPointsInside, maxPointsOutside, maxPointsSurface;
	std::ostream *log, *perftab, *perflabels;
	std::string folder;
	std::string filename;

//	UVector3 GetVectorOnOrb(G4Orb& orb, UVector3& norm);
	UVector3 GetRandomDirection();

	void TestInsideGeant4(int iteration);
	void TestInsideUSolids(int iteration);
	void TestInsideROOT(int iteration);

	void TestNormalGeant4(int iteration);
	void TestNormalUSolids(int iteration);
	void TestNormalROOT(int iteration);

	void TestSafetyFromInsideGeant4(int iteration);
	void TestSafetyFromInsideUSolids(int iteration);
	void TestSafetyFromInsideROOT(int iteration);

	void TestSafetyFromOutsideGeant4(int iteration);
	void TestSafetyFromOutsideUSolids(int iteration);
	void TestSafetyFromOutsideROOT(int iteration);

	void PropagatedNormal(const G4ThreeVector &point, const G4ThreeVector &direction, double distance, G4ThreeVector &normal);

	void TestDistanceToInUSolids(int iteration);
	void TestDistanceToInGeant4(int iteration);
	void TestDistanceToInROOT(int iteration);

	void TestDistanceToOutUSolids(int iteration);
	void TestDistanceToOutGeant4(int iteration);
	void TestDistanceToOutROOT(int iteration);

	void CreatePointsAndDirections();
	void CreatePointsAndDirectionsSurface();
	void CreatePointsAndDirectionsInside();
	void CreatePointsAndDirectionsOutside();

	void CompareResults(double resG, double resR, double resU);
	void CompareAndSaveResults(const std::string &method, double resG, double resR, double resU);

	int SaveResultsToFile(const std::string &method);

	void SavePolyhedra(const std::string &method);

	void CompareInside();
	void CompareNormal();
	void CompareSafetyFromInside();
	void CompareSafetyFromOutside();
	void CompareDistanceToIn();
	void CompareDistanceToOut();

	double MeasureTest (void (SBTperformance::*funcPtr)(int), const std::string &method);

	double NormalizeToNanoseconds(double time);

	void TestMethod(void (SBTperformance::*funcPtr)());
	void TestMethodAll();

	double ConvertInfinities(double value);

	void CheckPointsOnSurfaceOfOrb(const G4ThreeVector &point, double res, int count, EInside location);

  inline double RandomRange(double min, double max)
  {
    double rand = min + (max - min) * G4UniformRand();
    return rand;
  }

  inline void GetVectorGeant4(G4ThreeVector &point, const std::vector<UVector3> &points, int index)
  {
    const UVector3 &p = points[index];
    point.set(p.x, p.y, p.z);
  }

  inline void GetVectorUSolids(UVector3 &point, const std::vector<UVector3> &points, int index)
  {
    const UVector3 &p = points[index];
    point.Set(p.x, p.y, p.z);
  }

  inline void GetVectorRoot(double *point, const std::vector<UVector3> &points, int index)
  {
    const UVector3 &p = points[index];
    point[0] = p.x; point[1] = p.y; point[2] = p.z;
  }

  inline void SetVectorGeant4(const G4ThreeVector &point, std::vector<UVector3> &points, int index)
  {
    UVector3 &p = points[index];
    p.Set(point.getX(), point.getY(), point.getZ());
  }

  inline void SetVectorUSolids(const UVector3 &point, std::vector<UVector3> &points, int index)
  {
    UVector3 &p = points[index];
    p.Set(point.x, point.y, point.z);
  }

  inline void SetVectorRoot(const double *point, std::vector<UVector3> &points, int index)
  {
    UVector3 &p = points[index];
    p.Set (point[0], point[1], point[2]);
  }

  inline double RandomIncrease()
  {
    double tolerance = VUSolid::Tolerance();
    double rand = -1 + 2 * G4UniformRand();
    double sign = rand > 0 ? 1 : -1;
    double dif = tolerance * 0.1 * rand; // 19000000000
    //	if (abs(dif) < 9 * tolerance) dif = dif;
    return dif;
  }
};

#endif
