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

class SBTVisManager;

class SBTperformance {

public:
	SBTperformance();
	~SBTperformance();
	void SetDefaults();

	void Run(G4VSolid *testVolume, std::ofstream &logger);

	G4int DrawError( const G4VSolid *testVolume, std::istream &logger, const G4int errorIndex,
		SBTVisManager *visManager ) const;

	inline void SetMaxPoints( const G4int newMaxPoints ) { maxPoints = newMaxPoints; }
	inline void SetRepeat( const G4int newRepeat ) { repeat = newRepeat; }
	inline void SetMethod( const G4String newMethod ) { method = newMethod; }
	inline void SetInsidePercent( const G4double percent ) { insidePercent = percent; }
	inline void SetOutsidePercent( const G4double percent ) { outsidePercent = percent; }

	inline void SetOutsideMaxRadiusMultiple( const G4double percent ) { outsideMaxRadiusMultiple = percent; }
	inline void SetOutsideRandomDirectionPercent( const G4double percent ) { outsideRandomDirectionPercent = percent; }
	inline void SetDifferenceTolerance( const G4double tolerance ) { differenceTolerance = tolerance; }
	void SetFolder( std::string newFolder );

	inline G4int GetMaxPoints() const { return maxPoints; }
	inline G4int GetRepeat() const { return repeat; }

	int SaveVectorToMatlabFile(std::vector<double> &vector, std::string filename);
	int SaveVectorToMatlabFile(std::vector<UVector3> &vector, std::string filename);
	int SaveLegend(std::string filename);
	int SaveDoubleResults(std::string filename);
	int SaveVectorResults(std::string filename);

	std::string printCoordinates (UVector3 &vec, std::string &delimiter, int precision=4);
	std::string printCoordinates (UVector3 &vec, const char *delimiter, int precision=4);
	void printCoordinates (std::stringstream &ss, UVector3 &vec, std::string &delimiter, int precision=4);
	void printCoordinates (std::stringstream &ss, UVector3 &vec, const char *delimiter, int precision=4);

	template <class T> 

	void VectorDifference(std::vector<T> &first, std::vector<T> &second, std::vector<T> &result);
	
	void VectorToDouble(std::vector<UVector3> &vectorUVector, std::vector<double> &vectorDouble);
	
	int CountDoubleDifferences(std::vector<double> &differences);
	int CountDoubleDifferences(std::vector<double> &differences, std::vector<double> &values1, std::vector<double> &values2);

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

	void FlushSS(std::stringstream &ss);
	void Flush(std::string s);

	G4VSolid *volumeGeant4;
	VUSolid *volumeUSolids;
	TGeoShape *volumeROOT;
	std::string volumeString;

	void setupSolids(G4VSolid *testVolume);

	std::vector<UVector3> points, directions;
	std::vector<UVector3> resultVectorGeant4;
	std::vector<UVector3> resultVectorRoot;
	std::vector<UVector3> resultVectorUSolids, resultVectorDifference;
	std::vector<double> resultDoubleGeant4, resultDoubleRoot, resultDoubleUSolids, resultDoubleDifference;

	int offsetSurface, offsetInside, offsetOutside;
	int maxPointsInside, maxPointsOutside, maxPointsSurface;
	std::ostream *log, *perftab, *perflabels;
	std::string folder;

	UVector3 GetVectorOnOrb(G4Orb& orb, UVector3& norm);
	UVector3 GetRandomDirection();

	inline void GetVectorGeant4(G4ThreeVector &point, std::vector<UVector3> &points, int index);
	inline void GetVectorUSolids(UVector3 &point, std::vector<UVector3> &points, int index);
	inline void GetVectorRoot(double *point, std::vector<UVector3> &points, int index);

	inline void SetVectorGeant4(G4ThreeVector &point, std::vector<UVector3> &points, int index);
	inline void SetVectorUSolids(UVector3 &point, std::vector<UVector3> &points, int index);
	inline void SetVectorRoot(double *point, std::vector<UVector3> &points, int index);

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

	void PropagatedNormal(G4ThreeVector &point, G4ThreeVector &direction, double distance, G4ThreeVector &normal);

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
	void CompareAndSaveResults(std::string method, double resG, double resR, double resU);

	int SaveResultsToFile(std::string method);

	void SavePolyhedra(std::string method);

	void CompareInside();
	void CompareNormal();
	void CompareSafetyFromInside();
	void CompareSafetyFromOutside();
	void CompareDistanceToIn();
	void CompareDistanceToOut();

	double MeasureTest (void (SBTperformance::*funcPtr)(int), std::string method);

	double normalizeToNanoseconds(double time);

	void TestMethod(void (SBTperformance::*funcPtr)());
	void TestMethodAll();

	double ConvertInfinities(double value);

	void CheckPointsOnSurfaceOfOrb(G4ThreeVector &point, double res, int count, EInside location);
};

#endif
