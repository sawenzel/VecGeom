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
// SBTperformanceMessenger.hh
//
// Definition of the messenger for controlling test 3
//
#ifndef SBTperformanceMessenger_hh
#define SBTperformanceMessenger_hh

#include "G4UImessenger.hh"
#include "SBTperformance.hh"
#include "globals.hh"
#include <fstream>

class SBTperformance;
class SBTVisManager;

class G4VSolid;
class G4SolidQuery;

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithAString;
class G4UIcmdWithoutParameter;

class SBTperformanceMessenger : public G4UImessenger
{
public:
	SBTperformanceMessenger( const G4String prefix, const G4SolidQuery *solidQuery, SBTVisManager *visManager );
	~SBTperformanceMessenger();

	void SetNewValue( G4UIcommand *command, G4String newValues );
	G4String GetCurrentValue( G4UIcommand *command );

	inline const G4SolidQuery *GetSolidQuery() const { return solidQuery; }

	inline SBTVisManager *GetVisManager() const { return visManager; }
	inline void SetVisManager( SBTVisManager *theVisManager ) { visManager = theVisManager; }

public:
	class Debugger {
	public:
		Debugger( const G4VSolid *aSolid, const SBTperformance *aTester ) 
		{ testSolid = aSolid; tester = aTester; }
		virtual ~Debugger() {;}

		virtual G4int DebugMe( std::ifstream &logFile, const G4int errorIndex ) = 0;

	protected:
		const G4VSolid *testSolid;
		const SBTperformance *tester;
	};

protected:
	class DrawError : public SBTperformanceMessenger::Debugger {
	public:
		DrawError( const G4VSolid *aSolid, const SBTperformance *aTester, SBTVisManager *theVisManager )
			: Debugger(aSolid,aTester) { visManager = theVisManager;}
		G4int DebugMe( std::ifstream &logFile, const G4int errorIndex );

	protected:
		SBTVisManager *visManager;
	};

	void InvokePerformanceTest();

private:
	SBTperformance		*tester;
	const G4SolidQuery *solidQuery;
	SBTVisManager	*visManager;

	G4String	errorFile, file;
	std::ofstream logger;

	G4UIdirectory			*test3Directory;
	G4UIcmdWith3VectorAndUnit	*targetCmd;
	G4UIcmdWith3VectorAndUnit	*widthsCmd;
	G4UIcmdWith3VectorAndUnit	*gridSizesCmd;
	G4UIcmdWithAnInteger		*maxPointsCmd, *repeatCmd;
	G4UIcmdWithADouble		*insidePercentCmd, *outsidePercentCmd, *outsideMaxRadiusMultipleCmd, *outsideRandomDirectionPercentCmd, *differenceToleranceCmd;

	G4UIcmdWithAnInteger		*maxErrorsCmd;
	G4UIcmdWithAString		*errorFileCmd, *methodCmd;
	G4UIcmdWithoutParameter		*runCmd;
	G4UIcmdWithAnInteger		*drawCmd;
	G4UIcmdWithAnInteger		*debugInsideCmd;
	G4UIcmdWithAnInteger		*debugToInPCmd;
	G4UIcmdWithAnInteger		*debugToInPVCmd;
	G4UIcmdWithAnInteger		*debugToOutPCmd;
	G4UIcmdWithAnInteger		*debugToOutPVCmd;
	G4UIcmdWithAnInteger		*debugSurfNormCmd;
	G4UIcmdWithoutParameter		*pauseCmd;
};

#endif
