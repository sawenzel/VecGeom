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
// SBTperformanceMessenger.cc
//
// Implementation of the messenger for controlling test 3
//

#include "SBTperformanceMessenger.hh"
#include "SBTperformance.hh"
#include "SBTVisManager.hh"
#include "G4SolidQuery.hh"

#include "G4ios.hh"
#include "G4UIdirectory.hh"
#include "G4UIcommand.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "SBTrun.hh"
 
#include <fstream>

//
// Constructor
//
SBTperformanceMessenger::SBTperformanceMessenger( const G4String prefix, const G4SolidQuery *theSolidQuery, SBTVisManager *theVisManager )
{
	//
	// Store solid query
	//
	solidQuery = theSolidQuery;
	
	//
	// Store visualization manager
	//
	visManager = theVisManager;

	//
	// Defaults (of locally stored values)
	//
	errorFile = "sbt.log";
	
	//
	// Create tester
	//
	tester = new SBTperformance();
	
	//
	// Declare directory
	//
	test3Directory = new G4UIdirectory( prefix );
	test3Directory->SetGuidance( "Controls for CSG batch test" );
	
	//
	// Target command
	//
	G4String com = prefix+"target";
	targetCmd = new G4UIcmdWith3VectorAndUnit(com, this );
	targetCmd->SetGuidance( "Center of distribution of random points" );
	targetCmd->SetParameterName( "X", "Y", "Z", true, true );
	
	//
	// Widths command
	//
	com = prefix+"widths";
	widthsCmd = new G4UIcmdWith3VectorAndUnit(com, this );
	widthsCmd->SetGuidance( "Widths of distribution of random points" );
	widthsCmd->SetParameterName( "Dx", "Dy", "Dz", true, true );
	
	//
	// Grid Size command
	//
	com = prefix+"gridSizes";
	gridSizesCmd = new G4UIcmdWith3VectorAndUnit(com, this );
	gridSizesCmd->SetGuidance( "Grid size, or zero for no grid" );
	gridSizesCmd->SetParameterName( "Dx", "Dy", "Dz", true, true );
	
	//
	// Max Points command
	//
	com = prefix+"maxPoints";
	maxPointsCmd = new G4UIcmdWithAnInteger(com, this );
	maxPointsCmd->SetGuidance( "Maximum number of points before test ends" );

	//
	// Repeat command
	//
	com = prefix+"repeat";
	repeatCmd = new G4UIcmdWithAnInteger(com, this );
	repeatCmd->SetGuidance( "Repeat test loops of methods" );

	//
	// Max Inside Percent command
	//
	com = prefix+"maxInsidePercent";
	insidePercentCmd = new G4UIcmdWithADouble (com, this );
	insidePercentCmd->SetGuidance( "Percent of inside points" );

	//
	// Max Inside Percent command
	//
	com = prefix+"maxOutsidePercent";
	outsidePercentCmd = new G4UIcmdWithADouble (com, this );
	outsidePercentCmd->SetGuidance( "Percent of outside points" );

	//
	// Max Random Direction Percent command
	//
	com = prefix+"outsideRandomDirectionPercent";
	outsideRandomDirectionPercentCmd = new G4UIcmdWithADouble (com, this );
	outsideRandomDirectionPercentCmd->SetGuidance("Percent of outside points with random direction" );

	//
	// Max Outside Max Radius Multiple command
	//
	com = prefix+"outsideMaxRadiusMultiple";
	outsideMaxRadiusMultipleCmd = new G4UIcmdWithADouble (com, this );
	outsideMaxRadiusMultipleCmd->SetGuidance("Maximum radius multiple for outside pount");

	//
	// Max Difference Tolerance
	//
	com = prefix+"differenceTolerance";
	differenceToleranceCmd = new G4UIcmdWithADouble (com, this );
	differenceToleranceCmd->SetGuidance("Difference Tolerance");

	//
	// Max Errors command
	//	
	com = prefix+"maxErrors";
	maxErrorsCmd = new G4UIcmdWithAnInteger(com, this );
	maxErrorsCmd->SetGuidance( "Maximum number of errors before test ends" );
	
	//
	// Error filename command
	//
	com = prefix+"errorFileName";
	errorFileCmd = new G4UIcmdWithAString(com, this );
	errorFileCmd->SetGuidance( "Filename in which to send error listings" );
	

	com = prefix+"method";
	methodCmd = new G4UIcmdWithAString(com, this );
	methodCmd->SetGuidance( "What method to measure" );

	//
	// Run command
	//
	com = prefix+"run";
	runCmd = new G4UIcmdWithoutParameter(com, this );
	runCmd->SetGuidance( "Execute a test" );
	
	//
	// Debug commands
	//
	com = prefix+"draw";
	drawCmd = new G4UIcmdWithAnInteger(com, this );
	drawCmd->SetGuidance( "Draw error listed in log file" );
	
	com = prefix+"debugInside";
	debugInsideCmd = new G4UIcmdWithAnInteger(com, this );
	debugInsideCmd->SetGuidance( "Call G4VSolid::Inside for error listed in log file" );
	
	com = prefix+"debugToInP";
	debugToInPCmd = new G4UIcmdWithAnInteger(com, this );
	debugToInPCmd->SetGuidance( "Call G4VSolid::DistanceToIn(p) for error listed in log file" );
	
	com = prefix+"debugToInPV";
	debugToInPVCmd = new G4UIcmdWithAnInteger(com, this );
	debugToInPVCmd->SetGuidance( "Call G4VSolid::DistanceToIn(p,v) for error listed in log file" );
	
	com = prefix+"debugToOutP";
	debugToOutPCmd = new G4UIcmdWithAnInteger(com, this );
	debugToOutPCmd->SetGuidance( "Call G4VSolid::DistanceToOut(p) for error listed in log file" );
	
	com = prefix+"debugToOutPV";
	debugToOutPVCmd = new G4UIcmdWithAnInteger(com, this );
	debugToOutPVCmd->SetGuidance( "Call G4VSolid::DistanceToOut(p,v) for error listed in log file" );

	com = prefix+"debugSurfNorm";
	debugSurfNormCmd = new G4UIcmdWithAnInteger(com, this );
	debugSurfNormCmd->SetGuidance( "Call G4VSolid::SurfaceNormal(p) for error listed in log file" );

	//
	// Pause command
	//
	com = prefix+"pause";
	pauseCmd = new G4UIcmdWithoutParameter(com, this );
	pauseCmd->SetGuidance( " Wait for a key " );
	
}


//
// Destructor
//
SBTperformanceMessenger::~SBTperformanceMessenger()
{
	delete targetCmd;
	delete widthsCmd;
	delete gridSizesCmd;
	delete maxPointsCmd;
	delete repeatCmd;
	delete maxErrorsCmd;
	delete errorFileCmd;
	delete methodCmd;
	delete test3Directory;
}


//
// InvokeTest3
//
// Run test 3
//

inline void getSolid()
{

}

void SBTperformanceMessenger::InvokePerformanceTest()
{
	//
	// Is there a Solid to test?
	//
	G4VSolid *testSolid = solidQuery->GetSolid();
	if (testSolid == 0) {
		G4cerr << "Please initialize geometry before running test 3" << G4endl;
		G4cerr << "Test 3 ABORTED" << G4endl;
		return;
	}

	tester->Run(testSolid, logger);
}

inline std::string dirname(std::string source)
{
    source.erase(std::find(source.rbegin(), source.rend(), '/').base(), source.end());
	source.erase(source.find_last_not_of("/")+1); // trim last /
    return source;
}

//
// SetNewValue
//
// Call by the UI when user requests a change
//
void SBTperformanceMessenger::SetNewValue( G4UIcommand *command, G4String newValues )
{
	if (command == maxPointsCmd) {
		tester->SetMaxPoints( maxPointsCmd->GetNewIntValue( newValues ) );
	}
	else if (command == repeatCmd) {
		tester->SetRepeat( repeatCmd->GetNewIntValue( newValues ) );
	}
	else if (command == insidePercentCmd) {
		tester->SetInsidePercent( insidePercentCmd->GetNewDoubleValue( newValues ) );
	}
	else if (command == outsidePercentCmd) {
		tester->SetOutsidePercent( outsidePercentCmd->GetNewDoubleValue( newValues ) );
	}
	else if (command == outsideMaxRadiusMultipleCmd) {
		tester->SetOutsideMaxRadiusMultiple( insidePercentCmd->GetNewDoubleValue( newValues ) );
	} 
	else if (command == outsidePercentCmd) {
		tester->SetOutsideRandomDirectionPercent( outsideRandomDirectionPercentCmd->GetNewDoubleValue( newValues ) );
	}
	else if (command == differenceToleranceCmd) {
		tester->SetDifferenceTolerance( differenceToleranceCmd->GetNewDoubleValue( newValues ) );
	}

	else if (command == errorFileCmd) {
		if (errorFile != newValues) 
		{
			logger.close();
			errorFile = newValues;
			tester->SetFolder(dirname (errorFile));
			logger.open(errorFile);
		}
	}
	else if (command == methodCmd) {
		// NEW: set method macro command can be used to specify which type of test we will use
		tester->SetMethod(newValues);
	}
	else if (command == runCmd) {
		InvokePerformanceTest();
		logger.flush();
	}
	else if (command == pauseCmd) {
	  char c;
	  
	  G4cout << "Press ENTER to continue..." << std::flush ;
	  G4cin.get(c);
	}
	else {
		G4Exception("SBTperformanceMessenger::SetNewValue", "", FatalException, "Unrecognized command");
	}
}


//
// GetCurrentValue
//
G4String SBTperformanceMessenger::GetCurrentValue( G4UIcommand *command )
{
	if (command == maxPointsCmd) {
		return maxPointsCmd->ConvertToString( tester->GetMaxPoints() );
	}
	if (command == repeatCmd) {
		return repeatCmd->ConvertToString( tester->GetRepeat() );
	}
	else if (command == runCmd) {
		return "";
	}
	
    G4Exception("SBTperformanceMessenger::GetCurrentValue", "", FatalException, "Unrecognized command");

	return "foo!";
}
	
