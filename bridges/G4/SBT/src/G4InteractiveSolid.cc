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
// G4InteractiveSolid.cc
//
// Implementation of a messenger for constructing solids interactively
//

#include "SBTrun.hh"

#include "G4GDMLParser.hh"

#include "G4InteractiveSolid.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithPargs.hh"
#include "G4UIcmdPargDouble.hh"
#include "G4UIcmdPargInteger.hh"
#include "G4UIcmdPargListDouble.hh"
#include "G4UIcmdWithAString.hh"

#include "UBox.hh"
#include "UMultiUnion.hh"
#include "UOrb.hh"
#include "UTrd.hh"
#include "G4USolid.hh"
#include "VUSolid.hh"
#include "UVector3.hh"
#include "UTypes.hh"
#include "UUtils.hh"

#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Orb.hh"
#include "G4Para.hh"
#include "G4Sphere.hh"
#include "G4Torus.hh"
#include "G4Trap.hh"
#include "G4Trd.hh"
#include "G4Tubs.hh"
#include "G4Ellipsoid.hh"
#include "G4EllipticalCone.hh"
#include "G4EllipticalTube.hh"
#include "G4ExtrudedSolid.hh"
#include "G4Hype.hh"
#include "G4Polycone.hh"
#include "G4Polyhedra.hh"
#include "G4TriangularFacet.hh"
#include "G4QuadrangularFacet.hh"
#include "G4TessellatedSolid.hh"
#include "G4Tet.hh"
#include "G4TwistedBox.hh"
#include "G4TwistedTrap.hh"
#include "G4TwistedTrd.hh"
#include "G4TwistedTubs.hh"
#include "G4PhysicalConstants.hh"

#include "G4IntersectionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"

#include "G4BREPSolidBox.hh"
#include "G4Point3D.hh"

#include "G4Tools.hh"

#include "USTL.hh"

using namespace std;
using namespace CLHEP;

//
// Constructor
// 
G4InteractiveSolid::G4InteractiveSolid( const G4String &prefix )
{
	//
	// Hmmm... well, how about a reasonable default?
	//
	solid = new G4Box( "interactiveBox", 1.0*m, 1.0*m, 1.0*m );

	//
	// Declare messenger directory
	//
	volumeDirectory = new G4UIdirectory( prefix );
	volumeDirectory->SetGuidance( "Solid construction using parameters from the command line" );

	box.Args.push_back(new G4UIcmdPargDouble( "dx", 1.0, m));
	box.Args.push_back(new G4UIcmdPargDouble( "dy", 1.0, m ));
	box.Args.push_back(new G4UIcmdPargDouble( "dz", 1.0, m ));
	box.Cmd = new G4UIcmdWithPargs(G4String(prefix+"G4Box"), this, &box.Args[0], box.Args.size());
	box.Cmd->SetGuidance( "Declare a G4Box solid" );
	box.Make = &G4InteractiveSolid::MakeBox;
	commands.push_back(&box);

	//
	// filename command
	//
	G4String com = prefix+"file";
	fileCmd = new G4UIcmdWithAString( com, this );
	fileCmd->SetGuidance( "Filename which is used for solids which needs it (such as tessellated solid)" );

	//
	// Declare UBox
	//
	ubox.Args.push_back(new G4UIcmdPargDouble( "dx", 1.0, m ));
	ubox.Args.push_back(new G4UIcmdPargDouble( "dy", 1.0, m ));
	ubox.Args.push_back(new G4UIcmdPargDouble( "dz", 1.0, m ));
	ubox.Cmd = new G4UIcmdWithPargs(G4String(prefix+"UBox"), this, &ubox.Args[0], ubox.Args.size());
	ubox.Cmd->SetGuidance( "Declare a U4Box solid" );
	ubox.Make = &G4InteractiveSolid::MakeUBox;
	commands.push_back(&ubox);

   	//
	// Declare UMultiUnion
	//
	umultiunion.Args.push_back(new G4UIcmdPargDouble( "dx", 1.0, m ));
	umultiunion.Args.push_back(new G4UIcmdPargDouble( "dy", 1.0, m ));
	umultiunion.Args.push_back(new G4UIcmdPargDouble( "dz", 1.0, m ));
	umultiunion.Cmd = new G4UIcmdWithPargs(G4String (prefix+"UMultiUnion"), this, &umultiunion.Args[0],		umultiunion.Args.size());
	umultiunion.Cmd->SetGuidance( "Declare a UMultiUnion solid" );
	umultiunion.Make = &G4InteractiveSolid::MakeUMultiUnion;
	commands.push_back(&umultiunion);

	// Declare UOrb
	//
	uorb.Args.push_back(new G4UIcmdPargDouble( "r", 1.0, m ));
	G4String uorbPath = prefix+"UOrb";
	uorb.Cmd = new G4UIcmdWithPargs( uorbPath, this, &uorb.Args[0], 1 );
	uorb.Cmd->SetGuidance( "Declare a UOrb solid" );
	uorb.Make = &G4InteractiveSolid::MakeUOrb;
	commands.push_back(&uorb);

	// Declare UTrd
	//
	utrd.Args.push_back(new G4UIcmdPargDouble( "dx1", 30.0, m ));
	utrd.Args.push_back(new G4UIcmdPargDouble( "dx2", 10.0, m ));
	utrd.Args.push_back(new G4UIcmdPargDouble( "dy1", 40.0, m ));
	utrd.Args.push_back(new G4UIcmdPargDouble( "dy2", 15.0, m ));
	utrd.Args.push_back(new G4UIcmdPargDouble( "dz", 60.0, m ));
	G4String utrdPath = prefix+"UTrd";
	utrd.Cmd = new G4UIcmdWithPargs( utrdPath, this, &utrd.Args[0], utrd.Args.size() );
	utrd.Cmd->SetGuidance( "Declare a UTrd solid" );
	utrd.Make = &G4InteractiveSolid::MakeUTrd;
	commands.push_back(&utrd);

	//
	// Declare G4Cons
	//
	cons.Args.push_back(new G4UIcmdPargDouble( "rmin1", 1.0, m ));
	cons.Args.push_back(new G4UIcmdPargDouble( "rmax1", 1.0, m ));
	cons.Args.push_back(new G4UIcmdPargDouble( "rmin2", 1.0, m ));
	cons.Args.push_back(new G4UIcmdPargDouble( "rmax2", 1.0, m ));
	cons.Args.push_back(new G4UIcmdPargDouble( "dz",    1.0, m ));
	cons.Args.push_back(new G4UIcmdPargDouble( "startPhi",  1.0, deg ));
	cons.Args.push_back(new G4UIcmdPargDouble( "deltaPhi",  1.0, deg ));
	G4String consPath = prefix+"G4Cons";
	cons.Cmd = new G4UIcmdWithPargs( consPath, this, &cons.Args[0], cons.Args.size() );
	cons.Cmd->SetGuidance( "Declare a G4Cons solid" );
	cons.Make = &G4InteractiveSolid::MakeCons;
	commands.push_back(&cons);

	//
	// Declare G4Orb
	//
	orb.Args.push_back(new G4UIcmdPargDouble( "r", 1.0, m ));
	G4String orbPath = prefix+"G4Orb";
	orb.Cmd = new G4UIcmdWithPargs( orbPath, this, &orb.Args[0], orb.Args.size() );
	orb.Cmd->SetGuidance( "Declare a G4Orb solid" );
	orb.Make = &G4InteractiveSolid::MakeOrb;
	commands.push_back(&orb);

	//
	// Declare G4Para
	//
	para.Args.push_back(new G4UIcmdPargDouble( "dx", 1.0, m ));
	para.Args.push_back(new G4UIcmdPargDouble( "dy", 1.0, m ));
	para.Args.push_back(new G4UIcmdPargDouble( "dz", 1.0, m ));
	para.Args.push_back(new G4UIcmdPargDouble( "alpha", 90, deg ));
	para.Args.push_back(new G4UIcmdPargDouble( "theta", 90, deg ));
	para.Args.push_back(new G4UIcmdPargDouble( "phi", 90, deg ));
	G4String paraPath = prefix+"G4Para";
	para.Cmd = new G4UIcmdWithPargs( paraPath, this, &para.Args[0], para.Args.size() );
	para.Cmd->SetGuidance( "Declare a G4Para solid" );
	para.Make = &G4InteractiveSolid::MakePara;
	commands.push_back(&para);

	//
	// Declare G4Sphere
	//
	sphere.Args.push_back(new G4UIcmdPargDouble( "rmin", 1.0, m ));
	sphere.Args.push_back(new G4UIcmdPargDouble( "rmax", 1.0, m ));
	sphere.Args.push_back(new G4UIcmdPargDouble( "startPhi", 1.0, deg ));
	sphere.Args.push_back(new G4UIcmdPargDouble( "deltaPhi", 1.0, deg ));
	sphere.Args.push_back(new G4UIcmdPargDouble( "startTheta",  1.0, deg ));
	sphere.Args.push_back(new G4UIcmdPargDouble( "deltaTheta",  1.0, deg ));
	G4String spherePath = prefix+"G4Sphere";
	sphere.Cmd = new G4UIcmdWithPargs( spherePath, this, &sphere.Args[0], sphere.Args.size() );
	sphere.Cmd->SetGuidance( "Declare a G4Sphere solid" );
	sphere.Make = &G4InteractiveSolid::MakeSphere;
	commands.push_back(&sphere);

	//
	// Declare G4Torus
	//
	torus.Args.push_back(new G4UIcmdPargDouble( "rmin", 1.0, m ));
	torus.Args.push_back(new G4UIcmdPargDouble( "rmax", 1.0, m ));
	torus.Args.push_back(new G4UIcmdPargDouble( "rtorus", 1.0, m ));
	torus.Args.push_back(new G4UIcmdPargDouble( "startPhi",  1.0, deg ));
	torus.Args.push_back(new G4UIcmdPargDouble( "deltaPhi",  1.0, deg ));
	G4String torusPath = prefix+"G4Torus";
	torus.Cmd = new G4UIcmdWithPargs( torusPath, this, &torus.Args[0], torus.Args.size() );
	torus.Cmd->SetGuidance( "Declare a G4Torus solid" );
	torus.Make = &G4InteractiveSolid::MakeTorus;
	commands.push_back(&torus);

	//
	// Declare G4Trap
	//
	trap.Args.push_back(new G4UIcmdPargDouble( "dz",    1.0, m ));
	trap.Args.push_back(new G4UIcmdPargDouble( "theta",  90, deg ));
	trap.Args.push_back(new G4UIcmdPargDouble( "phi",   90, deg ));
	trap.Args.push_back(new G4UIcmdPargDouble( "dy1",   1.0, m ));
	trap.Args.push_back(new G4UIcmdPargDouble( "dx1",   1.0, m ));
	trap.Args.push_back(new G4UIcmdPargDouble( "dx2",   1.0, m ));
	trap.Args.push_back(new G4UIcmdPargDouble( "alpha1", 90, deg ));
	trap.Args.push_back(new G4UIcmdPargDouble( "dy2",   1.0, m ));
	trap.Args.push_back(new G4UIcmdPargDouble( "dx3",   1.0, m ));
	trap.Args.push_back(new G4UIcmdPargDouble( "dx4",   1.0, m ));
	trap.Args.push_back(new G4UIcmdPargDouble( "alpha2", 90, deg ));
	G4String trapPath = prefix+"G4Trap";
	trap.Cmd = new G4UIcmdWithPargs( trapPath, this, &trap.Args[0], trap.Args.size() );
	trap.Cmd->SetGuidance( "Declare a G4Trap solid" );
	trap.Make = &G4InteractiveSolid::MakeTrap;
	commands.push_back(&trap);

	//
	// Declare G4Trd
	//
	trd.Args.push_back(new G4UIcmdPargDouble( "dx1", 1.0, m ));
	trd.Args.push_back(new G4UIcmdPargDouble( "dx2", 1.0, m ));
	trd.Args.push_back(new G4UIcmdPargDouble( "dy1", 1.0, m ));
	trd.Args.push_back(new G4UIcmdPargDouble( "dy2", 1.0, m ));
	trd.Args.push_back(new G4UIcmdPargDouble( "dz",  1.0, m ));
	G4String trdPath = prefix+"G4Trd";
	trd.Cmd = new G4UIcmdWithPargs( trdPath, this, &trd.Args[0], trd.Args.size() );
	trd.Cmd->SetGuidance( "Declare a G4Trd solid" );
	trd.Make = &G4InteractiveSolid::MakeTrd;
	commands.push_back(&trd);

	//
	// Declare G4Tubs
	//
	tubs.Args.push_back(new G4UIcmdPargDouble( "rmin", 1.0, m ));
	tubs.Args.push_back(new G4UIcmdPargDouble( "rmax", 1.0, m ));
	tubs.Args.push_back(new G4UIcmdPargDouble( "dz",   1.0, m ));
	tubs.Args.push_back(new G4UIcmdPargDouble( "startPhi",  1.0, deg ));
	tubs.Args.push_back(new G4UIcmdPargDouble( "deltaPhi",  1.0, deg ));
	G4String tubsPath = prefix+"G4Tubs";
	tubs.Cmd = new G4UIcmdWithPargs( tubsPath, this, &tubs.Args[0], tubs.Args.size() );
	tubs.Cmd->SetGuidance( "Declare a G4Tubs solid" );
	tubs.Make = &G4InteractiveSolid::MakeTubs;
	commands.push_back(&tubs);

	//
	// Declare G4Ellipsoid
	//
	ellipsoid.Args.push_back(new G4UIcmdPargDouble( "dx", 1.0, m ));
	ellipsoid.Args.push_back(new G4UIcmdPargDouble( "dy", 1.0, m ));
	ellipsoid.Args.push_back(new G4UIcmdPargDouble( "dz", 1.0, m ));
	ellipsoid.Args.push_back(new G4UIcmdPargDouble( "zBottomCut", 0, m ));
	ellipsoid.Args.push_back(new G4UIcmdPargDouble( "zTopCut", 0, m ));
	G4String ellipsoidPath = prefix+"G4Ellipsoid";
	ellipsoid.Cmd = new G4UIcmdWithPargs( ellipsoidPath, this, &ellipsoid.Args[0], ellipsoid.Args.size() );
	ellipsoid.Cmd->SetGuidance( "Declare a G4Ellipsoid solid" );
	ellipsoid.Make = &G4InteractiveSolid::MakeEllipsoid;
	commands.push_back(&ellipsoid);

	//
	// Declare G4EllipticalCone
	//
	elCone.Args.push_back(new G4UIcmdPargDouble( "dx", 1.0, 1 ));
	elCone.Args.push_back(new G4UIcmdPargDouble( "dy", 1.0, 1 ));
	elCone.Args.push_back(new G4UIcmdPargDouble( "dz", 1.0, m ));
	elCone.Args.push_back(new G4UIcmdPargDouble( "zTopCut", 1.0, m ));
	G4String elConePath = prefix+"G4EllipticalCone";
	elCone.Cmd = new G4UIcmdWithPargs( elConePath, this, &elCone.Args[0], elCone.Args.size() );
	elCone.Cmd->SetGuidance( "Declare a G4EllipticalCone solid" );
	elCone.Make = &G4InteractiveSolid::MakeEllipticalCone;
	commands.push_back(&elCone);

	//
	// Declare G4EllipticalTube
	//
	elTube.Args.push_back(new G4UIcmdPargDouble( "dx", 1.0, m ));
	elTube.Args.push_back(new G4UIcmdPargDouble( "dy", 1.0, m ));
	elTube.Args.push_back(new G4UIcmdPargDouble( "dz", 1.0, m ));
	G4String elTubePath = prefix+"G4EllipticalTube";
	elTube.Cmd = new G4UIcmdWithPargs( elTubePath, this, &elTube.Args[0], elTube.Args.size() );
	elTube.Cmd->SetGuidance( "Declare a G4EllipticalTube solid" );
	elTube.Make = &G4InteractiveSolid::MakeEllipticalTube;
	commands.push_back(&elTube);

	//
	// Declare G4ExtrudedSolid
	//
	extruded.Args.push_back(new G4UIcmdPargInteger( "numPoints", 8 ));
	extruded.Args.push_back(new G4UIcmdPargListDouble( "pgonx", 100, m ));
	extruded.Args.push_back(new G4UIcmdPargListDouble( "pgony", 100, m ));
	extruded.Args.push_back(new G4UIcmdPargInteger( "numSides", 8 ));
	extruded.Args.push_back(new G4UIcmdPargListDouble(     "z", 100, m ));
	extruded.Args.push_back(new G4UIcmdPargListDouble(  "offx", 100, m ));
	extruded.Args.push_back(new G4UIcmdPargListDouble(  "offy", 100, m ));
	extruded.Args.push_back(new G4UIcmdPargListDouble( "scale", 100, 1 ));
	G4String extrudedPath = prefix+"G4ExtrudedSolid";
	extruded.Cmd = new G4UIcmdWithPargs( extrudedPath, this, &extruded.Args[0], extruded.Args.size() );
	extruded.Cmd->SetGuidance( "Declare a G4ExtrudedSolid solid" );
	extruded.Make = &G4InteractiveSolid::MakeExtrudedSolid;
	commands.push_back(&extruded);

	//
	// Declare G4Hype
	//
	hype.Args.push_back(new G4UIcmdPargDouble( "innerRadius", 1.0, m ));
	hype.Args.push_back(new G4UIcmdPargDouble( "outerRadius", 1.0, m ));
	hype.Args.push_back(new G4UIcmdPargDouble( "innerStereo", 1.0, rad ));
	hype.Args.push_back(new G4UIcmdPargDouble( "outerStereo", 1.0, rad ));
	hype.Args.push_back(new G4UIcmdPargDouble( "dz",  1.0, m ));
	G4String hypePath = prefix+"G4Hype";
	hype.Cmd = new G4UIcmdWithPargs( hypePath, this, &hype.Args[0], hype.Args.size() );
	hype.Cmd->SetGuidance( "Declare a G4Hype solid" );
	hype.Make = &G4InteractiveSolid::MakeHype;
	commands.push_back(&hype);
	//
	// Declare G4Polycone
	//
	polycone.Args.push_back(new G4UIcmdPargDouble( "phiStart", 0, deg ));
	polycone.Args.push_back(new G4UIcmdPargDouble( "phiTotal", 0, deg ));
	polycone.Args.push_back(new G4UIcmdPargInteger( "numRZ", 1 ));
	polycone.Args.push_back(new G4UIcmdPargListDouble( "r", 100, m ));
	polycone.Args.push_back(new G4UIcmdPargListDouble( "z", 100, m ));
	G4String polyconePath = prefix+"G4Polycone";
	polycone.Cmd = new G4UIcmdWithPargs( polyconePath, this, &polycone.Args[0], polycone.Args.size() );
	polycone.Cmd->SetGuidance( "Declare a G4Polycone solid" );
	polycone.Make = &G4InteractiveSolid::MakePolycone;
	commands.push_back(&polycone);
	//
	// Declare G4Polycone2
	//
	polycone2.Args.push_back(new G4UIcmdPargDouble( "phiStart", 0, deg ));
	polycone2.Args.push_back(new G4UIcmdPargDouble( "phiTotal", 0, deg ));
	polycone2.Args.push_back(new G4UIcmdPargInteger( "numRZ", 1 ));
	polycone2.Args.push_back(new G4UIcmdPargListDouble( "z", 100, m ));
	polycone2.Args.push_back(new G4UIcmdPargListDouble( "rin", 100, m ));
	polycone2.Args.push_back(new G4UIcmdPargListDouble( "rout", 100, m ));
	G4String polycone2Path = prefix+"G4Polycone2";
	polycone2.Cmd = new G4UIcmdWithPargs( polycone2Path, this, &polycone2.Args[0], polycone2.Args.size() );
	polycone2.Cmd->SetGuidance( "Declare a G4Polycone solid (PCON style)" );
	polycone2.Make = &G4InteractiveSolid::MakePolycone2;
	commands.push_back(&polycone2);
	//
	// Declare G4Polyhedra
	//
	polyhedra.Args.push_back(new G4UIcmdPargDouble( "phiStart", 0, deg ));
	polyhedra.Args.push_back(new G4UIcmdPargDouble( "phiTotal", 0, deg ));
	polyhedra.Args.push_back(new G4UIcmdPargInteger( "numSides", 8 ));
	polyhedra.Args.push_back(new G4UIcmdPargInteger( "numRZ", 1 ));
	polyhedra.Args.push_back(new G4UIcmdPargListDouble( "r", 100, m ));
	polyhedra.Args.push_back(new G4UIcmdPargListDouble( "z", 100, m ));
	G4String polyhedraPath = prefix+"G4Polyhedra";
	polyhedra.Cmd = new G4UIcmdWithPargs( polyhedraPath, this, &polyhedra.Args[0], polyhedra.Args.size() );
	polyhedra.Cmd->SetGuidance( "Declare a G4Polyhedra solid" );
	polyhedra.Make = &G4InteractiveSolid::MakePolyhedra;
	commands.push_back(&polyhedra);
	//
	// Declare G4Polyhedra2
	//
	polyhedra2.Args.push_back(new G4UIcmdPargDouble( "phiStart", 0, deg ));
	polyhedra2.Args.push_back(new G4UIcmdPargDouble( "phiTotal", 0, deg ));
	polyhedra2.Args.push_back(new G4UIcmdPargInteger( "numSides", 8 ));
	polyhedra2.Args.push_back(new G4UIcmdPargInteger( "numRZ", 1 ));
	polyhedra2.Args.push_back(new G4UIcmdPargListDouble( "z", 100, m ));
	polyhedra2.Args.push_back(new G4UIcmdPargListDouble( "rin", 100, m ));
	polyhedra2.Args.push_back(new G4UIcmdPargListDouble( "rout", 100, m ));
	G4String polyhedra2Path = prefix+"G4Polyhedra2";
	polyhedra2.Make = &G4InteractiveSolid::MakePolyhedra2;
	polyhedra2.Cmd = new G4UIcmdWithPargs( polyhedra2Path, this, &polyhedra2.Args[0], polyhedra2.Args.size() );
	polyhedra2.Cmd->SetGuidance( "Declare a G4Polyhedra solid (PGON style)" );
	commands.push_back(&polyhedra2);
	//
	// Declare G4TessellatedSolid
	//
	tessel.Args.push_back(new G4UIcmdPargInteger("num3", 8 ));
	tessel.Args.push_back(new G4UIcmdPargListDouble("p1in3", 100, m ));
	tessel.Args.push_back(new G4UIcmdPargListDouble("p2in3", 100, m ));
	tessel.Args.push_back(new G4UIcmdPargListDouble("p3in3", 100, m ));
	tessel.Args.push_back(new G4UIcmdPargInteger("num4", 8 ));
	tessel.Args.push_back(new G4UIcmdPargListDouble("p1in4", 100, m ));
	tessel.Args.push_back(new G4UIcmdPargListDouble("p2in4", 100, m ));
	tessel.Args.push_back(new G4UIcmdPargListDouble("p3in4", 100, m ));
	tessel.Args.push_back(new G4UIcmdPargListDouble("p4in4", 100, m ));
	G4String tesselPath = prefix+"G4TessellatedSolid";
	tessel.Cmd = new G4UIcmdWithPargs( tesselPath, this, &tessel.Args[0], tessel.Args.size() );
	tessel.Cmd->SetGuidance( "Declare a G4TessellatedSolid solid" );
	tessel.Make = &G4InteractiveSolid::MakeTessellatedSolid;
	commands.push_back(&tessel);
	//
	// Declare G4TessellatedSolid2
	//
	tessel2.Args.push_back(new G4UIcmdPargInteger( "numPoints", 8 ));
	tessel2.Args.push_back(new G4UIcmdPargListDouble( "pgonx", 100, m ));
	tessel2.Args.push_back(new G4UIcmdPargListDouble( "pgony", 100, m ));
	tessel2.Args.push_back(new G4UIcmdPargInteger( "numSides", 8 ));
	tessel2.Args.push_back(new G4UIcmdPargListDouble(     "z", 100, m ));
	tessel2.Args.push_back(new G4UIcmdPargListDouble(  "offx", 100, m ));
	tessel2.Args.push_back(new G4UIcmdPargListDouble(  "offy", 100, m ));
	tessel2.Args.push_back(new G4UIcmdPargListDouble( "scale", 100, 1 ));
	G4String tessel2Path = prefix+"G4TessellatedSolid2";
	tessel2.Cmd = new G4UIcmdWithPargs( tessel2Path, this, &tessel2.Args[0], tessel2.Args.size() );
	tessel2.Cmd->SetGuidance( "Declare a G4TessellatedSolid solid as an extruded solid" );
	tessel2.Make = &G4InteractiveSolid::MakeTessellatedSolid2;
	commands.push_back(&tessel2);
	//
	// Declare G4Tet
	//
	tet.Args.push_back(new G4UIcmdPargListDouble( "p1", 3, m ));
	tet.Args.push_back(new G4UIcmdPargListDouble( "p2", 3, m ));
	tet.Args.push_back(new G4UIcmdPargListDouble( "p3", 3, m ));
	tet.Args.push_back(new G4UIcmdPargListDouble( "p4", 3, m ));
	G4String tetPath = prefix+"G4Tet";
	tet.Cmd = new G4UIcmdWithPargs( tetPath, this, &tet.Args[0], tet.Args.size() );
	tet.Cmd->SetGuidance( "Declare a G4Tet solid" );
	tet.Make = &G4InteractiveSolid::MakeTet;
	commands.push_back(&tet);
	//
	// Declare G4TwistedBox
	//
	twistedBox.Args.push_back(new G4UIcmdPargDouble( "phi", 0.0, deg ));
	twistedBox.Args.push_back(new G4UIcmdPargDouble( "dx",  1.0, m ));
	twistedBox.Args.push_back(new G4UIcmdPargDouble( "dy",  1.0, m ));
	twistedBox.Args.push_back(new G4UIcmdPargDouble( "dz",  1.0, m ));
	G4String twistedBoxPath = prefix+"G4TwistedBox";
	twistedBox.Cmd = new G4UIcmdWithPargs( twistedBoxPath, this, &twistedBox.Args[0], twistedBox.Args.size() );
	twistedBox.Cmd->SetGuidance( "Declare a G4TwistedBox solid" );
	twistedBox.Make = &G4InteractiveSolid::MakeTwistedBox;
	commands.push_back(&twistedBox);
	//
	// Declare regular G4TwistedTrap
        // 
	//
	twistedTrap.Args.push_back(new G4UIcmdPargDouble( "phi", 0.0, deg ));
	twistedTrap.Args.push_back(new G4UIcmdPargDouble( "dx1", 1.0, m ));
	twistedTrap.Args.push_back(new G4UIcmdPargDouble( "dx2", 1.0, m ));
	twistedTrap.Args.push_back(new G4UIcmdPargDouble( "dy",  1.0, m ));
	twistedTrap.Args.push_back(new G4UIcmdPargDouble( "dz",  1.0, m ));
	G4String twistedTrapPath = prefix+"G4TwistedTrap";
	twistedTrap.Cmd = new G4UIcmdWithPargs( twistedTrapPath, this, &twistedTrap.Args[0], twistedTrap.Args.size() );
	twistedTrap.Cmd->SetGuidance( "Declare a regular G4TwistedTrap solid" );
	twistedTrap.Make = &G4InteractiveSolid::MakeTwistedTrap;
    commands.push_back(&twistedTrap);

	//
	// Declare general G4TwistedTrap
	//
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "phi",   0.0, deg ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "dz",    1.0, m ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "theta", 0.0, deg ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "phi",   1.0, deg ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "dy1",   1.0, m ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "dx1",   1.0, m ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "dx2",   1.0, m ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "dy2",   1.0, m ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "dx3",   1.0, m ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "dx4",   1.0, m ));
	twistedTrap2.Args.push_back(new G4UIcmdPargDouble( "alpha", 0.0, deg ));
	G4String twistedTrap2Path = prefix+"G4TwistedTrap2";
	twistedTrap2.Cmd = new G4UIcmdWithPargs( twistedTrap2Path, this, &twistedTrap2.Args[0], twistedTrap2.Args.size() );
	twistedTrap2.Cmd->SetGuidance( "Declare a general G4TwistedTrap solid" );
	twistedTrap2.Make = &G4InteractiveSolid::MakeTwistedTrap2;
	commands.push_back(&twistedTrap2);
	//
	// Declare G4TwistedTrd
        // 
	//
	twistedTrd.Args.push_back(new G4UIcmdPargDouble( "dx1", 1.0, m ));
	twistedTrd.Args.push_back(new G4UIcmdPargDouble( "dx2", 1.0, m ));
	twistedTrd.Args.push_back(new G4UIcmdPargDouble( "dy1",  1.0, m ));
	twistedTrd.Args.push_back(new G4UIcmdPargDouble( "dy2",  1.0, m ));
	twistedTrd.Args.push_back(new G4UIcmdPargDouble( "dz",  1.0, m ));
	twistedTrd.Args.push_back(new G4UIcmdPargDouble( "phi", 0.0, deg ));
	G4String twistedTrdPath = prefix+"G4TwistedTrd";
	twistedTrd.Cmd = new G4UIcmdWithPargs( twistedTrdPath, this, &twistedTrd.Args[0], twistedTrd.Args.size() );
	twistedTrd.Cmd->SetGuidance( "Declare a regular G4TwistedTrd solid" );
	twistedTrd.Make = &G4InteractiveSolid::MakeTwistedTrd;
    commands.push_back(&twistedTrd);

	//
	// Declare G4TwistedTubs
	//
	twistedTubs.Args.push_back(new G4UIcmdPargDouble( "phi", 0.0, deg ));
	twistedTubs.Args.push_back(new G4UIcmdPargDouble( "rmin", 1.0, m ));
	twistedTubs.Args.push_back(new G4UIcmdPargDouble( "rmax", 1.0, m ));
	twistedTubs.Args.push_back(new G4UIcmdPargDouble( "zneg", 1.0, m ));
	twistedTubs.Args.push_back(new G4UIcmdPargDouble( "zpos", 1.0, m ));
	twistedTubs.Args.push_back(new G4UIcmdPargInteger( "nseg", 1 ));
	twistedTubs.Args.push_back(new G4UIcmdPargDouble( "totphi", 360.0, deg ));
	G4String twistedTubsPath = prefix+"G4TwistedTubs";
	twistedTubs.Cmd = new G4UIcmdWithPargs( twistedTubsPath, this, &twistedTubs.Args[0], twistedTubs.Args.size() );
	twistedTubs.Cmd->SetGuidance( "Declare a G4TwistedTubs solid" );
	twistedTubs.Make = &G4InteractiveSolid::MakeTwistedTubs;
	commands.push_back(&twistedTubs);
	//
	// Declare DircTest
	//
	G4String dircTestPath = prefix+"DircTest";
	dircTest.Cmd = new G4UIcmdWithPargs( dircTestPath, this, 0, 0 );
	dircTest.Cmd->SetGuidance( "Declare a DircTest solid" );
	dircTest.Make = &G4InteractiveSolid::MakeDircTest;
	commands.push_back(&dircTest);
	//
	// Declare BooleanSolid1
	//
	G4String BooleanSolid1Path = prefix+"BooleanSolid";
	booleanSolid1.Cmd = new G4UIcmdWithPargs( BooleanSolid1Path, this, 0, 0 );
	booleanSolid1.Cmd->SetGuidance( "Declare a Boolean solid #1" );
	booleanSolid1.Make = &G4InteractiveSolid::MakeBooleanSolid1;
	commands.push_back(&booleanSolid1);

	G4String MultiUnionPath = prefix+"G4MultiUnion";
	multiUnion.Args.push_back(new G4UIcmdPargInteger( "n", 1));
	multiUnion.Cmd = new G4UIcmdWithPargs( MultiUnionPath, this, &multiUnion.Args[0], multiUnion.Args.size());
	multiUnion.Cmd->SetGuidance( "Declare a Boolean solid #1" );
	multiUnion.Make = &G4InteractiveSolid::MakeMultiUnion;
	commands.push_back(&multiUnion);
	//
	G4String TessellatedSolidTransformPath = prefix+"G4TessellatedSolidTransform";
	tessellatedSolidTransform.Args.push_back(new G4UIcmdPargInteger("n", 1)); // G4UIcmdWithAString
	tessellatedSolidTransform.Cmd = new G4UIcmdWithPargs(TessellatedSolidTransformPath, this, &tessellatedSolidTransform.Args[0], tessellatedSolidTransform.Args.size());
	tessellatedSolidTransform.Cmd->SetGuidance("Declare a Tessellated Solid, by using transformation from already declated solid" );
	tessellatedSolidTransform.Make = &G4InteractiveSolid::MakeTessellatedSolidFromTransform;
	commands.push_back(&tessellatedSolidTransform);

	G4String TessellatedSolidPlainPath = prefix+"G4TessellatedSolidFromPlainFile";
	tessellatedSolidPlain.Args.push_back(new G4UIcmdPargInteger("n", 1)); // G4UIcmdWithAString
	tessellatedSolidPlain.Cmd = new G4UIcmdWithPargs(TessellatedSolidPlainPath, this, &tessellatedSolidPlain.Args[0], tessellatedSolidPlain.Args.size());
	tessellatedSolidPlain.Cmd->SetGuidance("Declare a Tessellated Solid From Plain File");
	tessellatedSolidPlain.Make = &G4InteractiveSolid::MakeTessellatedSolidFromPlainFile;
	commands.push_back(&tessellatedSolidPlain);

	G4String TessellatedSolidSTLPath = prefix+"G4TessellatedSolidFromSTLFile";
	tessellatedSolidSTL.Args.push_back(new G4UIcmdPargInteger("n", 1)); // G4UIcmdWithAString
	tessellatedSolidSTL.Cmd = new G4UIcmdWithPargs(TessellatedSolidSTLPath, this, &tessellatedSolidSTL.Args[0], tessellatedSolidSTL.Args.size());
	tessellatedSolidSTL.Cmd->SetGuidance("Declare a Tessellated Solid From STL File");
	tessellatedSolidSTL.Make = &G4InteractiveSolid::MakeTessellatedSolidFromSTLFile;
	commands.push_back(&tessellatedSolidSTL);

	G4String TessellatedSolidGdmlPath = prefix+"G4TessellatedSolidFromGDMLFile";
	tessellatedSolidGdml.Args.push_back(new G4UIcmdPargInteger("n", 0));
	tessellatedSolidGdml.Cmd = new G4UIcmdWithPargs(TessellatedSolidGdmlPath, this, &tessellatedSolidGdml.Args[0], tessellatedSolidGdml.Args.size());
	tessellatedSolidGdml.Cmd->SetGuidance("Declare a Tessellated Solid From GDML File" );
	tessellatedSolidGdml.Make = &G4InteractiveSolid::MakeTessellatedSolidFromGDMLFile;
	commands.push_back(&tessellatedSolidGdml);
}

//
// Destructor
//
G4InteractiveSolid::~G4InteractiveSolid()
{
	if (solid) delete solid;

	delete fileCmd;
	
	int size = commands.size();
	for (int i = 0; i < size; i++) 
	{
		G4SBTSolid &command = *commands[i];
		DeleteArgArray( command.Args);
	}
}


//
// DeleteArgArray
//
void G4InteractiveSolid::DeleteArgArray (vector<G4UIcmdParg *> &args)
{
	int n = args.size();
	for(int i = 0; i < n; i++) delete args[i];
}


//
// MakeBox
//
void G4InteractiveSolid::MakeBox( G4String values )
{
	if (box.Cmd->GetArguments( values )) {
		delete solid;
		
		double dx = box.GetDouble(0), dy = box.GetDouble(1), dz = box.GetDouble(2);
		solid = new G4Box( "interactiveBox", dx, dy, dz);
	}
	else
		G4cerr << "G4Box not created" << G4endl;
}

/*
G4RotationMatrix rotation;
rotation.rotateY(0);
G4ThreeVector translation(x,y,z);

G4UnionSolid  boxUnionCyl1(“box union cylinder”, &box1,&cylinder1,&roation,translation);

// the same solid with the active method


G4Transform3D transform(rotation,translation);
G4UnionSolid   boxUnionCyl2(“box union cylinder”,
			&box1,&cylinder1,transform)
*/

G4VSolid *CreateTestMultiUnion(int numNodes) // Number of nodes to implement
{
   // Instance:
      // Creation of several nodes:
   const double unionMaxX = 1.; // putting it larger can cause particles to escape
   const double unionMaxY = 1.;
   const double unionMaxZ = 1.;

   double extentVolume = 8 * unionMaxX * unionMaxY * unionMaxZ;
   double ratio = 1.0/3.0; // ratio of inside points vs (inside + outside points)   
   double length = pow (ratio * extentVolume / numNodes, 1./3.) / 2;

   G4Box *box = new G4Box("G4Box", length, length, length);
   // double capacity = 
	box->GetCubicVolume();

   double r = pow (ratio * extentVolume *3./(4 * UUtils::kPi * numNodes), 1./3.);
   G4Orb *orb = new G4Orb("G4Orb", r);

   G4Trd *trd = new G4Trd("G4Trd", length * 1.2, length * 0.9, length * 0.8, length * 1.1, length);

   G4VSolid *solids[] = {box, trd, orb};
   int solidsCount = sizeof(solids) / sizeof(G4VSolid *);

   G4VSolid *previousSolid = NULL;

     // Constructor:
	for(int i = 0; i < numNodes; i++)
	{
		double x = UUtils::Random(-unionMaxX + length, unionMaxX - length);
		double y = UUtils::Random(-unionMaxY + length, unionMaxY - length);
		double z = UUtils::Random(-unionMaxZ + length, unionMaxZ - length);
//		y = z = 0;
//		if (i == 0) x = 2, y = 3, z = 0; 
//		else x = 3, y = 2, z = 0;

		G4RotationMatrix rotation;
		G4ThreeVector translation(x,y,z);
		G4AffineTransform transformation(rotation, translation);
		G4Transform3D *transform = new G4Transform3D(rotation,translation);

		G4VSolid *solid;
		solid = solids[i % solidsCount];

		if (i == 0)
			previousSolid = new G4DisplacedSolid ("MultiUnion", solid, transformation);
		else
			previousSolid = new G4UnionSolid ("MultiUnion", previousSolid, solid, *transform /*&rotation, translation*/);
	}

   /*
   else 
   {
		const int carBoxesX = 20;
		const int carBoxesY = 20;
		const int carBoxesZ = 20;

	   // Transformation:
	   for(int n = 0, o = 0, m = 0; m < numNodes ; m++)
	   {
		   if (m >= carBoxesX*carBoxesY*carBoxesZ) break;
		   double spacing = 50;
		   double x = -unionMaxX+spacing+2*spacing*(m%carBoxesX);
		   double y = -unionMaxY+spacing+2*spacing*n;
		   double z = -unionMaxZ+spacing+2*spacing*o;

		  arrayTransformations[m] = new UTransform3D(x,y,z,0,0,0);
		  multiUnion->AddNode(*box,arrayTransformations[m]);
           
		  // Preparing "Draw":
		  if (m % carBoxesX == carBoxesX-1)
		  {
			  if (n % carBoxesY == carBoxesY-1)
			  {
				 n = 0;
				 o++;
			  }      
			  else n++;
		  }
	   }
   }
   */

   return previousSolid;
}

//
// MakeUMultiUnion
//
void G4InteractiveSolid::MakeUMultiUnion( G4String values )
{
	if (umultiunion.Cmd->GetArguments( values )) {
		delete solid; //NOTE: At VS2010 it used to crash, although currently it is OK

		int n = umultiunion.GetInteger(0);	
		// CODE FOR TESTS WITH THREE SOLIDS IS AT OTHER PLACE
		// !!!!!!!!!!!!!!!!!!!!!
		VUSolid* tmp = UMultiUnion::CreateTestMultiUnion(n);
		// !!!!!!!!!!!!!!!!!!!!!
	
		solid = new G4USolid("interactiveUMultiUnion",tmp);
	}
	else
		G4cerr << "UMultiUnion not created" << G4endl;
}

//
// MakeUBox
//
void G4InteractiveSolid::MakeUBox( G4String values )
{
	if (ubox.Cmd->GetArguments( values )) {
		delete solid;

		double dx = ubox.GetDouble(0), dy = ubox.GetDouble(1), dz = ubox.GetDouble(2);

		VUSolid* tmp = new UBox("interactiveBox", dx, dy, dz);

		solid = new G4USolid("interactiveUBox",tmp );
	}
	else
		G4cerr << "UBox not created" << G4endl;
}

// MakeUOrb
//
void G4InteractiveSolid::MakeUOrb( G4String values )
{
	if (uorb.Cmd->GetArguments( values )) {
			delete solid;
			
		double r = uorb.GetDouble(0);
        VUSolid* tmp = new UOrb( "interactiveUOrb", r*1000);

		solid = new G4USolid("interactiveUOrb", tmp);
	}
	else
		G4cerr << "UOrb not created" << G4endl;
}

// MakeUTrd
//
void G4InteractiveSolid::MakeUTrd( G4String values )
{
	if (utrd.Cmd->GetArguments( values )) {
		delete solid; //NOTE: At VS2010 it used to crash, although currently it is OK
		
		double fDx1 = utrd.GetDouble(0), fDx2 = utrd.GetDouble(1);
		double fDy1 = utrd.GetDouble(2), fDy2 = utrd.GetDouble(3), fDz = utrd.GetDouble(4);
	
        VUSolid* tmp=new UTrd( "interactiveUTrd", fDx1, fDx2, fDy1, fDy2, fDz);
		
		solid = new G4USolid("interactiveUTrd", tmp);
	}
	else
		G4cerr << "UTrd not created" << G4endl;
}
 
//
// MakeCons
//
void G4InteractiveSolid::MakeCons( G4String values )
{
	if (cons.Cmd->GetArguments( values )) {
		delete solid;
		
		solid = new G4Cons( "interactiveCons", cons.GetDouble(0),
						       cons.GetDouble(1), cons.GetDouble(2),
						       cons.GetDouble(3), cons.GetDouble(4),
						       cons.GetDouble(5), cons.GetDouble(6));
	}
	else
		G4cerr << "G4Cons not created" << G4endl;
}


//
// MakeOrb
//
void G4InteractiveSolid::MakeOrb( G4String values )
{
	if (orb.Cmd->GetArguments( values )) {
		delete solid;
		solid = new G4Orb( "interactiveOrb", orb.GetDouble(0));
	}
	else
		G4cerr << "G4Orb not created" << G4endl;
}


//
// MakePara
//
void G4InteractiveSolid::MakePara( G4String values )
{
	if (para.Cmd->GetArguments( values )) {
		delete solid;
		
		double dx = para.GetDouble(0), dy = para.GetDouble(1), dz = para.GetDouble(2);
		double alpha = para.GetDouble(3), theta = para.GetDouble(4), phi = para.GetDouble(5);
		
		solid = new G4Para( "interactivePara", dx, dy, dz, alpha, theta, phi);
	}
	else
		G4cerr << "G4Para not created" << G4endl;
}


//
// MakeSphere
//
void G4InteractiveSolid::MakeSphere( G4String values )
{
	if (sphere.Cmd->GetArguments( values )) {
		delete solid;
			
		solid = new G4Sphere( "interactiveSphere", sphere.GetDouble(0),
						           sphere.GetDouble(1), sphere.GetDouble(2),
						           sphere.GetDouble(3), sphere.GetDouble(4),
						           sphere.GetDouble(5));
	}
	else
		G4cerr << "G4Sphere not created" << G4endl;
}


//
// MakeTorus
//
void G4InteractiveSolid::MakeTorus( G4String values )
{
	if (torus.Cmd->GetArguments( values )) {
		delete solid;
				
		solid = new G4Torus( "interactiveTorus", torus.GetDouble(0),
						         torus.GetDouble(1), torus.GetDouble(2),
						         torus.GetDouble(3), torus.GetDouble(4));
	}
	else
		G4cerr << "G4Torus not created" << G4endl;
}


//
// MakeTrap
//
void G4InteractiveSolid::MakeTrap( G4String values )
{
	if (trap.Cmd->GetArguments( values )) {
		delete solid;
			
		solid = new G4Trap( "interactiveTrap", trap.GetDouble(0),
						       trap.GetDouble(1), trap.GetDouble(2),
						       trap.GetDouble(3), trap.GetDouble(4),
						       trap.GetDouble(5), trap.GetDouble(6),
						       trap.GetDouble(7), trap.GetDouble(8),
						       trap.GetDouble(9), trap.GetDouble(10));
	}
	else
		G4cerr << "G4Trap not created" << G4endl;
}


//
// MakeTrd
//
void G4InteractiveSolid::MakeTrd( G4String values )
{
	if (trd.Cmd->GetArguments( values )) {
		delete solid;
		
		double dx1 = trd.GetDouble(0), dx2 = trd.GetDouble(1), dy1 = trd.GetDouble(2),
			   dy2 = trd.GetDouble(3), dz = trd.GetDouble(4);
		
		solid = new G4Trd( "interactiveTrd", dx1, dx2, dy1, dy2, dz);
	}
	else
		G4cerr << "G4Trd not created" << G4endl;
}


//
// MakeTubs
//
void G4InteractiveSolid::MakeTubs( G4String values )
{
	if (tubs.Cmd->GetArguments( values )) {
		delete solid;
			
		solid = new G4Tubs( "interactiveTubs", tubs.GetDouble(0),
						       tubs.GetDouble(1), tubs.GetDouble(2),
						       tubs.GetDouble(3), tubs.GetDouble(4));
	}
	else
		G4cerr << "G4Tubs not created" << G4endl;
}


//
// MakeEllipsoid
//
void G4InteractiveSolid::MakeEllipsoid( G4String values )
{
	if (ellipsoid.Cmd->GetArguments( values )) {
		delete solid;
		
		double dx = ellipsoid.GetDouble(0), dy = ellipsoid.GetDouble(1), dz = ellipsoid.GetDouble(2),
			   pzBottomCut = ellipsoid.GetDouble(3), pzTopCut = ellipsoid.GetDouble(4);
		
		solid = new G4Ellipsoid( "interactiveEllipsoid", dx, dy, dz, pzBottomCut, pzTopCut);
	}
	else
		G4cerr << "G4Ellipsoid not created" << G4endl;
}


//
// MakeEllipticalTube
//
void G4InteractiveSolid::MakeEllipticalCone( G4String values )
{
	if (elCone.Cmd->GetArguments( values )) {
		delete solid;
		
		double dx = elCone.GetDouble(0), dy = elCone.GetDouble(1), dz = elCone.GetDouble(2);
		double pzTopCut = elCone.GetDouble(3);
                                  
        G4cout << "Making G4EllipticalCone: " <<  dx << " " <<  dy << " " <<  dz << " " <<  pzTopCut << G4endl;
		
		solid = new G4EllipticalCone( "interactiveEllipticalCone",  dx, dy, dz, pzTopCut);
	}
	else
		G4cerr << "G4EllipticalCone not created" << G4endl;
}

//
// MakeEllipticalTube
//
void G4InteractiveSolid::MakeEllipticalTube( G4String values )
{
	if (elTube.Cmd->GetArguments( values )) {
		delete solid;
		
		double dx = elTube.GetDouble(0), dy = elTube.GetDouble(1), dz = elTube.GetDouble(2);
		
		solid = new G4EllipticalTube( "interactiveEllipticalTube", dx, dy, dz);
	}
	else
		G4cerr << "G4EllipticalTube not created" << G4endl;
}

//
// MakeExtrudedSolid
//
void G4InteractiveSolid::MakeExtrudedSolid( G4String values )
{
	if (extruded.Cmd->GetArguments( values )) {
		delete solid;
                
		int numPoints = extruded.GetInteger(0);

		G4UIcmdPargListDouble &pgonxArg = extruded.GetArgListDouble(1);
		G4UIcmdPargListDouble &pgonyArg = extruded.GetArgListDouble(2);

		int numSides = extruded.GetInteger(3);

		G4UIcmdPargListDouble &zArg = extruded.GetArgListDouble(4);
		G4UIcmdPargListDouble &offxArg = extruded.GetArgListDouble(5);
		G4UIcmdPargListDouble &offyArg = extruded.GetArgListDouble(6);

		double scale = extruded.GetDouble(7);

		std::vector<G4TwoVector> polygon;
		for (int i = 0; i < numPoints; ++i)
		{
			polygon.push_back(G4TwoVector(pgonxArg.GetValues()[i], pgonyArg.GetValues()[i]));
		}   
		
        std::vector<G4ExtrudedSolid::ZSection> zsections;
        
		for (int i=0; i<numSides; ++i)
		{        
			zsections.push_back(G4ExtrudedSolid::ZSection(
				zArg.GetValues()[i], 
				G4TwoVector(offxArg.GetValues()[i], offyArg.GetValues()[i]), scale));
		}
                
        solid = new G4ExtrudedSolid("interactiveExtrudedSolid", polygon, zsections);                                        
	}
	else
		G4cerr << "G4ExtrudedSolid not created" << G4endl;
}

//
// MakeHype
//
void G4InteractiveSolid::MakeHype( G4String values )
{
	if (hype.Cmd->GetArguments( values )) {
		delete solid;
				
		solid = new G4Hype( "interactiveHype", hype.GetDouble(0), hype.GetDouble(1),
						       hype.GetDouble(2), hype.GetDouble(3), hype.GetDouble(4));
        G4cout << *solid << G4endl;                                                       
	}
	else
		G4cerr << "G4Hype not created" << G4endl;
}


//
// MakePolycone
//
void G4InteractiveSolid::MakePolycone( G4String values )
{
	if (polycone.Cmd->GetArguments( values )) {
		double phiStart = polycone.GetDouble(0), phiTotal = polycone.GetDouble(1),
				numRZ    = polycone.GetDouble(2);

		G4UIcmdPargListDouble &rArg = polycone.GetArgListDouble(3);
		G4UIcmdPargListDouble &zArg = polycone.GetArgListDouble(4);
		//
		// Check consistency
		//
		if (numRZ != rArg.GetNItem() || numRZ != zArg.GetNItem())
		{
		    G4cerr << "numRZ inconsistent among polycone arguments" << G4endl;
			G4cerr << "G4Polycone not created" << G4endl;
			return;
		}
		
		delete solid;

		solid = new G4Polycone( "interactivePolycone", 
					phiStart, phiTotal, (int) numRZ, rArg.GetValues(), zArg.GetValues() );
	}
	else
		G4cerr << "G4Polycone not created" << G4endl;
}


//
// MakePolycone2
//
void G4InteractiveSolid::MakePolycone2( G4String values )
{
	if (polycone2.Cmd->GetArguments( values )) {
		double phiStart = polycone2.GetDouble(0), phiTotal = polycone2.GetDouble(1);
		int numRZ = (int) polycone2.GetDouble(2);

		G4UIcmdPargListDouble &zArg = polycone2.GetArgListDouble(3);
		G4UIcmdPargListDouble &rInArg = polycone2.GetArgListDouble(4);
		G4UIcmdPargListDouble &rOutArg = polycone2.GetArgListDouble(5);
		//
		// Check consistency
		//
		if (numRZ != zArg.GetNItem() ||
		    numRZ != rInArg.GetNItem() ||
		    numRZ != rOutArg.GetNItem()    ) {
		    	G4cerr << "numRZ inconsistent among polycone arguments" << G4endl;
			G4cerr << "G4Polycone not created" << G4endl;
			return;
		}
		
		delete solid;
		solid = new G4Polycone( "interactivePolycone", 
					phiStart, phiTotal, numRZ,
					zArg.GetValues(), rInArg.GetValues(), rOutArg.GetValues() );
	}
	else
		G4cerr << "G4Polycone not created" << G4endl;
}


//
// MakePolyhedra
//
void G4InteractiveSolid::MakePolyhedra( G4String values )
{
	if (polyhedra.Cmd->GetArguments( values )) {
		double phiStart = polyhedra.GetDouble(0), phiTotal = polyhedra.GetDouble(1);
		int numSides = (int) polyhedra.GetDouble(2), numRZ = polyhedra.GetInteger(3);
		G4UIcmdPargListDouble &rArg = polyhedra.GetArgListDouble(4);
		G4UIcmdPargListDouble &zArg = polyhedra.GetArgListDouble(5);
		//
		// Check consistency
		//
		if (numRZ != rArg.GetNItem() ||
		    numRZ != zArg.GetNItem()    ) {
		    	G4cerr << "numRZ inconsistent among polyhedra arguments" << G4endl;
			G4cerr << "G4Polyhedra not created" << G4endl;
			return;
		}
		
		delete solid;
		solid = new G4Polyhedra( "interactivePolyhedra", 
					phiStart, phiTotal, numSides, numRZ,
					rArg.GetValues(), zArg.GetValues());
	}
	else
		G4cerr << "G4Polyhedra not created" << G4endl;
}


//
// MakePolyhedra2
//
void G4InteractiveSolid::MakePolyhedra2( G4String values )
{
	if (polyhedra2.Cmd->GetArguments( values )) {
		double phiStart = polyhedra2.GetDouble(0), phiTotal = polyhedra2.GetDouble(1),
			numSides = polyhedra2.GetDouble(2);
		int numRZ = polyhedra2.GetInteger(3);
		G4UIcmdPargListDouble &zArg = polyhedra2.GetArgListDouble(4);
		G4UIcmdPargListDouble &rinArg = polyhedra2.GetArgListDouble(5);
		G4UIcmdPargListDouble &routArg = polyhedra2.GetArgListDouble(6);
		//
		// Check consistency
		//
		if (numRZ != zArg.GetNItem() ||
		    numRZ != rinArg.GetNItem()  ||
		    numRZ != routArg.GetNItem()    ) {
		    	G4cerr << "numRZ inconsistent among polyhedra arguments" << G4endl;
			G4cerr << "G4Polyhedra not created" << G4endl;
			return;
		}
		
		delete solid;
		solid = new G4Polyhedra( "interactivePolyhedra", 
					phiStart, phiTotal, (int) numSides, numRZ,
					zArg.GetValues(), rinArg.GetValues(), routArg.GetValues() );
	}
	else
		G4cerr << "G4Polyhedra not created" << G4endl;
}


//
// MakeTessellatedSolid
//
void G4InteractiveSolid::MakeTessellatedSolid( G4String values )
{
	if (tessel.Cmd->GetArguments( values )) {

		int num3 = tessel.GetInteger(0);
		
		G4UIcmdPargListDouble &p1in3Arg = tessel.GetArgListDouble(1),
		                      &p2in3Arg = tessel.GetArgListDouble(2),
	 	                      &p3in3Arg = tessel.GetArgListDouble(3);
                
		int num4 = tessel.GetInteger(4);

		G4UIcmdPargListDouble &p1in4Arg = tessel.GetArgListDouble(5),
		                      &p2in4Arg = tessel.GetArgListDouble(6),
		                      &p3in4Arg = tessel.GetArgListDouble(7),
		                      &p4in4Arg = tessel.GetArgListDouble(8);
		//
		// Check consistency
		//
		int nump1in3 = p1in3Arg.GetNItem();
		int nump2in3 = p2in3Arg.GetNItem();
		int nump3in3 = p3in3Arg.GetNItem();
		if ( nump1in3 != 3*num3 || nump2in3 != 3*num3 || nump3in3 != 3*num3 ) {
		    	G4cerr << "Wrong number of points coordinates among triangular tessel arguments" << G4endl;
			G4cerr << "G4TessellatedSolid not created" << G4endl;
			return;
		}
		int nump1in4 = p1in4Arg.GetNItem();
		int nump2in4 = p2in4Arg.GetNItem();
		int nump3in4 = p3in4Arg.GetNItem();
		int nump4in4 = p4in4Arg.GetNItem();
		if ( nump1in4 != 3*num4 || nump2in4 != 3*num4 || nump3in4 != 3*num4 || nump4in4 != 3*num4) {
		    	G4cerr << "Wrong number of points coordinates among quadrangular tessel arguments" << G4endl;
			G4cerr << "G4TessellatedSolid not created" << G4endl;
			return;
		}
		
		delete solid;
                                   
		G4TessellatedSolid* tessel 
                  = new G4TessellatedSolid( "interactiveTessellatedSolid");
                  
                for ( G4int i=0; i<num3; ++i) {
                  G4ThreeVector p1(p1in3Arg.GetValues()[3*i+0], p1in3Arg.GetValues()[3*i+1], p1in3Arg.GetValues()[3*i+2]);
                  G4ThreeVector p2(p2in3Arg.GetValues()[3*i+0], p2in3Arg.GetValues()[3*i+1], p2in3Arg.GetValues()[3*i+2]);
                  G4ThreeVector p3(p3in3Arg.GetValues()[3*i+0], p3in3Arg.GetValues()[3*i+1], p3in3Arg.GetValues()[3*i+2]);
				  G4VFacet *facet = new G4TriangularFacet(p1, p2, p3, ABSOLUTE); 
                  tessel->AddFacet(facet);
                }  
                
                for ( G4int i=0; i<num4; ++i) {
                  G4ThreeVector p1(p1in4Arg.GetValues()[3*i+0], p1in4Arg.GetValues()[3*i+1], p1in4Arg.GetValues()[3*i+2]);
                  G4ThreeVector p2(p2in4Arg.GetValues()[3*i+0], p2in4Arg.GetValues()[3*i+1], p2in4Arg.GetValues()[3*i+2]);
                  G4ThreeVector p3(p3in4Arg.GetValues()[3*i+0], p3in4Arg.GetValues()[3*i+1], p3in4Arg.GetValues()[3*i+2]);
                  G4ThreeVector p4(p4in4Arg.GetValues()[3*i+0], p4in4Arg.GetValues()[3*i+1], p4in4Arg.GetValues()[3*i+2]);
                  tessel->AddFacet(new G4QuadrangularFacet(p1, p2, p3, p4, ABSOLUTE));
                }
                tessel->SetSolidClosed(true);
                G4cout << *tessel << G4endl;
                solid = tessel;  
               
	}
	else
		G4cerr << "G4TessellatedSolid not created" << G4endl;
}

//
// MakeTessellatedSolid2
//
void G4InteractiveSolid::MakeTessellatedSolid2( G4String values )
{
	if (tessel2.Cmd->GetArguments( values )) {
		delete solid;
                
		int numPoints = tessel2.GetInteger(0);
		G4UIcmdPargListDouble &pgonxArg = tessel2.GetArgListDouble(1),
				&pgonyArg = tessel2.GetArgListDouble(2);
                                  
                                  
		int numSides = tessel2.GetInteger(1);
        G4UIcmdPargListDouble &zArg = tessel2.GetArgListDouble(4),
				        &offxArg = tessel2.GetArgListDouble(5), &offyArg = tessel2.GetArgListDouble(6),
				       &scaleArg = tessel2.GetArgListDouble(7);
                
                std::vector<G4TwoVector> polygon;
                for ( G4int i=0; i<numPoints; ++i ) {
                  polygon.push_back(G4TwoVector(pgonxArg.GetValues()[i], pgonyArg.GetValues()[i]));
                }   
		
                std::vector<G4ExtrudedSolid::ZSection> zsections;
                for ( G4int i=0; i<numSides; ++i ) {
                  zsections.push_back(G4ExtrudedSolid::ZSection(
                                        zArg.GetValues()[i],
                                        G4TwoVector(offxArg.GetValues()[i], offyArg.GetValues()[i]),
                                        scaleArg.GetValues()[i]));
                }
                
                G4ExtrudedSolid* xtru
                  = new G4ExtrudedSolid("interactiveTessellatedSolid", polygon, zsections); 
                solid = new G4TessellatedSolid(*xtru);
                delete xtru;
                                                       
	}
	else
		G4cerr << "G4TessellatedSolid not created" << G4endl;
}


//
// MakeTet
//
void G4InteractiveSolid::MakeTet( G4String values )
{
	if (tet.Cmd->GetArguments( values )) {
		G4UIcmdPargListDouble &p1Arg = tet.GetArgListDouble(0), &p2Arg = tet.GetArgListDouble(1), &p3Arg = tet.GetArgListDouble(2), &p4Arg = tet.GetArgListDouble(3);
				  
		//
		// Check consistency
		//
		G4int numCoor1 = p1Arg.GetNItem();
		G4int numCoor2 = p2Arg.GetNItem();
		G4int numCoor3 = p3Arg.GetNItem();
		G4int numCoor4 = p4Arg.GetNItem();
		if (numCoor1 != 3 || numCoor2 != 3 || numCoor3 != 3 || numCoor4 != 3 ) {
		    	G4cerr << "Wrong number of points coordinates among tet arguments" << G4endl;
			G4cerr << "G4Tet not created" << G4endl;
			return;
		}
		
		delete solid;
                G4ThreeVector p1(p1Arg.GetValues()[0], p1Arg.GetValues()[1], p1Arg.GetValues()[2]);
                G4ThreeVector p2(p2Arg.GetValues()[0], p2Arg.GetValues()[1], p2Arg.GetValues()[2]);
                G4ThreeVector p3(p3Arg.GetValues()[0], p3Arg.GetValues()[1], p3Arg.GetValues()[2]);
                G4ThreeVector p4(p4Arg.GetValues()[0], p4Arg.GetValues()[1], p4Arg.GetValues()[2]);
                                   
		solid = new G4Tet( "interactiveTet", p1, p2, p3, p4 );
	}
	else
		G4cerr << "G4Tet not created" << G4endl;
}


//
// MakeBox
//
void G4InteractiveSolid::MakeTwistedBox( G4String values )
{
	if (twistedBox.Cmd->GetArguments( values )) {
		delete solid;
		
		double phi = twistedBox.GetDouble(0),
		dx = twistedBox.GetDouble(1), dy = twistedBox.GetDouble(2), dz = twistedBox.GetDouble(3);
		
		solid = new G4TwistedBox( "interactiveTwistedBox", phi, dx, dy, dz);                                     G4cout << *solid << G4endl;                                          
	}
	else
		G4cerr << "G4TwistedBox not created" << G4endl;
}

//
// MakeTwistedTrap
//
void G4InteractiveSolid::MakeTwistedTrap( G4String values )
{
	if (twistedTrap.Cmd->GetArguments( values )) {
		delete solid;
		
		double phi = twistedTrap.GetDouble(0);
		double dx1 = twistedTrap.GetDouble(1), dx2 = twistedTrap.GetDouble(2);
		double dy  = twistedTrap.GetDouble(3), dz = twistedTrap.GetDouble(4);
		
		solid = new G4TwistedTrap( "interactiveTwistedTrap", phi, dx1, dx2, dy, dz);
	}
	else
		G4cerr << "G4TwistedTrap not created" << G4endl;
}


//
// MakeTwistedTrap
//
void G4InteractiveSolid::MakeTwistedTrap2( G4String values )
{
	if (twistedTrap2.Cmd->GetArguments( values )) {
		delete solid;
			
		solid = new G4TwistedTrap( "interactiveTwistedTrap2", 
					   twistedTrap2.GetDouble(0), twistedTrap2.GetDouble(1),
					   twistedTrap2.GetDouble(2), twistedTrap2.GetDouble(3),
					   twistedTrap2.GetDouble(4), twistedTrap2.GetDouble(5),
					   twistedTrap2.GetDouble(6), twistedTrap2.GetDouble(7),
					   twistedTrap2.GetDouble(8), twistedTrap2.GetDouble(9),
					   twistedTrap2.GetDouble(10));
	}
	else
		G4cerr << "G4TwistedTrap2 not created" << G4endl;
}

//
// MakeTwistedTrd
//
void G4InteractiveSolid::MakeTwistedTrd( G4String values )
{
	if (twistedTrd.Cmd->GetArguments( values )) {
		delete solid;
		
		double dx1 = twistedTrd.GetDouble(0), dx2 = twistedTrd.GetDouble(1),
			dy1 = twistedTrd.GetDouble(2), dy2 = twistedTrd.GetDouble(3),
			dz  = twistedTrd.GetDouble(4), phi = twistedTrd.GetDouble(5);
				  
		solid = new G4TwistedTrd( "interactiveTwistedTrd", dx1, dx2, dy1, dy2, dz, phi);
	} 
	else
		G4cerr << "G4TwistedTrd not created" << G4endl;
}


//
// MakeTwistedTubs
//
void G4InteractiveSolid::MakeTwistedTubs( G4String values )
{
	if (twistedTubs.Cmd->GetArguments( values )) {
		delete solid;
		
		double phi = twistedTubs.GetDouble(0), rmin = twistedTubs.GetDouble(1),
		rmax = twistedTubs.GetDouble(2), zneg = twistedTubs.GetDouble(3),
		zpos = twistedTubs.GetDouble(4), nseg = twistedTubs.GetDouble(5),
		totphi = twistedTubs.GetDouble(6);
		
		solid = new G4TwistedTubs( "interactiveTwistedTubs", phi, rmin, rmax, zneg, zpos, (int) nseg, totphi);
                                           
        G4cout << *solid << G4endl;                                    
	}
	else
		G4cerr << "G4TwistedTubs not created" << G4endl;
}



//
// MakeDircTest
//
void G4InteractiveSolid::MakeDircTest(G4String /*values*/)
{
	delete solid;

	G4Tubs *outside = new G4Tubs( "OuterFrame",	// name (arbitrary) 
				      1.0*m, 		// inner radius
				      1.1*m, 		// outer radius
				      0.01*m, 		// half-thickness in z
				      -15*deg, 		// start angle
				      30*deg );		// total angle
				      
	G4Box *cutout = new G4Box( "Cutout", 	// name (arbitrary)
				   0.02*m,	// half-width (x)
				   0.25*m,	// half-height (y)
				   0.01001*m );	// half-thickness (z)
				   
	G4Transform3D tran = G4Translate3D( 1.03*m, 0.0, 0.0 );
	
	solid = new G4SubtractionSolid( "drcExample", outside, cutout, tran );
}

//
// BooleanSolid1Test
//
void G4InteractiveSolid::MakeBooleanSolid1(G4String)
{
  /*
    G4IntersectionSolid.hh  G4SubtractionSolid.hh  G4UnionSolid.hh
    all CSGs : Box Tubs Sphere Cons Torus
    So: Boolean type operation and 2 CSG Objects with parameters for each (..)
    plus a transformation to apply to the second solid
   */
	delete solid;
	BooleanOp OperationType;

	//OperationType = INTERSECTION;
	OperationType = SUBTRACTION;
		
	/*
	G4Tubs *outside = new G4Tubs( "OuterFrame",	// name (arbitrary) 
				      1.0*m, 		// inner radius
				      1.1*m, 		// outer radius
				      0.50*m, 		// half-thickness in z
				      0*deg, 		// start angle
				      180*deg ));		// total angle
	*/
	/*
	G4Torus *outside = new G4Torus( "interactiveTorus",
					0.2*m,
				        0.8*m,
				        1.4*m,
				        0*deg,
				        360*deg ));
	*/
	
	G4Cons *outside = new G4Cons( "OuterFrame",
				      0.6*m, // pRmin1
				      1.0*m, // pRmax1
				      0.2*m, // pRmin2
				      0.8*m, // pRmax2
				      0.2*m,
				      0*deg,
				      180*deg );
		
	/* Dirctest Box cutout
	G4Box *cutout = new G4Box( "Cutout", 	// name (arbitrary)
				   0.02*m,	// half-width (x)
				   0.25*m,	// half-height (y)
				   0.01001*m ));	// half-thickness (z)
	*/
	
	/*
	G4Tubs *cutout = new G4Tubs("AnotherTubs",
				    1.0*m,
				    1.1*m,
				    0.50*m,
				    0*deg,
				    180*deg
				    );
	*/

	
	 G4Cons *cutout = new G4Cons( "OuterFrame",
				      0.6*m, // pRmin1
				      1.0*m, // pRmax1
				      0.2*m, // pRmin2
				      0.8*m, // pRmax2
				      0.2*m,
				      0*deg,
				      180*deg );
	
	/*
	G4Torus *cutout = new G4Torus( "interactiveTorus",
					0.2*m,
				        0.8*m,
				        1.4*m,
				        0*deg,
				        360*deg );

	*/
	
	
	G4RotationMatrix rm;
	rm.rotateY(pi/4.0);

	G4Transform3D tran = G4Transform3D(rm,G4ThreeVector(0.0,0.0,0.0));
	
	/* G4Transform3D tran = G4Translate3D( 1.03*m, 0.0, 0.0 ); */

	switch (OperationType) {
	case INTERSECTION:
	  solid = new G4IntersectionSolid( "drcExample", outside, cutout, tran );
	  break;	
	case SUBTRACTION:
	  solid = new G4SubtractionSolid( "drcExample", outside, cutout, tran );
	  break;	
	case UNION:
	  solid = new G4UnionSolid( "drcExample", outside, cutout, tran );
	  break;	
	}
	
}



void G4InteractiveSolid::MakeMultiUnion(G4String values)
{
  /*
    G4IntersectionSolid.hh  G4SubtractionSolid.hh  G4UnionSolid.hh
    all CSGs : Box Tubs Sphere Cons Torus
    So: Boolean type operation and 2 CSG Objects with parameters for each (..)
    plus a transformation to apply to the second solid
   */
	if (multiUnion.Cmd->GetArguments( values )) {
		delete solid; 

		int n = multiUnion.GetInteger(0);
		solid = CreateTestMultiUnion(n);
	}
	else
		G4cerr << "MultiUnion not created" << G4endl;
}

void G4InteractiveSolid::CreateTessellatedSolid(const vector<UVector3> &vertices, const vector<vector<int> > &nodes)
{
    G4TessellatedSolid &tessel = *new G4TessellatedSolid("interactiveTessellatedSolid");
    for (int i = 0; i < (int) nodes.size(); i++)
    {
        const vector<int> &node = nodes[i];
        G4VFacet *facet;
        int n = node.size();
        vector<G4ThreeVector> v(n);
        for (int j = 0; j < n; j++)
        {
            const UVector3 &vertice = vertices[node[j]-1];
            v[j].set(vertice.x, vertice.y, vertice.z);
        }
        switch (n)
        {
            case 3:
                facet = new G4TriangularFacet(v[0], v[1], v[2], ABSOLUTE);
                break;
            case 4:
                facet = new G4QuadrangularFacet(v[0], v[1], v[2], v[3], ABSOLUTE);
                break;
        }
        tessel.AddFacet(facet);
    }
    tessel.SetSolidClosed(true);
    solid = &tessel;
}

void G4InteractiveSolid::MakeTessellatedSolidFromTransform(G4String /*values*/)
{
//	MakeMultiUnion(values);

    vector<UVector3> vertices;
    vector<vector<int> > nodes;
    G4Tools::GetPolyhedra(*solid, vertices, nodes);
	CreateTessellatedSolid(vertices, nodes);
}


#include <iostream>
#include <fstream>

void G4InteractiveSolid::MakeTessellatedSolidFromPlainFile(G4String /*values*/)
{
	vector<UVector3> vertices;
	vector<vector<int> > nodes;

	// TODO: na linuxu to nefunguje, asi problemy s konci radku souboru!!!

	G4String filename = SBTrun::GetCurrentFilename();

	G4String svertices = filename+".vertices";
	G4String striangles = filename+".triangles";

	cout << "Reading " << svertices << "\n";
	ifstream fvertices(svertices);

	int count = 0;
	while (!fvertices.eof())
	{
		UVector3 v;
		fvertices >> v.x >> v.y >> v.z;
		vertices.push_back(v);
		if (count++ % 10000 == 0)
			cout << count << "\n";
	}
	vector<int> node(3);

	cout << "Reading " << striangles << "\n";

	ifstream ftriangles(striangles);
	while (!ftriangles.eof())
	{
		ftriangles >> --node[0] >> --node[1] >> --node[2];
		nodes.push_back(node);
		if (count++ % 10000 == 0)
			cout << count << "\n";
	}

	cout << "Read " << vertices.size() << " vertices and " << nodes.size() << " nodes\n";

//	exit(0);

	CreateTessellatedSolid(vertices, nodes);
}




void G4InteractiveSolid::MakeTessellatedSolidFromSTLFile(G4String values)
{
	if (tessellatedSolidGdml.Cmd->GetArguments( values )) 
	{
		delete solid;
		
		int maxVoxels = tessellatedSolidGdml.GetInteger(0);
		G4String filename = SBTrun::GetCurrentFilename();
		
		solid = USTL::ReadFromSTLBinaryFile(filename, maxVoxels);
	}
}

void G4InteractiveSolid::MakeTessellatedSolidFromGDMLFile(G4String values)
{
	if (tessellatedSolidGdml.Cmd->GetArguments( values )) 
	{
		delete solid;
		
//		double phi = tessellatedSolidGdml.GetDouble(0), rmin = tessellatedSolidGdml.GetDouble(1);
		SBTrun::maxVoxels = tessellatedSolidGdml.GetInteger(0); // though this value is retreived, it cannot be used in GDML parser => ignored

		G4String filename = SBTrun::GetCurrentFilename();

		solid = USTL::ReadGDML(filename);
   }
//   MakeTessellatedSolidFromPlainFile(values);
}

// G4VPhysicalVolume* ExN03DetectorConstruction::MakeDircTest()
  
//
// SetNewValue
//
// Invoked by the UI when the user enters a command
//
void G4InteractiveSolid::SetNewValue( G4UIcommand *command, G4String newValues )
{
  /*
    We want to retrieve the current solid
    So we keep the current solid command
   */
  G4String CurrentSolid = command->GetCommandPath() + " " + newValues ;

  SBTrun::SetCurrentSolid (CurrentSolid);
  
  if (command == fileCmd) {
		SBTrun::SetCurrentFilename(newValues);
		return;
  }

  G4String path = command->GetCommandPath();

	int size = commands.size();
	for (int i = 0; i < size; i++)
	{
		G4SBTSolid &cmd = *commands[i];
		if (command == cmd.Cmd) 
		{
			(*this.*cmd.Make)(newValues); 
			return;
		}
	}

	G4Exception("G4InteractiveSolid::SetNewValue", "", FatalException, "Unrecognized command");
}


//
// GetCurrentValue
//
G4String G4InteractiveSolid::GetCurrentValue( G4UIcommand *command )
{
	if (command == fileCmd) {
		return SBTrun::GetCurrentFilename();
	}

	int size = commands.size();
	for (int i = 0; i < size; i++)
	{
		G4SBTSolid &cmd = *commands[i];
		if (command == cmd.Cmd) 
			return ConvertArgsToString(cmd.Args);
	}

	G4Exception("G4InteractiveSolid::GetCurrentValue", "", FatalException, "Unrecognized command" );
	return "foo!";
}


//
// Convert.argsToString
//
G4String G4InteractiveSolid::ConvertArgsToString( vector<G4UIcmdParg *> &args)
{
	G4String answer = "(";
	int n = args.size();
	for (int i = 0; i < n; i++) 
	{
		if (i) answer += ",";
		answer += args[i]->ConvertToString();
	}	
	return answer + ")";
}
