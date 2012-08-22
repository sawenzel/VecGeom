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
// G4InteractiveSolid.hh
//
// A messenger that allows one to construct a solid interactively
// (i.e. via command line)
//
// The solid thus created can be recovered using the G4SolidQuery
// method GetSolid.
//
// Notes:
//    * The G4UIcommand design is somewhat inflexible. It would have
//      been much better to specify each command argument as a
//      class of it's own (with input methods, help methods, etc.)
//      then to create the monolithic monstrosities like G4UICmdWith****.
//      Alas, I'm tempted to fix this, but for the sake of expediency
//      I will cheat and use my own string interpretator. The 
//      side effect is that interactive help is much compromised.
//

#ifndef G4InteractiveSolid_hh
#define G4InteractiveSolid_hh

#include "G4UImessenger.hh"
#include "G4SolidQuery.hh"

#include "G4UIcmdPargInteger.hh"
#include "G4UIcmdPargDouble.hh"
#include "G4UIcmdPargListDouble.hh"
#include "G4UIcmdWithAString.hh"

#include "UVector3.hh"

class G4UIdirectory;
class G4UIcommand;
class G4UIcmdWithPargs;
class G4UIcmdParg;

struct G4InteractiveSolid;

class G4SBTSolid
{
    public:
  
		std::vector<G4UIcmdParg *> Args;
        G4UIcmdWithPargs *Cmd;
        void (G4InteractiveSolid::*MakeMe)(G4String values);

        inline double GetDouble(int i)
        {
            G4UIcmdPargDouble &dArg = *(G4UIcmdPargDouble *)Args[i];
            return dArg.GetValue();
        }

        inline G4UIcmdPargListDouble &GetArgListDouble(int i)
        {
            G4UIcmdPargListDouble *dArg = (G4UIcmdPargListDouble *)&Args[i];
            return *dArg;
        }

        inline double GetInteger(int i)
        {
            G4UIcmdPargInteger &dArg = *(G4UIcmdPargInteger *)Args[i];
            return dArg.GetValue();
        }
};


class G4InteractiveSolid : public G4UImessenger, public G4SolidQuery {
public:
  G4InteractiveSolid( const G4String &commandPrefix );
  virtual ~G4InteractiveSolid();
	
  inline G4VSolid *GetSolid() const { return solid; }
	
  void SetNewValue( G4UIcommand *command, G4String newValues );
  G4String GetCurrentValue( G4UIcommand *command );
	
protected:
	void DeleteArgArray( std::vector<G4UIcmdParg *> &args);
  G4String ConvertArgsToString(std::vector<G4UIcmdParg *> &args);
	
  G4VSolid	*solid;
	
  G4UIdirectory	*volumeDirectory;

  G4String prefix;
  std::vector<G4SBTSolid *> commands;

  G4UIcmdWithAString		*fileCmd;

  G4SBTSolid box, ubox, umultiunion,  uorb, utrd, cons, orb, para, sphere, torus, 
trap, trd, generics, paraboloid, tubs, ellipsoid, elCone, elTube,
extruded, hype, polycone, polycone2, polyhedra, polyhedra2, tessel,
tessel2, tet, twistedBox, twistedTrap, twistedTrap2, twistedTrd, twistedTubs,
  dircTest, booleanSolid1, multiUnion, tessellatedSolidTransform, tessellatedSolidPlain, tessellatedSolidGdml;

  void MakeMeBox( G4String values );
  void MakeMeUBox( G4String values );
  void MakeMeUMultiUnion( G4String values );
  void MakeMeUOrb( G4String values );
  void MakeMeUTrd( G4String values );
  void MakeMeCons( G4String values );
  void MakeMeOrb( G4String values );
  void MakeMePara( G4String values );
  void MakeMeSphere( G4String values );	
  void MakeMeTorus( G4String values );
  void MakeMeTrap( G4String values );
  void MakeMeTrd( G4String values );
  void MakeMeGenericTrap( G4String values );
  void MakeMeParaboloid( G4String values );
  void MakeMeTubs( G4String values );
  void MakeMeEllipsoid( G4String values );
  void MakeMeEllipticalCone( G4String values );
  void MakeMeEllipticalTube( G4String values );
  void MakeMeExtrudedSolid( G4String values );
  void MakeMeHype( G4String values );
  void MakeMePolycone( G4String values );
  void MakeMePolycone2( G4String values );
  void MakeMePolyhedra( G4String values );
  void MakeMePolyhedra2( G4String values );
  void MakeMeTessellatedSolid( G4String values );
  void MakeMeTessellatedSolid2( G4String values );
  void MakeMeTet( G4String values );
  void MakeMeTwistedBox( G4String values );
  void MakeMeTwistedTrap( G4String values );
  void MakeMeTwistedTrap2( G4String values );
  void MakeMeTwistedTrd( G4String values );
  void MakeMeTwistedTubs( G4String values );
  void MakeMeDircTest(G4String values);
  void MakeMeTessellatedSolidFromTransform(G4String values);
  void MakeMeMultiUnion(G4String values);
  void MakeMeBooleanSolid1(G4String values);
  void MakeMeTessellatedSolidFromPlainFile(G4String values);
  void MakeMeTessellatedSolidFromGDMLFile(G4String values);
  
  void TraverseGDML(G4LogicalVolume *root);

  void CreateTessellatedSolid(const std::vector<UVector3> &vertices, const std::vector<std::vector<int> > &nodes);

  typedef enum BooleanOp {
    INTERSECTION,
    SUBTRACTION,
    UNION
  } BooleanOp;

  /* Here add new commands and functions to create solids */
};


#endif
