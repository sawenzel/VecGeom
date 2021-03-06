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
// $Id: AXPETEventAction.cc,v 1.1 2008-09-03 13:34:03 gcosmo Exp $
// ------------------------------------------------------------
// Geant4 class implementation file
//
// 03/09/2008, by T.Nikitina
// ------------------------------------------------------------

#include "AXPETEventAction.hh"

#include "AXPETRunAction.hh"

#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Trajectory.hh"
#include "G4VVisManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"
#include "Randomize.hh"

AXPETEventAction::AXPETEventAction(AXPETRunAction* RA)
 :runaction(RA), verboselevel(1),
   drawFlag("all"), printModulo(10000)
{
  ;
}

AXPETEventAction::~AXPETEventAction()
{
  ;
}

void AXPETEventAction::BeginOfEventAction(const G4Event* evt)
{
  if(verboselevel>1){
  G4int evtNb = evt->GetEventID();

    if (evtNb%printModulo == 0) {
    G4cout << "\n---> Begin of Event: " << evtNb << G4endl;

    G4cout << "<<< Event  " << evtNb << " started." << G4endl;
    }    
  }
}

void AXPETEventAction::EndOfEventAction(const G4Event* evt)
{  
  

  if(verboselevel>0)
    G4cout << "<<< Event  " << evt->GetEventID() << " ended." << G4endl;
 
}

G4int AXPETEventAction::GetEventno()
{
  G4int evno = fpEventManager->GetConstCurrentEvent()->GetEventID() ;
  return evno ;
}

void AXPETEventAction::setEventVerbose(G4int level)
{
  verboselevel = level ;
}
