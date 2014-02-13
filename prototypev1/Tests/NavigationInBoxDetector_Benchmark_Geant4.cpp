/*
 * BuildBoxDetector.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: swenzel
 */

#include <iostream>
#include "mm_malloc.h"
#include <map>
#include <cassert>

#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4GeometryManager.hh"
#include "../Vector3D.h"
#include "G4Navigator.hh"
#include "../GlobalDefs.h"
#include "../Utils.h"
#include "../PhysicalVolume.h"

G4VPhysicalVolume const * BuildGeometry()
{
	double L=10.;
	double Lz=10.;
	const double Sqrt2 = sqrt(2.);

	G4Box * boxlevel2 = new G4Box("b2", Sqrt2*L/2./2., Sqrt2*L/2./2.,Lz );
	// make a logical volume
	G4LogicalVolume * bl2lv = new G4LogicalVolume(boxlevel2,0,"box2",0,0,0);

	G4Box * boxlevel1 = new G4Box("b1", L/2.,L/2.,Lz);
	G4LogicalVolume * bl1lv = new G4LogicalVolume(boxlevel1,0,"box1",0,0,0);

	G4Box * boxlevel3 = new G4Box("b3",  L/2./2.,L/2./2., Lz);
	G4LogicalVolume * bl3lv = new G4LogicalVolume(boxlevel3,0,"box3",0,0,0);

	G4Box * worldb = new G4Box("w",L,L,Lz);
	G4LogicalVolume * worldlv =new G4LogicalVolume(worldb, 0, "world", 0,0,0);
	G4PVPlacement * worldpv =new G4PVPlacement(0,G4ThreeVector(0,0,0),"world", worldlv, 0,false, 0,0);

	// now do a placement of box3 inside box2
	G4RotationMatrix *rot1=new G4RotationMatrix();
	rot1->rotateZ(-pi/4.);
	G4PVPlacement * box3pl=new G4PVPlacement(
				rot1, /* rotation */
				G4ThreeVector(0,0,0), /* translation */
				bl3lv, /* current logical */
				"box3pl",
				bl2lv, /* this is where it is placed */
				0,0);


	// now do a placement of box2 inside logical box1
	G4RotationMatrix *rot2=new G4RotationMatrix();
	rot2->rotateZ(pi/4.);
	G4PVPlacement * box2pl=new G4PVPlacement(
				rot2, /* rotation */
				G4ThreeVector(0,0,0), /* translation */
				bl2lv, /* current logical */
				"box2pl",
				bl1lv, /* this is where it is placed */
				0,0);

	// now do a placement of box1 inside world
	G4PVPlacement * box1pl_1=new G4PVPlacement(
					0, /* no rotation */
					G4ThreeVector(-L/2.,0,0), /* translation */
					bl1lv, /* current logical */
					"box1pl1",
					worldlv, /* this is where it is placed */
					0,0);

	// now do a placement of box1 inside world
	G4PVPlacement * box1pl_2=new G4PVPlacement(
			0, /* no rotation */
			G4ThreeVector(L/2.,0,0), /* translation */
			bl1lv, /* current logical */
			"box1pl2",
			worldlv, /* this is where it is placed */
			0,0);

	return worldpv;
}


int main(int argc, char * argv[])
{
	double L=10.;
		double Lz=10.;

	G4VPhysicalVolume const *top = BuildGeometry();
	G4GeometryManager::GetInstance()->CloseGeometry(true);

	G4Navigator * nav = new G4Navigator();
	nav->SetWorldVolume( const_cast<G4VPhysicalVolume *>(top) );

    StopWatch timer;
	timer.Start();
	int stepsdone=0;
	for(int i=0;i<100000;i++)
	// testing the NavigationAndStepInterface
	{
		int localstepsdone=0;
		double distancetravelled=0.;
		Vector3D p;
		PhysicalVolume::samplePoint( p, L, L, Lz, 1. );

		//std::cerr << p << std::endl;
		//setup point in world
		G4ThreeVector d(1,0,0);
		G4ThreeVector g4p(p.x,p.y,p.z);


		G4VPhysicalVolume *vol;
		vol = nav->LocateGlobalPointAndSetup( g4p, &d, false); // false because we search from top

		while( vol!=NULL )
		{
//			std::cerr << g4p << std::endl;
			localstepsdone++;

			double s,x;
			// do one step ( this will internally adjust the current point and so on )
			// also calculates safety
			double step = nav->ComputeStep( g4p, d, 1e30, s );

			// calculate next point ( do transportation ) and volume ( should go across boundary )
			G4ThreeVector next = g4p + (step + Utils::frHalfTolerance) * d;
			nav->SetGeometricallyLimitedStep();

			vol = nav->LocateGlobalPointAndSetup( next, &d, true);
			if( vol!=0) g4p=next;
		}
	//	std::cerr << localstepsdone << " " << distancetravelled << std::endl;
		stepsdone+=localstepsdone;
	}
	timer.Stop();
	std::cout << " time for 100000 particles " << timer.getDeltaSecs( ) << std::endl;
	std::cout << " average steps done " << stepsdone / 100000. << std::endl;
	std::cout << " time per step " << timer.getDeltaSecs()/stepsdone << std::endl;

	G4GeometryManager::GetInstance()->OpenGeometry();
}


