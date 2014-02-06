/*
 * BuildBoxDetector.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: swenzel
 */

#include "../TransformationMatrix.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../PhysicalBox.h"
#include "../SimpleVecNavigator.h"
#include <map>
#include <cassert>

int main()
{
	Vectors3DSOA points, dirs, intermediatepoints, intermediatedirs;
	StructOfCoord rpoints, rintermediatepoints, rdirs, rintermediatedirs;


	int np=1024;
	int NREPS = 1000;

	points.alloc(np);
	dirs.alloc(np);

    // generate benchmark cases
	TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

	// the world volume is a tube
	double worlddx = 100.;
	double worlddy = 100;
	double worlddz = 10.;
	BoxParameters *	worldp =  new BoxParameters(worlddx, worlddy, worlddz);
	PhysicalVolume * world = GeoManager::MakePlacedBox( worldp , identity );
	double volworld = worldp->GetVolume();


	BoxParameters * cellparams = new BoxParameters( worlddx/20., worlddy/20., worlddz/4 );
	BoxParameters * waiverparams = new BoxParameters( worlddx/3., worlddy/3., worlddz/2 );
	double volcell = cellparams->GetVolume();
	double volwaiver = waiverparams->GetVolume();

	PhysicalVolume *waiver = GeoManager::MakePlacedBox( waiverparams, identity);

	// this just adds daughters which have been created in a placed way
	waiver->AddDaughter(GeoManager::MakePlacedBox( cellparams, new TransformationMatrix( -waiverparams->GetDX()/2., waiverparams->GetDY()/2., 0, 0, 0, 0) ));
	waiver->AddDaughter(GeoManager::MakePlacedBox( cellparams, new TransformationMatrix( waiverparams->GetDX()/2., waiverparams->GetDY()/2., 0, 0, 0, 45) ));
	waiver->AddDaughter(GeoManager::MakePlacedBox( cellparams, new TransformationMatrix( waiverparams->GetDX()/2., -waiverparams->GetDY()/2., 0, 0, 0, 0) ));
	waiver->AddDaughter(GeoManager::MakePlacedBox( cellparams, new TransformationMatrix( -waiverparams->GetDX()/2., -waiverparams->GetDY()/2., 0, 0, 0, 45)));

	// at this moment the waiver is not yet placed into the world; this will be done now with the new interface
	// we are basically replacing the waiver by using its existing parameters and daughterlist
	// TODO: the future interface will hide much of the details here
	world->PlaceDaughter(GeoManager::MakePlacedBox(waiverparams, new TransformationMatrix( worlddx/2., worlddy/2., 0, 0, 0, 45 )), waiver->GetDaughterList());
	world->PlaceDaughter(GeoManager::MakePlacedBox(waiverparams, new TransformationMatrix( -worlddx/2., worlddy/2., 0, 0, 0, 0  )), waiver->GetDaughterList());
	world->PlaceDaughter(GeoManager::MakePlacedBox(waiverparams, new TransformationMatrix( -worlddx/2., -worlddy/2., 0, 0, 0, 45 )), waiver->GetDaughterList());
	world->PlaceDaughter(GeoManager::MakePlacedBox(waiverparams, new TransformationMatrix( worlddx/2., -worlddy/2., 0, 0, 0, 0 )), waiver->GetDaughterList());


	world->fillWithRandomPoints(points,np);
	world->fillWithBiasedDirections(points, dirs, np, 9/10.);

	std::cerr << " Number of daughters " << world->GetNumberOfDaughters() << std::endl;

	// try to locate a global point

	StopWatch timer;
	timer.Start();
	VolumePath path(4), newpath(4);
	std::map<PhysicalVolume const *, int> volcounter;
	int total=0;
	TransformationMatrix * globalm=new TransformationMatrix();
	TransformationMatrix * globalm2 = new TransformationMatrix();
	SimpleVecNavigator nav(1, world);
	Vector3D displacementvector( worlddx/40, 0., 0. );
	int counter[2]={0,0};

	// shoot random positions in detector
	// task: determine last global point in detector before exiting it

	// we use: initial localization of point ( without existing path )
	// start loop
	// 		then displacement of particle locally
	// 		then relocalization until exiting the detector

	// this should be compared with a ROOT Geometry for a couple of ( deterministically chosen points )

	for(int i=0;i<1000000;i++)
	{
		globalm->SetToIdentity();
		globalm2->SetToIdentity();
		Vector3D point;
		Vector3D localpoint;
		Vector3D newlocalpoint;
		Vector3D oldpoint;
		Vector3D finalglobalpoint;
		PhysicalVolume::samplePoint( point, worlddx, worlddy, worlddz, 1 );

		localpoint = point;
		PhysicalVolume const * deepestnode = nav.LocateGlobalPoint( world, point, localpoint, path );

		while(deepestnode != NULL)
		{
			// move particle
			localpoint += displacementvector;

			// determine new location / new path for particle
			deepestnode = nav.LocateLocalPointFromPath_Relative( localpoint, newlocalpoint, path );

			oldpoint = localpoint;
			localpoint = newlocalpoint;

			//	deepestnode->printInfo();
		}
		// now we know last point in detector

		// get global matrix from path but still have to transform it back to reference frame of detector
		path.GetGlobalMatrixFromPath( globalm );
		globalm->LocalToMaster( oldpoint, finalglobalpoint );
		finalglobalpoint.print();
		path.Clear();
	}
	timer.Stop();
	std::cerr << " step took " << timer.getDeltaSecs() << " seconds " << std::endl;
	std::cerr << counter[0] << std::endl;
	std::cerr << counter[1] << std::endl;
	
	for(auto k=volcounter.begin();k!=volcounter.end();k++)
	  {
	    total+=k->second;
	  }
	
	for(auto k=volcounter.begin();k!=volcounter.end();k++)
	  {
	    std::cerr << k->first << " " << k->second << " " << k->second/(1.*total) << std::endl; 
	  }
	std::cerr << 4*volcell/volworld << std::endl;
	std::cerr << volwaiver/volworld << std::endl;
	std::cerr << (volworld-4*volwaiver)/volworld << std::endl;
}

