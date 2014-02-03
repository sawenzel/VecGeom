/*
 * BuildBoxDetector.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: swenzel
 */


int main()
{
	StopWatch timer;

    // generate benchmark cases
	TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

	// the world volume is a tube
	double worlddx = 100.;
	double worlddy = 100;
	double worlddz = 10.;
	PhysicalVolume * world = GeoManager::MakePlacedBox( new BoxParameters(worlddx, worlddy, worlddz), identity );

	BoxParameters * cellparams = new BoxParameters( worlddx/20., worlddy/20., worlddz/4);


	BoxParameters * waiverparams = new BoxParameters( worlddx/5., worlddy/5., worlddz/2);
	PhysicalVolume *waiver = GeoManager::MakePlacedBox( waiverparams, identity);
	waiver->AddDaughter( GeoManager::MakePlacedBox(cellparams, new TransformationMatrix() ) );
	waiver->AddDaughter( GeoManager::MakePlacedBox(cellparams, new TransformationMatrix() ) );
	waiver->AddDaughter( GeoManager::MakePlacedBox(cellparams, new TransformationMatrix() ) );
	waiver->AddDaughter( GeoManager::MakePlacedBox(cellparams, new TransformationMatrix() ) );

	PhysicalVolume * shield = GeoManager::MakePlacedTube( new TubeParameters<>(9*worldrmax/11, 9*worldrmax/10, 8*worldz/10), identity );
	world->AddDaughter( shield );

	ConeParameters<double> * endcapparams = new ConeParameters<double>( worldrmax/20., worldrmax,
					worldrmax/20., worldrmax/10., worldz/10., 0, 2.*M_PI );
	PhysicalVolume * endcap1 = GeoManager::MakePlacedCone( endcapparams, new TransformationMatrix(0,0,-9.*worldz/10., 0, 0, 0) );
	PhysicalVolume * endcap2 = GeoManager::MakePlacedCone( endcapparams, new TransformationMatrix(0,0,9*worldz/10, 0, 180, 0) );
	world->AddDaughter( endcap1 );
	world->AddDaughter( endcap2 );

	world->fillWithRandomPoints(points,np);
	world->fillWithBiasedDirections(points, dirs, np, 9/10.);

	points.toPlainArray(plainpointarray,np);
	dirs.toPlainArray(plaindirtarray,np);

	std::cerr << " Number of daughters " << world->GetNumberOfDaughters() << std::endl;

	// time performance for this placement ( we should probably include some random physical steps )

	// do some navigation with a simple Navigator
	SimpleVecNavigator vecnav(np);
	PhysicalVolume ** nextvolumes  = ( PhysicalVolume ** ) _mm_malloc(sizeof(PhysicalVolume *)*np, ALIGNMENT_BOUNDARY);

	timer.Start();
	for(int reps=0 ;reps < NREPS; reps++ )
	{
		vecnav.DistToNextBoundary( world, points, dirs, steps, distances, nextvolumes , np );
	}
	timer.Stop();
	double t0 = timer.getDeltaSecs();
	std::cerr << t0 << std::endl;
	// give out hit pointers
	double d0=0.;
	for(auto k=0;k<np;k++)
	{
		d0+=distances[k];
		distances[k]=Utils::kInfinity;
	}


	timer.Start();
	for(int reps=0 ;reps < NREPS; reps++ )
	{
		vecnav.DistToNextBoundaryUsingUnplacedVolumes( world, points, dirs, steps, distances, nextvolumes , np );
	}
	timer.Stop();
	double t1= timer.getDeltaSecs();

	std::cerr << t1 << std::endl;

	double d1;
	for(auto k=0;k<np;k++)
	{
		d1+=distances[k];
		distances[k]=Utils::kInfinity;

	}

	// now using the ROOT Geometry library (scalar version)
	timer.Start();
	for(int reps=0;reps < NREPS; reps ++ )
	{
		vecnav.DistToNextBoundaryUsingROOT( world, plainpointarray, plaindirtarray, steps, distances, nextvolumes, np );
	}
	timer.Stop();
	double t3 = timer.getDeltaSecs();

	std::cerr << t3 << std::endl;
	double d3;
	for(auto k=0;k<np;k++)
		{
			d3+=distances[k];
		}
	std::cerr << d0 << " " << d1 << " " << d3 << std::endl;


	//vecnav.DistToNextBoundaryUsingUnplacedVolumes( world, points, dirs, steps, distances, nextvolumes , np );
	//( world, points, dirs,  );


	// give out hit pointers
	/*
	for(auto k=0;k<np;k++)
	{
		if( nextvolumes[k] !=0 )
		{
			nextvolumes[k]->printInfo();
		}
		else
		{
			std::cerr << "hitting boundary of world"  << std::endl;
		}
	}
*/
    _mm_free(distances);
    return 1;
}


