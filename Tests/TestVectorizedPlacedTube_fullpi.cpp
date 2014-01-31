/*
 * TestVectorizedPlacedTube.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */

// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoShape.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../PhysicalTube.h"

// in order to compare to USolids
#include "VUSolid.hh"
#include "UTubs.hh"


const std::vector<std::vector<double> > TransCases {{0,0,0},
    {10, 10, 0}};


// for the rotations ( C++11 feature )
const std::vector<std::vector<double> > EulerAngles  {{0.,0.,0.},
    {30.,0, 0.},
   //  {0, 45., 0.},
    //	{0, 0, 67.5}};
    {180, 0, 0},
	//{30, 48, 0.},
	{78.2, 81.0, -10.}};
//const std::vector<std::vector<double> > EulerAngles {{180,0,0}};

static void cmpresults(double * a1, double * a2, int np,
		PhysicalVolume const * vol, std::vector<Vector3D> const & points, std::vector<Vector3D> const & dirs)
{
	int counter=0;
	for( auto i=0; i<np; ++i )
	{
		if( std::abs( a1[i] - a2[i] ) > Utils::GetCarTolerance() && (a1[i] < 1E30 | a2[i] < 1E30) )
		{
			counter++;
#ifdef SHOWDIFFERENCES
			std::cerr << i << " " << a1[i] << " " << a2[i] << std::endl;
			vol->DebugPointAndDirDistanceToIn( points[i], dirs[i] );
#endif
		}
	}
	std::cerr << " have " << counter << " differences " << std::endl;
}


int main()
{
	Vectors3DSOA points, dirs, intermediatepoints, intermediatedirs;
	StructOfCoord rpoints, rintermediatepoints, rdirs, rintermediatedirs;


	int np=1024;
	int NREPS = 1000;

	points.alloc(np);
	dirs.alloc(np);
	intermediatepoints.alloc(np);
	intermediatedirs.alloc(np);

	rpoints.alloc(np);
	rdirs.alloc(np);
	rintermediatepoints.alloc(np);
	rintermediatedirs.alloc(np);

	double *distances = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distancesROOTSCALAR = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distancesUSOLIDSCALAR = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distances2 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *steps = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	for(auto i=0;i<np;++i) steps[i]=1E30;

	std::vector<Vector3D> conventionalpoints(np);
	std::vector<Vector3D> conventionaldirs(np);
	Vector3D * conventionalpoints2 = (Vector3D *) new Vector3D[np];
	Vector3D * conventionaldirs2 = (Vector3D *) new Vector3D[np];

	StopWatch timer;

    // generate benchmark cases
    for( int r=0; r< EulerAngles.size(); ++r ) // rotation cases
    		for( int t=0; t< TransCases.size(); ++t ) // translation cases
    		  {
    			TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);
    			PhysicalVolume * world = GeoManager::MakePlacedBox( new BoxParameters(100,100,100), identity );

    			TransformationMatrix * tm = new TransformationMatrix(TransCases[t][0], TransCases[t][1], TransCases[t][2],
    					EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]);

    			// these dispatch to specialized matrices
    			TransformationMatrix const * sm = TransformationMatrix::createSpecializedMatrix( TransCases[t][0], TransCases[t][1], TransCases[t][2],
    						EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2] );

    			double rmin = 0.;
    			double rmax = 20.;
    			double dz = 30.;
    			double phis  =0.;
    			//double dphi = 2.*M_PI;
    			double dphi = M_PI;
    			PhysicalVolume * daughter = GeoManager::MakePlacedTube( new TubeParameters<>( rmin, rmax, dz, phis, dphi), tm );

    			//std::cerr << daughter->UnplacedContains( Vector3D(15, 1, 15) ) << std::endl;
    			//std::cerr << daughter->UnplacedContains( Vector3D(-15, 1, 15) ) << std::endl;
    			// testing UnplacedContains
    			//	for(auto k=0;k<100;k++)
    			//	{
    			//		Vector3D x( cos(k/(100.)*2*M_PI), sin(k/(100.)*2*M_PI), 0 );
    			//		std::cerr << "## " << k/100.*2*M_PI << " "  << daughter->UnplacedContains( x ) << std::endl;
    			//	}

    			world->AddDaughter(daughter);

    			world->fillWithRandomPoints(points,np);
    			world->fillWithBiasedDirections(points, dirs, np, 2/10.);

    			points.toStructureOfVector3D( conventionalpoints );
    			dirs.toStructureOfVector3D( conventionaldirs );
    			points.toStructureOfVector3D( conventionalpoints2 );
    			dirs.toStructureOfVector3D( conventionaldirs2 );

//// time performance for this placement ( we should probably include some random physical steps )


    			timer.Start();
    			for(int reps=0;reps< NREPS ;reps++)
    			{
    				daughter->DistanceToIn(points,dirs,steps,distances);
    			}
    			timer.Stop();
    			double t0 = timer.getDeltaSecs();

    				//
//    			// std::cerr << tm->GetTranslationIdType() << " " << tm->getNumberOfZeroEntries() << " " << timer.getDeltaSecs() << std::endl;
//
//    				timer.Start();
//					for(int reps=0;reps<NREPS;reps++)
//    				{
//    					daughter->DistanceToInIL(points,dirs,steps,distances);
//    				}
//    				timer.Stop();
//    				double til = timer.getDeltaSecs();
//
//    				timer.Start();
//    				for(int reps=0;reps<NREPS;reps++)
//    				{
//    					daughter->DistanceToInIL( conventionalpoints2, conventionaldirs2, steps, distances, np );
//    				}
//    				timer.Stop();
//    				double til2 = timer.getDeltaSecs();
//
//
    				// compare with case that uses external unspecialized transformation
    				//0, 20, 30, M_PI

    				PhysicalVolume * unplaceddaughter = GeoManager::MakePlacedTube( new TubeParameters<>( rmin, rmax ,dz, phis, dphi ), identity );
    				timer.Start();
    				for(int reps=0;reps<NREPS;reps++)
    				{
    					if(! tm->isIdentity() )
    					{
    						tm->MasterToLocal(points, intermediatepoints );
    						tm->MasterToLocalVec( dirs, intermediatedirs );
    						unplaceddaughter->DistanceToIn( intermediatepoints, intermediatedirs, steps, distances2);
    					}
    					else
    					{
    						unplaceddaughter->DistanceToIn( points, dirs, steps, distances2);
    					}
    				}
    				timer.Stop();
    				double t1 = timer.getDeltaSecs();

    				//
//
//    				// compare with external specialized transformation ( sm )
//    				sm->print();
    				timer.Start();
    				for(int reps=0;reps<NREPS;reps++)
    				{
    					sm->MasterToLocal(points, intermediatepoints );
    					sm->MasterToLocalVec( dirs, intermediatedirs );
    					unplaceddaughter->DistanceToIn( intermediatepoints, intermediatedirs, steps, distances2);
    				}
    				timer.Stop();
    				double t2 = timer.getDeltaSecs();


//    				std::cerr << "VECTOR " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t0 <<  " " << t1 << " " << t2 << " " << til << " " << til2 << std::endl;
//    				cmpresults( distances, distances2, np );
//
//
//    				// now we do the scalar interface: first of all placed version
//    				timer.Start();
//    				for(int reps=0;reps<NREPS;reps++)
//    				{
//    					for(auto j=0;j<np;++j)
//    					{
//    						distances[j]=daughter->DistanceToIn( conventionalpoints[j], conventionaldirs[j], steps[j]);
//    					}
//    				}
//    				timer.Stop();
//    				double t3 = timer.getDeltaSecs();
//
//    				// now unplaced version
//    				timer.Start();
//    				for(int reps=0;reps<NREPS;reps++)
//    				{
//    					for(auto j=0;j<np;++j)
//    		    			{
//    		    				Vector3D localp, localdir;
//    		    				tm->MasterToLocal(conventionalpoints[j], localp);
//    		    				tm->MasterToLocalVec(conventionaldirs[j], localdir);
//    		    				distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, steps[j]);
//    		    			}
//    				}
//    				timer.Stop();
//    				double t4 = timer.getDeltaSecs();
//
//    				// now unplaced version
//    				timer.Start();
//    				for(int reps=0;reps<NREPS;reps++)
//    				{
//    					for(auto j=0;j<np;++j)
//    		      			{
//    		       				Vector3D localp, localdir;
//    		       				sm->MasterToLocal(conventionalpoints[j], localp);
//    		       				sm->MasterToLocalVec(conventionaldirs[j], localdir);
//    		       				distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, steps[j]);
//    		      			}
//    				}
//    				timer.Stop();
//    				double t5 = timer.getDeltaSecs();
//
//    				// now unplaced version but inlined matrices
//    				timer.Start();
//    				for(int reps=0;reps<NREPS;reps++)
//    				{
//    					for(auto j=0;j<np;++j)
//    					{
//    		        	     Vector3D localp, localdir;
//    		        	     // this inlines I think
//    		           		 tm->MasterToLocal<-1,-1>(conventionalpoints[j], localp);
//    		           		 tm->MasterToLocalVec<-1>(conventionaldirs[j], localdir);
//    		           		 distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, 1E30);
//    		            }
//    				}
//    				timer.Stop();
//    				double t6 = timer.getDeltaSecs();
//
//    				std::cerr << "SCALAR " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t3 <<  " " << t4 << " " << t5 << " " << t6 << std::endl;
//
    			 TGeoMatrix * rootmatrix= new TGeoCombiTrans(TransCases[t][0], TransCases[t][1], TransCases[t][2],
						   new TGeoRotation("rot1",EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]));
    			 TGeoManager *geom = new TGeoManager("","");
    		     TGeoVolume * vol = geom->MakeTubs("atube",0,rmin,rmax,dz, phis *360/(2.*M_PI), phis+360*dphi/(2.*M_PI));
    		     TGeoShape *  roottube=vol->GetShape();

    		     // now the scalar version from ROOTGeantV
    		     timer.Start();
    		     for(int reps=0;reps<NREPS;reps++)
    		     {
    		    	 for(auto j=0;j<np;++j)
    		    	 {
    		    		 Vector3D localp, localdir;
    		    		 rootmatrix->MasterToLocal( &conventionalpoints[j].x, &localp.x );
    		        	 rootmatrix->MasterToLocalVect( &conventionaldirs[j].x, &localdir.x );
    		             distancesROOTSCALAR[j]=roottube->DistFromOutside( &localp.x, &localdir.x, 3,1e30, 0);
    		         }
    		     }
    		     timer.Stop();
    		     double t7 = timer.getDeltaSecs();

    		     // now the VECTOR version from ROOT
    		     // now the scalar version from ROOTGeantV
    		     timer.Start();
    		     for(int reps=0;reps<NREPS;reps++)
    		     {
    		    	 rootmatrix->MasterToLocalCombined_v( reinterpret_cast<StructOfCoord const &>(points), reinterpret_cast<StructOfCoord &>(intermediatepoints),
    		    			     		    			 reinterpret_cast<StructOfCoord const &>(dirs), reinterpret_cast<StructOfCoord &>(intermediatedirs), np );
    		         roottube->DistFromOutsideSOA_v( reinterpret_cast<StructOfCoord const &>(intermediatepoints),
    		        		 	 reinterpret_cast<StructOfCoord const &>(intermediatedirs), 3, steps, 0, distances2, np);
    		     }
    		     timer.Stop();
    		     double t8 = timer.getDeltaSecs();

    		     cmpresults( distancesROOTSCALAR, distances, np, daughter, conventionalpoints, conventionaldirs );

    		     // now we compare with loop over USolids version (scalar; trying to inline matrices as done in Geant4 typically)
    		     VUSolid * utub =  new UTubs("utubs1",rmin,rmax,dz, phis, dphi);
    		     timer.Start();
    		     for(int reps=0;reps<NREPS;reps++)
    		     {
    		        	 for(auto j=0;j<np;++j)
    		     	     {
    		        		 Vector3D localp, localdir;
    		        		  // this inlines I think
    		        		 tm->MasterToLocal<1,-1>( conventionalpoints[j], localp );
    		        		 tm->MasterToLocalVec<-1>( conventionaldirs[j], localdir );
    		        		 distancesUSOLIDSCALAR[j]=utub->DistanceToIn( reinterpret_cast<UVector3 const & > (localp), reinterpret_cast<UVector3 &> ( localdir ), 1e30);
    		     	     }
    		      }
    		     timer.Stop();
    		     double t9 = timer.getDeltaSecs();

    		     std::cerr << "new vec (placed)" << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t0 << std::endl;
    		     std::cerr << "new vec (old matrix)" << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t1 << std::endl;
    		     std::cerr << "new vec (unplaced)" << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t2 << std::endl;
    		     std::cerr << "RSCAL " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t7 << std::endl;
    		     std::cerr << "RVEC " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t8 << std::endl;
    		     std::cerr << "USOLIDS SCAL " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t9 << std::endl;

    		     delete tm;
    		     delete sm;
    		  }
//
    _mm_free(distances);
    return 1;
}
