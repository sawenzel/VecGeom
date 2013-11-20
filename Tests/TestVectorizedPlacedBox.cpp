/*
 * TestVectorizedPlacedBox.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */

// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#//include "TGeoMatrix.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"

const std::vector<std::vector<double>> TransCases {{0,0,0},
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

static void cmpresults(double * a1, double * a2, int np)
{
	int counter=0;
	for(auto i=0;i<np;++i)
	{
		if( a1[i] != a2[i] ) counter++;
	}
	std::cerr << " have " << counter << " differences " << std::endl;
}


int main()
{
	Vectors3DSOA points, dirs, intermediatepoints, intermediatedirs;
	int np=1024;
	points.alloc(np);
	dirs.alloc(np);
	intermediatepoints.alloc(np);
	intermediatedirs.alloc(np);
	double *distances = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
	double *distances2 = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);

	std::vector<Vector3D> conventionalpoints(np);
	std::vector<Vector3D> conventionaldirs(np);

	StopWatch timer;

    // generate benchmark cases
    for( int r=0; r< EulerAngles.size(); ++r ) // rotation cases
    		for( int t=0; t<TransCases.size(); ++t ) // translation cases
    		  {
    			TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);
    			PhysicalVolume * world = GeoManager::MakePlacedBox( new BoxParameters(100,100,100), identity );

    			TransformationMatrix * tm = new TransformationMatrix(TransCases[t][0], TransCases[t][1], TransCases[t][2],
						EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]);

    			// these dispatch to specialized matrices
    			TransformationMatrix const * sm = TransformationMatrix::createSpecializedMatrix( TransCases[t][0], TransCases[t][1], TransCases[t][2],
    											EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2] );


    			PhysicalVolume * daughter = GeoManager::MakePlacedBox( new BoxParameters(10,15,20), tm );

    			world->AddDaughter(daughter);

    			world->fillWithRandomPoints(points,np);
    			world->fillWithBiasedDirections(points, dirs, np, 1./3);

    			points.toStructureOfVector3D( conventionalpoints );
    			dirs.toStructureOfVector3D( conventionaldirs );

    			// time performance for this placement ( we should probably include some random physical steps )
    			timer.Start();
    			for(int r=0;r<1000;r++)
    			{
    				daughter->DistanceToIn(points,dirs,1E30,distances);
    			}
    			timer.Stop();
    			double t0 = timer.getDeltaSecs();

    			// std::cerr << tm->GetTranslationIdType() << " " << tm->getNumberOfZeroEntries() << " " << timer.getDeltaSecs() << std::endl;

    			// compare with case that uses external unspecialized transformation
    			PhysicalVolume * unplaceddaughter = GeoManager::MakePlacedBox(new BoxParameters(10,15,20), identity);
    			timer.Start();
    			for(int r=0;r<1000;r++)
    			{
    				if(! tm->isIdentity() )
    				{
    					tm->MasterToLocal(points, intermediatepoints );
    					tm->MasterToLocalVec( dirs, intermediatedirs );
    					unplaceddaughter->DistanceToIn( intermediatepoints, intermediatedirs, 1E30, distances2);
    				}
    				else
    				{
    					unplaceddaughter->DistanceToIn( points, dirs, 1E30, distances2);
    				}
    			}
    			timer.Stop();
    			double t1 = timer.getDeltaSecs();


    		    // compare with external specialized transformation ( sm )
    			sm->print();
    			timer.Start();
    			for(int r=0;r<1000;r++)
    			{
    				sm->MasterToLocal(points, intermediatepoints );
    			    sm->MasterToLocalVec( dirs, intermediatedirs );
    			    unplaceddaughter->DistanceToIn( intermediatepoints, intermediatedirs, 1E30, distances2);
    			}
    			timer.Stop();
    			double t2 = timer.getDeltaSecs();

    			std::cerr << "VECTOR " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t0 <<  " " << t1 << " " << t2 << std::endl;

    			cmpresults( distances, distances2, np );


    			// now we do the scalar interface: first of all placed version
    			timer.Start();
    			for(int r=0;r<1000;r++)
    			{
    				for(auto j=0;j<np;++j)
    				{
    					distances[j]=daughter->DistanceToIn( conventionalpoints[j], conventionaldirs[j], 1E30);
    				}
    			}
    			timer.Stop();
    			double t3 = timer.getDeltaSecs();

    			// now unplaced version
    			timer.Start();
    		    for(int r=0;r<1000;r++)
    		    {
    		    	for(auto j=0;j<np;++j)
    		    		{
    		    			Vector3D localp, localdir;
    		    			tm->MasterToLocal(conventionalpoints[j], localp);
    		    			tm->MasterToLocalVec(conventionaldirs[j], localdir);
    		    			distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, 1E30);
    		    		}
    		    }
    		    timer.Stop();
    		    double t4 = timer.getDeltaSecs();

    		    // now unplaced version
    		    timer.Start();
    		    for(int r=0;r<1000;r++)
    		    {
    		       	for(auto j=0;j<np;++j)
    		      		{
    		       			Vector3D localp, localdir;
    		       			sm->MasterToLocal(conventionalpoints[j], localp);
    		       			sm->MasterToLocalVec(conventionaldirs[j], localdir);
    		       			distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, 1E30);
    		      		}
    		     }
    		     timer.Stop();
    		     double t5 = timer.getDeltaSecs();

    		     // now unplaced version but inlined matrices
    		     timer.Start();
    		     for(int r=0;r<1000;r++)
    		     {
    		         for(auto j=0;j<np;++j)
    		            {
    		        	     Vector3D localp, localdir;
    		           		 // this inlines I think
    		           		 tm->MasterToLocal<-1,-1>(conventionalpoints[j], localp);
    		           		 tm->MasterToLocalVec<-1>(conventionaldirs[j], localdir);
    		           		 distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, 1E30);
    		            }
    		     }
    		     timer.Stop();
    		     double t6 = timer.getDeltaSecs();

    		     std::cerr << "SCALAR " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t3 <<  " " << t4 << " " << t5 << " " << t6 << std::endl;

    		     TGeoMatrix * rootmatrix= new TGeoCombiTrans(TransCases[t][0], TransCases[t][1], TransCases[t][2],
						   new TGeoRotation("rot1",EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]));
    		     TGeoManager *geom = new TGeoManager("","");
    		     TGeoVolume * vol = geom->MakeBox("abox",0,10,15,20);
    		     TGeoShape *  rootbox=vol->GetShape();

    		     // now the scalar version from ROOTGeantV
    		     timer.Start();
    		     for(int r=0;r<1000;r++)
    		     {
    		    	 for(auto j=0;j<np;++j)
    		    	 {
    		    		 Vector3D localp, localdir;
    		    		 // this inlines I think
    		    		 rootmatrix->MasterToLocal( &conventionalpoints[j], &localp);
    		        	 rootmatrix->MasterToLocalVect( &conventionaldirs[j], &localdir);
    		             distances[j]=box->DistFromOutside( localp, localdir, 3, 0, 1e30);
    		         }
    		     }
    		     timer.Stop();
    		     double t7 = timer.getDeltaSecs();

    		     // now the VECTOR version from ROOT
    		     // now the scalar version from ROOTGeantV
    		     timer.Start();
    		     for(int r=0;r<1000;r++)
    		     {
    		    	 rootmatrix->MasterToLocalCombined_v( points, intermediatepoints, dirs, intermediatedirs );
    		         box->DistFromOutside_v( intermediatepoints, intermediatedirs, 1e30, distances2, np);
    		     }
    		     timer.Stop();
    		     double t7 = timer.getDeltaSecs();

    		    delete tm;
    			delete sm;
    		  }

    _mm_free(distances);
    return 1;
}
