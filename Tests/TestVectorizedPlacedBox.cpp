/*
 * TestVectorizedPlacedBox.cpp
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
	StructOfCoord rpoints, rintermediatepoints, rdirs, rintermediatedirs;


	int np=1024;

	points.alloc(np);
	dirs.alloc(np);
	intermediatepoints.alloc(np);
	intermediatedirs.alloc(np);

	rpoints.alloc(np);
	rdirs.alloc(np);
	rintermediatepoints.alloc(np);
	rintermediatedirs.alloc(np);

	double *distances = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
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
    				world->fillWithBiasedDirections(points, dirs, np, 1./10);

    				points.toStructureOfVector3D( conventionalpoints );
    				dirs.toStructureOfVector3D( conventionaldirs );
    				points.toStructureOfVector3D( conventionalpoints2 );
    				dirs.toStructureOfVector3D( conventionaldirs2 );


    				// time performance for this placement ( we should probably include some random physical steps )
    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
    				{
    					daughter->DistanceToIn(points,dirs,steps,distances);
    				}
    				timer.Stop();
    				double t0 = timer.getDeltaSecs();

    			// std::cerr << tm->GetTranslationIdType() << " " << tm->getNumberOfZeroEntries() << " " << timer.getDeltaSecs() << std::endl;

    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
    				{
    					daughter->DistanceToInIL(points,dirs,steps,distances);
    				}
    				timer.Stop();
    				double til = timer.getDeltaSecs();

    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
    				{
    					daughter->DistanceToInIL( conventionalpoints2, conventionaldirs2, steps, distances, np );
    				}
    				timer.Stop();
    				double til2 = timer.getDeltaSecs();


    				// compare with case that uses external unspecialized transformation
    				PhysicalVolume * unplaceddaughter = GeoManager::MakePlacedBox(new BoxParameters(10,15,20), identity);
    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
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


    				// compare with external specialized transformation ( sm )
    				sm->print();
    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
    				{
    					sm->MasterToLocal(points, intermediatepoints );
    					sm->MasterToLocalVec( dirs, intermediatedirs );
    					unplaceddaughter->DistanceToIn( intermediatepoints, intermediatedirs, steps, distances2);
    				}
    				timer.Stop();
    				double t2 = timer.getDeltaSecs();

    				std::cerr << "VECTOR " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t0 <<  " " << t1 << " " << t2 << " " << til << " " << til2 << std::endl;

    				cmpresults( distances, distances2, np );


    				// now we do the scalar interface: first of all placed version
    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
    				{
    					for(auto j=0;j<np;++j)
    					{
    						distances[j]=daughter->DistanceToIn( conventionalpoints[j], conventionaldirs[j], steps[j]);
    					}
    				}
    				timer.Stop();
    				double t3 = timer.getDeltaSecs();

    				// now unplaced version
    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
    				{
    					for(auto j=0;j<np;++j)
    		    			{
    		    				Vector3D localp, localdir;
    		    				tm->MasterToLocal(conventionalpoints[j], localp);
    		    				tm->MasterToLocalVec(conventionaldirs[j], localdir);
    		    				distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, steps[j]);
    		    			}
    				}
    				timer.Stop();
    				double t4 = timer.getDeltaSecs();

    				// now unplaced version
    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
    				{
    					for(auto j=0;j<np;++j)
    		      			{
    		       				Vector3D localp, localdir;
    		       				sm->MasterToLocal(conventionalpoints[j], localp);
    		       				sm->MasterToLocalVec(conventionaldirs[j], localdir);
    		       				distances2[j]=unplaceddaughter->DistanceToIn( localp, localdir, steps[j]);
    		      			}
    				}
    				timer.Stop();
    				double t5 = timer.getDeltaSecs();

    				// now unplaced version but inlined matrices
    				timer.Start();
    				for(int reps=0;reps<1000;reps++)
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
    		     for(int reps=0;reps<1000;reps++)
    		     {
    		    	 for(auto j=0;j<np;++j)
    		    	 {
    		    		 Vector3D localp, localdir;
    		    		 // this inlines I think
    		    		 rootmatrix->MasterToLocal( &conventionalpoints[j].x, &localp.x );
    		        	 rootmatrix->MasterToLocalVect( &conventionaldirs[j].x, &localdir.x );
    		             distances[j]=rootbox->DistFromOutside( &localp.x, &localdir.x, 3,1e30, 0);
    		         }
    		     }
    		     timer.Stop();
    		     double t7 = timer.getDeltaSecs();

    		     // now the VECTOR version from ROOT
    		     // now the scalar version from ROOTGeantV
    		     timer.Start();
    		     for(int reps=0;reps<1000;reps++)
    		     {
    		    	 rootmatrix->MasterToLocalCombined_v( reinterpret_cast<StructOfCoord const &>(points), reinterpret_cast<StructOfCoord &>(intermediatepoints),
    		    			     		    			 reinterpret_cast<StructOfCoord const &>(rdirs), reinterpret_cast<StructOfCoord &>(intermediatedirs), np );
    		         rootbox->DistFromOutsideSOA_v( reinterpret_cast<StructOfCoord const &>(intermediatepoints),
    		        		 	 reinterpret_cast<StructOfCoord const &>(intermediatedirs), 3, steps, 0, distances2, np);
    		     }
    		     timer.Stop();
    		     double t8 = timer.getDeltaSecs();
    		     std::cerr << "RSCAL " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t7 << std::endl;
    		     std::cerr << "RVEC " << tm->isTranslation() << " " << tm->isRotation() << "("<<tm->getNumberOfZeroEntries()<<")" << " " << t8 << std::endl;

    		    delete tm;
    			delete sm;
    		  }

    _mm_free(distances);
    return 1;
}
