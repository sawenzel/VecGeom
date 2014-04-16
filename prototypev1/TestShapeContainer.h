/*
 * TestShapeContainer.h
 *
 *  Created on: Dec 12, 2013
 *      Author: swenzel
 */

// a class which creates a couple of test instances for various shapes and methods to access those shapes

#ifndef TESTSHAPECONTAINER_H_
#define TESTSHAPECONTAINER_H_


#include <vector>
#include <cmath>
#include "PhysicalCone.h"
#include "PhysicalTube.h"

class TestShapeContainer
{
private:
   // a simple vector storing various cones as sets of parameters
   const std::vector<std::vector<double> > coneparams { /* rmin1, rmin2, rmax1, rmax2, dz, phistart, deltaphi */
                                 { 0., 0., 10., 15., 30., 0, 2*M_PI}, // a full cone without inner radius
                                 { 0., 0., 10., 15., 30., 0, M_PI/2.}, // a half cone without inner radius
                                 { 8., 8., 10., 15., 30., 0, 3.*M_PI/2.}, //
                                 { 0., 0., 10., 15., 30., 0, M_PI},  // a half cone without inner radius
                                 { 8., 8., 10., 15., 30., 0, 2*M_PI}, //
                                 { 8., 8., 10., 15., 30., 0, M_PI}, //
                                 { 8., 0., 10., 0., 30., 0, M_PI}, //
                                 { 8., 7., 10., 15., 30., 0, 3.21*M_PI/2.} // some weird angle phi
                               };

   const std::vector<std::vector<double> > tubeparams { /* rmin, rmax, dz, phistart, deltaphi */
                                    { 0., 15., 30., 0, 2.*M_PI}, // a full tube without inner radius
                                    { 10., 15., 30., 0, M_PI/2.}, // with inner radius and phi < PI section
                                    { 10., 15., 30., 0, M_PI}, // with inner radius no phi = PI section
                                    { 0., 15., 30., 0, M_PI}, // no inner radius no phi = PI section
                                    { 10., 15., 30., 0, 3*M_PI/2.}, // with inner tube and phi > PI
                                    { 10., 15., 30., 0, 3.21*M_PI/2.}, // with inner tube and weird angle
                              };


public:
   int GetNumberOfConeTestShapes() const
   {
      return coneparams.size();
   }

   ConeParameters<double> const * GetConeParams(int i) const
   {
      return new ConeParameters<double>( coneparams[i][0],
                                 coneparams[i][2],
                                 coneparams[i][1],
                                 coneparams[i][3],
                                 coneparams[i][4],
                                 coneparams[i][5],
                                 coneparams[i][6] );
   }


   int GetNumberOfTubeTestShapes()
   {
      return tubeparams.size();
   }


   TubeParameters<double> const * GetTubeParams(int i) const
   {
      return new TubeParameters<double>( tubeparams[i][0],
                                 tubeparams[i][1],
                                 tubeparams[i][2],
                                 tubeparams[i][3],
                                 tubeparams[i][4] );
   }
};

extern TestShapeContainer gTestShapeContainer;

#endif /* TESTSHAPECONTAINER_H_ */
