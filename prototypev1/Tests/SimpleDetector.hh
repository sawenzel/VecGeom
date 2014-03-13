/*
 * File: SimpleDetector.hh
 * Purpose: A simple collider detector for testing the VecGeom package.  
 *
 * Change Log:
 *  140305 G.Lima - Created, using Geant4 example ExN04 as starting point.
 */

#ifndef __SimpleDetector_HH__
#define __SimpleDetector_HH__

class SimpleDetector {

public:
  SimpleDetector();
  ~SimpleDetector() {}

  PhysicalVolume const* getPhysicalVolume() const {
    return _world;
  };

private:
  PhysicalVolume* _world;
};

#include "Tests/SimpleDetector.icc"

#endif // __SimpleDetector_HH__
