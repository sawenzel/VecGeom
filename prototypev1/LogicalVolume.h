/*
 * LogicalVolume.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef LOGICALVOLUME_H_
#define LOGICALVOLUME_H_

// abstract base class for shape parameters
// this is used to define a logical volume
class ShapeParameters
{
public:
   virtual void inspect() const = 0;

   virtual ~ShapeParameters(){};
};


class Medium;

class LogicalVolume
{
private:
   ShapeParameters const * shapeparam;
   Medium          const * med;
};


#endif /* LOGICALVOLUME_H_ */
