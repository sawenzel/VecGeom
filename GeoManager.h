/*
 * GeoManager.h
 *
 *  Created on: Nov 11, 2013
 *      Author: swenzel
 */

#ifndef GEOMANAGER_H_
#define GEOMANAGER_H_

#include "PhysicalVolume.h"
#include "PhysicalBox.h"
#include "PhysicalTube.h"
#include "PhysicalCone.h"
#include "TransformationMatrix.h"
#include "ShapeFactories.h"
#include <list>

class PhysicalVolume;
// the GeoManager manages the geometry hierarchy. It keeps a pointer to the top volume
// etc.

template <typename Shape>
struct
ShapeParametersMap
{
	typedef int type;
};


template <int, int>
struct Foo {};


struct Box {};
template <>
struct
ShapeParametersMap<Box>
{
	typedef BoxParameters type;
};

struct Cone {};
template <>
struct
ShapeParametersMap<Cone>
{
	typedef ConeParameters<> type;
};

struct Tube {};
template <>
struct
ShapeParametersMap<Tube>
{
	typedef TubeParameters<> type;
};

template <typename Shape>
struct
ShapeToFactoryMap
{
	typedef NullFactory type;
};

template <>
struct
ShapeToFactoryMap<Cone>
{
	typedef ConeFactory type;
};

template <>
struct
ShapeToFactoryMap<Tube>
{
	typedef TubeFactory type;
};

template <>
struct
ShapeToFactoryMap<Box>
{
	typedef BoxFactory type;
};


// is a singleton class
class GeoManager
{
//private:
	//PhysicalVolume const *top;

public:
	// a couple of useful static matrix instances
//	static TransformationMatrix const * IdentityTransformationMatrix;

	//void Export(){}
	//void Import(){}
	//void ExportToDatabase(){}

	//PhysicalVolume const * GetTop() const {return top;}

	//int GetNumberOfPlacedVolumes(){return 0;}
	//int GetNumberOfLogicalVolumes(){return 0;}

	//
	//void GetListOfLogicalVolumes(){;}


	// it would be nice to have something like an SQL query here
	//

	// static factory methods
	static
	PhysicalVolume * MakePlacedBox( BoxParameters const * bp, TransformationMatrix const * tm)
	{
		return MakePlacedShape<Box>( bp, tm);

	}

	static
	PhysicalVolume * MakePlacedTube( TubeParameters<> const * tp, TransformationMatrix const * tm)
	{
		return MakePlacedShape<Tube> (tp, tm);
	}

	// we need to solve problem of pointer ownership --> let's use smart pointers and move semantics




	// this would be general ( could give in an a reference to a static template function )
	template <typename Shape, typename Parameter = ShapeParametersMap<Shape> >
	static
	PhysicalVolume * MakePlacedShape( Parameter const * param, TransformationMatrix const * tm)
	{
		// get footprint of TransformationMatrix
		int rid = tm->getRotationFootprint();
		int tid = tm->GetTranslationIdType();

		typedef typename ShapeToFactoryMap<Shape>::type ShapeFactory;

		// the following piece of code is script generated
		if( tid == 0 && rid == 1296 ) return ShapeFactory::template Create<0,1296>( param,tm ); // identity
		if( tid == 1 && rid == 1296 ) return ShapeFactory::template Create<1,1296>( param,tm ); // identity

		/*
		if(tid == 0 && rid == 252 ) return  ShapeFactoryFunctor<0,252>::create(param,tm);
		if(tid == 1 && rid == 252 ) return  ShapeFactoryFunctor<1,252>::create(param,tm);
		if(tid == 0 && rid == 405 ) return  ShapeFactoryFunctor<0,405>::create(param,tm);
		if(tid == 1 && rid == 405 ) return  ShapeFactoryFunctor<1,405>::create(param,tm);
		if(tid == 0 && rid == 882 ) return  ShapeFactoryFunctor<0,882>::create(param,tm);
		if(tid == 1 && rid == 882 ) return  ShapeFactoryFunctor<1,882>::create(param,tm);
		if(tid == 0 && rid == 415 ) return  ShapeFactoryFunctor<0,415>::create(param,tm);
		if(tid == 1 && rid == 415 ) return  ShapeFactoryFunctor<1,415>::create(param,tm);
		if(tid == 0 && rid == 496 ) return  ShapeFactoryFunctor<0,496>::create(param,tm);
		if(tid == 1 && rid == 496 ) return  ShapeFactoryFunctor<1,496>::create(param,tm);
		if(tid == 0 && rid == 793 ) return  ShapeFactoryFunctor<0,793>::create(param,tm);
		if(tid == 1 && rid == 793 ) return  ShapeFactoryFunctor<1,793>::create(param,tm);
		if(tid == 0 && rid == 638 ) return  ShapeFactoryFunctor<0,638>::create(param,tm);
		if(tid == 1 && rid == 638 ) return  ShapeFactoryFunctor<1,638>::create(param,tm);
		if(tid == 0 && rid == 611 ) return  ShapeFactoryFunctor<0,611>::create(param,tm);
		if(tid == 1 && rid == 611 ) return  ShapeFactoryFunctor<1,611>::create(param,tm);
		if(tid == 0 && rid == 692 ) return  ShapeFactoryFunctor<0,692>::create(param,tm);
		if(tid == 1 && rid == 692 ) return  ShapeFactoryFunctor<1,692>::create(param,tm);
		if(tid == 0 && rid == 720 ) return  ShapeFactoryFunctor<0,720>::create(param,tm);
		if(tid == 1 && rid == 720 ) return  ShapeFactoryFunctor<1,720>::create(param,tm);
		if(tid == 0 && rid == 828 ) return  ShapeFactoryFunctor<0,828>::create(param,tm);
		if(tid == 1 && rid == 828 ) return  ShapeFactoryFunctor<1,828>::create(param,tm);
		if(tid == 0 && rid == 756 ) return  ShapeFactoryFunctor<0,756>::create(param,tm);
		if(tid == 1 && rid == 756 ) return  ShapeFactoryFunctor<1,756>::create(param,tm);
		if(tid == 0 && rid == 918 ) return  ShapeFactoryFunctor<0,918>::create(param,tm);
		if(tid == 1 && rid == 918 ) return  ShapeFactoryFunctor<1,918>::create(param,tm);
		if(tid == 0 && rid == 954 ) return  ShapeFactoryFunctor<0,954>::create(param,tm);
		if(tid == 1 && rid == 954 ) return  ShapeFactoryFunctor<1,954>::create(param,tm);
		if(tid == 0 && rid == 1008 ) return ShapeFactoryFunctor<0,1008>::create(param,tm);
		if(tid == 1 && rid == 1008 ) return ShapeFactoryFunctor<1,1008>::create(param,tm);
*/
		// fallback case
		return ShapeFactory::template Create<1,-1>(param,tm);
	}

	static
	PhysicalVolume * MakePlacedCone( ConeParameters<> const * cp, TransformationMatrix const * tm)
	{
		return MakePlacedShape<Cone> (cp, tm);
	}

};

// static const GeoManager::IdentityTransformationMatrix=new TransformationMatrix(0,0,0,0,0,0);


#endif /* GEOMANAGER_H_ */
