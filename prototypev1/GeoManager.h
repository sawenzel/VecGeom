/*
 * GeoManager.h
 *
 *  Created on: Nov 11, 2013
 *      Author: swenzel
 */

#ifndef GEOMANAGER_H_
#define GEOMANAGER_H_


#include "TransformationMatrix.h"
#include "ShapeFactories.h"
#include <list>
#include "PhysicalPolycone.h"


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

struct Polycone {};
template <>
struct
ShapeParametersMap<Polycone>
{
   typedef PolyconeParameters<> type;
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


template <>
struct
ShapeToFactoryMap<Polycone>
{
   typedef PolyconeFactory type;
};


// is a singleton class
class GeoManager
{
//private:
   //PhysicalVolume const *top;

public:
   // a couple of useful static matrix instances
//   static TransformationMatrix const * IdentityTransformationMatrix;

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

   // static factory methods ( they will be implemented in separate source files to parallize the compilation process )
   static
   PhysicalVolume * MakePlacedBox( BoxParameters const * bp, TransformationMatrix const * tm, bool=true);

   static
   PhysicalVolume * MakePlacedTube( TubeParameters<> const * tp, TransformationMatrix const * tm, bool=true);

   static
   PhysicalVolume * MakePlacedCone( ConeParameters<> const * cp, TransformationMatrix const * tm, bool=true);

   static
   PhysicalVolume * MakePlacedPolycone( PolyconeParameters<> const * cp, TransformationMatrix const * tm, bool=true);


   // we need to solve problem of pointer ownership --> let's use smart pointers and move semantics

   // this would be general ( could give in an a reference to a static template function )
   template <typename Shape, typename Parameter = ShapeParametersMap<Shape> >
   static
   PhysicalVolume * MakePlacedShape( Parameter const * param, TransformationMatrix const * tm, bool specialize_placement=true)
   {
      // get footprint of TransformationMatrix
      int rid = tm->getRotationFootprint();
      int tid = tm->GetTranslationIdType();

      typedef typename ShapeToFactoryMap<Shape>::type ShapeFactory;

      // the following piece of code is script generated

#ifndef AVOIDSPECIALIZATION
      if( specialize_placement ){
         if( tid == 0 && rid == 1296 ) return ShapeFactory::template Create<0,1296>( param,tm ); // identity
         if( tid == 1 && rid == 1296 ) return ShapeFactory::template Create<1,1296>( param,tm ); // identity
         if( tid == 0 && rid == 252 ) return  ShapeFactory::template Create<0,252>( param,tm );
         if( tid == 1 && rid == 252 ) return  ShapeFactory::template Create<1,252>( param,tm );
         if( tid == 0 && rid == 405 ) return  ShapeFactory::template Create<0,405>( param,tm );
         if( tid == 1 && rid == 405 ) return  ShapeFactory::template Create<1,405>( param,tm );
         if( tid == 0 && rid == 882 ) return  ShapeFactory::template Create<0,882>( param,tm );
         if( tid == 1 && rid == 882 ) return  ShapeFactory::template Create<1,882>( param,tm );
         if( tid == 0 && rid == 415 ) return  ShapeFactory::template Create<0,415>( param,tm );
         if( tid == 1 && rid == 415 ) return  ShapeFactory::template Create<1,415>( param,tm );
         if( tid == 0 && rid == 496 ) return  ShapeFactory::template Create<0,496>( param,tm );
         if( tid == 1 && rid == 496 ) return  ShapeFactory::template Create<1,496>( param,tm );
         if( tid == 0 && rid == 793 ) return  ShapeFactory::template Create<0,793>( param,tm );
         if( tid == 1 && rid == 793 ) return  ShapeFactory::template Create<1,793>( param,tm );
         if( tid == 0 && rid == 638 ) return  ShapeFactory::template Create<0,638>( param,tm );
         if( tid == 1 && rid == 638 ) return  ShapeFactory::template Create<1,638>( param,tm );
         if( tid == 0 && rid == 611 ) return  ShapeFactory::template Create<0,611>( param,tm );
         if( tid == 1 && rid == 611 ) return  ShapeFactory::template Create<1,611>( param,tm );
         if( tid == 0 && rid == 692 ) return  ShapeFactory::template Create<0,692>( param,tm );
         if( tid == 1 && rid == 692 ) return  ShapeFactory::template Create<1,692>( param,tm );
         if( tid == 0 && rid == 720 ) return  ShapeFactory::template Create<0,720>( param,tm );
         if( tid == 1 && rid == 720 ) return  ShapeFactory::template Create<1,720>( param,tm );
         if( tid == 0 && rid == 828 ) return  ShapeFactory::template Create<0,828>( param,tm );
         if( tid == 1 && rid == 828 ) return  ShapeFactory::template Create<1,828>( param,tm );
         if( tid == 0 && rid == 756 ) return  ShapeFactory::template Create<0,756>( param,tm );
         if( tid == 1 && rid == 756 ) return  ShapeFactory::template Create<1,756>( param,tm );
         if( tid == 0 && rid == 918 ) return  ShapeFactory::template Create<0,918>( param,tm );
         if( tid == 1 && rid == 918 ) return  ShapeFactory::template Create<1,918>( param,tm );
         if( tid == 0 && rid == 954 ) return  ShapeFactory::template Create<0,954>( param,tm );
         if( tid == 1 && rid == 954 ) return  ShapeFactory::template Create<1,954>( param,tm );
         if( tid == 0 && rid == 1008 ) return ShapeFactory::template Create<0,1008>( param,tm );
         if( tid == 1 && rid == 1008 ) return ShapeFactory::template Create<1,1008>( param,tm );
      }
#endif
      // fallback case: place shape with unspecialized translation and rotation matrix
      return ShapeFactory::template Create<1,-1>(param,tm);
   }


};




#endif /* GEOMANAGER_H_ */
