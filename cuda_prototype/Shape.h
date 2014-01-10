#ifndef SHAPE_H
#define SHAPE_H

#include "LibraryGeneric.h"
#include <list>

class Box;

class Shape {

private:

  std::list<Shape const*> daughters;

protected:

  Box const *bounding_box;
  TransMatrix<double> const *trans_matrix;
  TransMatrix<float> const *trans_matrix_cuda;

public:

  void AddDaughter(Shape const * const daughter) {
    daughters.push_back(daughter);
  }

  static void FillRandomDirections(SOA3D<double>& /*dirs*/);

  void FillUncontainedPoints(SOA3D<double>& /*points*/) const;

  void FillBiasedDirections(SOA3D<double> const& /*points*/,
                            const double /*bias*/,
                            SOA3D<double>& /*dirs*/) const;

  // // Contains

  // virtual bool Contains(Vector3D<double> const& /*point*/) const =0;

  // virtual void Contains(SOA3D<double> const& /*points*/,
  //                       bool* /*output*/) const =0;

  // virtual void Contains_Vc(SOA3D<double> const& /*points*/,
  //                          bool* /*output*/) const =0;

  // virtual void Contains_CUDA(SOA3D<double> const& /*points*/,
  //                            bool* /*output*/) const =0;

  // // DistanceToIn

  // virtual double DistanceToIn(Vector3D<double> const& /*point*/,
  //                             Vector3D<double> const& /*dir*/) const =0;

  // virtual void DistanceToIn(SOA3D<double> const& /*pos*/,
  //                           SOA3D<double> const& /*dir*/,
  //                           double* distance) const =0;

  // virtual void DistanceToIn_Vc(SOA3D<double> const& /*pos*/,
  //                              SOA3D<double> const& /*dir*/,
  //                              double* distance) const =0;

  // virtual void DistanceToIn_CUDA(SOA3D<double> const& /*pos*/,
  //                                SOA3D<double> const& /*dir*/,
  //                                double* distance) const =0;

private:

  static Vector3D<double> SamplePoint(Vector3D<double> const& /*size*/,
                                      const double scale = 1);

  static Vector3D<double> SampleDirection();

};

#endif /* SHAPE_H */