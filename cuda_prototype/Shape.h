  #ifndef SHAPE_H
#define SHAPE_H

#include "LibraryGeneric.h"
#include <list>

class Box;

class Shape {

private:

  std::list<Shape const*> daughters;

protected:

  TransMatrix<double> const *trans_matrix;
  Box const *bounding_box;

public:

  #ifdef STD_CXX11
  Shape(TransMatrix<double> const * const trans, Box const * const bounds)
      : trans_matrix(trans), bounding_box(bounds) {}
  #else
  Shape(TransMatrix<double> const * const trans, Box const * const bounds) {
    trans_matrix = trans;
    bounding_box = bounds;
  }
  #endif

  CUDA_HEADER_BOTH
  inline __attribute__((always_inline))
  TransMatrix<double> const* TransformationMatrix() const {
    return trans_matrix;
  }

  void AddDaughter(Shape const * const daughter) {
    daughters.push_back(daughter);
  }

  static void FillRandomDirections(SOA3D<double>& /*dirs*/)  ;

  void FillUncontainedPoints(SOA3D<double>& /*points*/) const;

  void FillBiasedDirections(SOA3D<double> const& /*points*/,
                            const double /*bias*/,
                            SOA3D<double>& /*dirs*/) const;

  // Contains

  virtual bool Contains(Vector3D<double> const& /*point*/) const =0;

  virtual void Contains(SOA3D<double> const& /*points*/,
                        bool* /*output*/) const =0;

  // DistanceToIn

  virtual double DistanceToIn(Vector3D<double> const& /*point*/,
                              Vector3D<double> const& /*dir*/,
                              double const step_max) const =0;

  virtual void DistanceToIn(SOA3D<double> const& /*pos*/,
                            SOA3D<double> const& /*dir*/,
                            double const *step_max,
                            double* distance) const =0;

private:

  static Vector3D<double> SamplePoint(Vector3D<double> const& /*size*/,
                                      const double scale = 1);

  static Vector3D<double> SampleDirection();

};

#endif /* SHAPE_H */