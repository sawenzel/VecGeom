#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEBOOLEAN_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEBOOLEAN_HH

#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/volumes/Polyhedron.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeomTest/TestCaseCommon.h"

namespace vecgeom {
namespace test {

template <vecgeom::BooleanOperation Op>
inline vecgeom::VPlacedVolume *MakeLeakedBooleanNode(const char *name, vecgeom::VPlacedVolume const *left,
                                                     vecgeom::VPlacedVolume const *right,
                                                     vecgeom::Transformation3D const *transformation = nullptr)
{
  return MakeLeakedStandalonePlacedVolume(name, new vecgeom::UnplacedBooleanVolume<Op>(Op, left, right),
                                          transformation);
}

inline vecgeom::VPlacedVolume *MakeLeakedPolyhedronNode(const char *name, vecgeom::Precision phi_start,
                                                        vecgeom::Precision phi_delta, int side_count, int num_z,
                                                        vecgeom::Precision const z[], vecgeom::Precision const rmin[],
                                                        vecgeom::Precision const rmax[],
                                                        vecgeom::Transformation3D const *transformation = nullptr)
{
  return MakeLeakedStandalonePlacedVolume(
      name, new vecgeom::UnplacedPolyhedron(phi_start, phi_delta, side_count, num_z, z, rmin, rmax), transformation);
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanUnionOffsetBoxesTestSolid()
{
  auto *left = MakeLeakedStandalonePlacedVolume("test-bool-union-left", new vecgeom::UnplacedBox(5., 5., 5.));
  vecgeom::Transformation3D right_transform(-2.5, 0., 3.5);
  auto *right = MakeLeakedStandalonePlacedVolume("test-bool-union-right", new vecgeom::UnplacedBox(2., 2., 10.),
                                                 &right_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-union-offset-boxes",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kUnion>(vecgeom::kUnion, left, right));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanUnionRotatedTouchingBoxesTestSolid()
{
  auto *left = MakeLeakedStandalonePlacedVolume("test-bool-union-rot-touch-left", new vecgeom::UnplacedBox(1., 1., 1.));
  vecgeom::Transformation3D right_transform(2., -0.9, 0., 90., 0., 0.);
  auto *right = MakeLeakedStandalonePlacedVolume("test-bool-union-rot-touch-right",
                                                 new vecgeom::UnplacedBox(0.95, 1., 1.), &right_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-union-rotated-touching-boxes",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kUnion>(vecgeom::kUnion, left, right));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanUnionOverlappingTubeConeTestSolid()
{
  auto *tube = MakeLeakedStandalonePlacedVolume("test-bool-union-overlap-tube",
                                                new vecgeom::GenericUnplacedTube(0., 3., 6., 0., vecgeom::kTwoPi));
  vecgeom::Transformation3D cone_transform(1.8, 0.4, 0., 25., 10., 0.);
  auto *cone = MakeLeakedStandalonePlacedVolume(
      "test-bool-union-overlap-cone", new vecgeom::GenericUnplacedCone(0., 1.8, 0., 4., 5., 0., vecgeom::kTwoPi),
      &cone_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-union-overlapping-tube-cone",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kUnion>(vecgeom::kUnion, tube, cone));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanUnionDisjointTubesTestSolid()
{
  vecgeom::Transformation3D left_transform(-5., 0., 0.);
  auto *left = MakeLeakedStandalonePlacedVolume("test-bool-union-disjoint-left",
                                                new vecgeom::GenericUnplacedTube(0., 2., 4., 0., vecgeom::kTwoPi),
                                                &left_transform);
  vecgeom::Transformation3D right_transform(5., 0., 0., 0., 25., 0.);
  auto *right = MakeLeakedStandalonePlacedVolume("test-bool-union-disjoint-right",
                                                 new vecgeom::GenericUnplacedTube(0., 2., 4., 0., vecgeom::kTwoPi),
                                                 &right_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-union-disjoint-tubes",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kUnion>(vecgeom::kUnion, left, right));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanIntersectionBoxTubeTestSolid()
{
  auto *left  = MakeLeakedStandalonePlacedVolume("test-bool-intersection-left", new vecgeom::UnplacedBox(5., 5., 5.));
  auto *right = MakeLeakedStandalonePlacedVolume("test-bool-intersection-right",
                                                 new vecgeom::GenericUnplacedTube(2., 4., 6., 0., vecgeom::kTwoPi));
  return MakeStandalonePlacedTestSolid(
      "test-boolean-intersection-box-tube",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kIntersection>(vecgeom::kIntersection, left, right));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanIntersectionRotatedBoxesTestSolid()
{
  auto *left =
      MakeLeakedStandalonePlacedVolume("test-bool-intersection-rot-left", new vecgeom::UnplacedBox(0.6, 1., 1.));
  vecgeom::Transformation3D right_transform(-0.5, -1.5, 0., 45., 0., 0.);
  auto *right = MakeLeakedStandalonePlacedVolume("test-bool-intersection-rot-right",
                                                 new vecgeom::UnplacedBox(1., 2., 1.), &right_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-intersection-rotated-boxes",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kIntersection>(vecgeom::kIntersection, left, right));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanIntersectionRotatedTubeConeTestSolid()
{
  auto *tube = MakeLeakedStandalonePlacedVolume("test-bool-intersection-rot-tube",
                                                new vecgeom::GenericUnplacedTube(0., 4., 6., 0., vecgeom::kTwoPi));
  vecgeom::Transformation3D cone_transform(0.5, 0., 0., 15., 20., 10.);
  auto *cone = MakeLeakedStandalonePlacedVolume(
      "test-bool-intersection-rot-cone", new vecgeom::GenericUnplacedCone(0., 2.5, 0., 5., 6., 0., vecgeom::kTwoPi),
      &cone_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-intersection-rotated-tube-cone",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kIntersection>(vecgeom::kIntersection, tube, cone));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanIntersectionNestedUnionTestSolid()
{
  auto *box =
      MakeLeakedStandalonePlacedVolume("test-bool-intersection-nested-box", new vecgeom::UnplacedBox(3., 3., 3.));
  vecgeom::Transformation3D tube_transform(2., 0., 0.);
  auto *tube       = MakeLeakedStandalonePlacedVolume("test-bool-intersection-nested-tube",
                                                      new vecgeom::GenericUnplacedTube(0., 2., 5., 0., vecgeom::kTwoPi),
                                                      &tube_transform);
  auto *union_node = MakeLeakedBooleanNode<vecgeom::kUnion>("test-bool-intersection-nested-union", box, tube);
  vecgeom::Transformation3D cone_transform(0.4, 0., 0., 0., 20., 10.);
  auto *cone = MakeLeakedStandalonePlacedVolume(
      "test-bool-intersection-nested-cone", new vecgeom::GenericUnplacedCone(0., 4., 0., 3., 4., 0., vecgeom::kTwoPi),
      &cone_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-intersection-nested-union",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kIntersection>(vecgeom::kIntersection, union_node, cone));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionBoxTubeTestSolid()
{
  auto *left = MakeLeakedStandalonePlacedVolume("test-bool-subtraction-left", new vecgeom::UnplacedBox(10., 10., 10.));
  vecgeom::Transformation3D hole_transform(-2.5, -2.5, 0.);
  auto *hole = MakeLeakedStandalonePlacedVolume(
      "test-bool-subtraction-hole", new vecgeom::GenericUnplacedTube(0., 0.9 * 10. / 4., 10., 0., vecgeom::kTwoPi),
      &hole_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-box-tube",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, left, hole));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionTubeFromConeTestSolid()
{
  auto *cone = MakeLeakedStandalonePlacedVolume(
      "test-bool-subtraction-cone-host", new vecgeom::GenericUnplacedCone(0., 4., 0., 6., 6., 0., vecgeom::kTwoPi));
  vecgeom::Transformation3D tube_transform(1.5, 0., 0., 0., 15., 0.);
  auto *tube = MakeLeakedStandalonePlacedVolume("test-bool-subtraction-cone-tube",
                                                new vecgeom::GenericUnplacedTube(0., 1.1, 8., 0., vecgeom::kTwoPi),
                                                &tube_transform);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-tube-from-cone",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, cone, tube));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionUnionFromBoxTestSolid()
{
  auto *box = MakeLeakedStandalonePlacedVolume("test-bool-subtraction-union-box", new vecgeom::UnplacedBox(6., 6., 6.));
  vecgeom::Transformation3D tube_transform(-2., 0., 0.);
  auto *tube = MakeLeakedStandalonePlacedVolume("test-bool-subtraction-union-tube",
                                                new vecgeom::GenericUnplacedTube(0., 1., 7., 0., vecgeom::kTwoPi),
                                                &tube_transform);
  vecgeom::Transformation3D cone_transform(2., 0.2, 0., 0., 20., 0.);
  auto *cone = MakeLeakedStandalonePlacedVolume(
      "test-bool-subtraction-union-cone", new vecgeom::GenericUnplacedCone(0., 0.8, 0., 1.6, 4., 0., vecgeom::kTwoPi),
      &cone_transform);
  auto *cutters = MakeLeakedBooleanNode<vecgeom::kUnion>("test-bool-subtraction-union-cutters", tube, cone);
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-union-from-box",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, box, cutters));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionExactPhiWedgeTestSolid()
{
  auto *host  = MakeLeakedStandalonePlacedVolume("test-bool-subtraction-phi-host",
                                                 new vecgeom::GenericUnplacedTube(1.2, 5., 7., 0., vecgeom::kTwoPi));
  auto *wedge = MakeLeakedStandalonePlacedVolume("test-bool-subtraction-phi-wedge",
                                                 new vecgeom::GenericUnplacedTube(1.2, 5., 7., 0.35, 1.45));

  // The cutter exactly matches the host annular tube in radius and z extent,
  // so only the cutter phi planes remain as real Boolean boundaries.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-exact-phi-wedge",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, host, wedge));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanUnionOfSubtractionsTestSolid()
{
  auto *box =
      MakeLeakedStandalonePlacedVolume("test-bool-union-subtractions-box", new vecgeom::UnplacedBox(3., 3., 3.));
  vecgeom::Transformation3D box_tube_transform(0.5, 0., 0.);
  auto *box_tube    = MakeLeakedStandalonePlacedVolume("test-bool-union-subtractions-box-tube",
                                                       new vecgeom::GenericUnplacedTube(0., 0.8, 4., 0., vecgeom::kTwoPi),
                                                       &box_tube_transform);
  auto *left_branch = MakeLeakedBooleanNode<vecgeom::kSubtraction>("test-bool-union-subtractions-left", box, box_tube);

  auto *cone = MakeLeakedStandalonePlacedVolume(
      "test-bool-union-subtractions-cone", new vecgeom::GenericUnplacedCone(0., 2.2, 0., 3.2, 4., 0., vecgeom::kTwoPi));
  vecgeom::Transformation3D cone_tube_transform(0.6, 0., 0., 0., 15., 0.);
  auto *cone_tube = MakeLeakedStandalonePlacedVolume("test-bool-union-subtractions-cone-tube",
                                                     new vecgeom::GenericUnplacedTube(0., 0.7, 5., 0., vecgeom::kTwoPi),
                                                     &cone_tube_transform);
  vecgeom::Transformation3D right_branch_transform(3.8, 0., 0., 0., 10., 20.);
  auto *right_branch = MakeLeakedBooleanNode<vecgeom::kSubtraction>("test-bool-union-subtractions-right", cone,
                                                                    cone_tube, &right_branch_transform);

  return MakeStandalonePlacedTestSolid(
      "test-boolean-union-of-subtractions",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kUnion>(vecgeom::kUnion, left_branch, right_branch));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanIntersectionOfSubtractionsTestSolid()
{
  auto *box =
      MakeLeakedStandalonePlacedVolume("test-bool-intersection-subtractions-box", new vecgeom::UnplacedBox(4., 3., 3.));
  vecgeom::Transformation3D box_tube_transform(-1., 0., 0.);
  auto *box_tube = MakeLeakedStandalonePlacedVolume("test-bool-intersection-subtractions-box-tube",
                                                    new vecgeom::GenericUnplacedTube(0., 0.7, 5., 0., vecgeom::kTwoPi),
                                                    &box_tube_transform);
  auto *left_branch =
      MakeLeakedBooleanNode<vecgeom::kSubtraction>("test-bool-intersection-subtractions-left", box, box_tube);

  auto *cone =
      MakeLeakedStandalonePlacedVolume("test-bool-intersection-subtractions-cone",
                                       new vecgeom::GenericUnplacedCone(0., 2.5, 0., 4., 4., 0., vecgeom::kTwoPi));
  vecgeom::Transformation3D cone_tube_transform(0.8, 0., 0., 0., 12., 0.);
  auto *cone_tube = MakeLeakedStandalonePlacedVolume(
      "test-bool-intersection-subtractions-cone-tube",
      new vecgeom::GenericUnplacedTube(0., 0.6, 4.5, 0., vecgeom::kTwoPi), &cone_tube_transform);
  vecgeom::Transformation3D right_branch_transform(0.5, 0., 0., 0., 18., 10.);
  auto *right_branch = MakeLeakedBooleanNode<vecgeom::kSubtraction>("test-bool-intersection-subtractions-right", cone,
                                                                    cone_tube, &right_branch_transform);

  return MakeStandalonePlacedTestSolid(
      "test-boolean-intersection-of-subtractions",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kIntersection>(vecgeom::kIntersection, left_branch, right_branch));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanNestedTransformedSubtractionTestSolid()
{
  auto *base = MakeLeakedStandalonePlacedVolume("test-bool-nested-transformed-subtraction-base",
                                                new vecgeom::UnplacedBox(4., 3., 3.));

  constexpr vecgeom::Precision lobe_x     = 2.8;
  constexpr vecgeom::Precision lobe_y     = 0.;
  constexpr vecgeom::Precision rot_z      = 30.;
  constexpr vecgeom::Precision cos_rot_z  = 0.86602540378443865;
  constexpr vecgeom::Precision sin_rot_z  = 0.5;
  vecgeom::Transformation3D lobe_transform(lobe_x, lobe_y, 0., 0., 0., rot_z);
  auto *lobe = MakeLeakedStandalonePlacedVolume("test-bool-nested-transformed-subtraction-lobe",
                                                new vecgeom::UnplacedBox(2., 1.2, 2.5), &lobe_transform);
  auto *nested_left =
      MakeLeakedBooleanNode<vecgeom::kUnion>("test-bool-nested-transformed-subtraction-left", base, lobe);

  // The cutter has the same rotation and y/z half lengths as the lobe. Its
  // local +X face exactly coincides with the lobe local +X face, while its
  // local volume covers only the lobe half-space 0 <= x <= 2.
  vecgeom::Transformation3D cutter_transform(lobe_x + cos_rot_z, lobe_y + sin_rot_z, 0., 0., 0., rot_z);
  auto *cutter = MakeLeakedStandalonePlacedVolume("test-bool-nested-transformed-subtraction-cutter",
                                                  new vecgeom::UnplacedBox(1., 1.2, 2.5), &cutter_transform);

  // This probes nested Boolean transform handling plus coincident constituent
  // surfaces: the subtraction shape shares a face with the transformed lobe but
  // still removes material from the nested union and the base box.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-nested-transformed-subtraction",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, nested_left, cutter));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionExactPolyconeSectionsTestSolid()
{
  const int num_z                      = 5;
  const vecgeom::Precision z[num_z]    = {-12., -6., 0., 6., 12.};
  const vecgeom::Precision rmin[num_z] = {0., 0., 0., 0., 0.};
  const vecgeom::Precision rmax[num_z] = {3., 5., 4., 6., 2.};
  auto *host =
      new vecgeom::SimplePolycone("test-bool-polycone-section-host", 0., vecgeom::kTwoPi, num_z, z, rmin, rmax);
  vecgeom::Transformation3D cut1_transform(0., 0., -3.);
  auto *cut1 = MakeLeakedStandalonePlacedVolume(
      "test-bool-polycone-section-cut1", new vecgeom::GenericUnplacedCone(0., 5., 0., 4., 3., 0., vecgeom::kTwoPi),
      &cut1_transform);
  auto *minus_first =
      MakeLeakedBooleanNode<vecgeom::kSubtraction>("test-bool-polycone-section-minus-first", host, cut1);

  vecgeom::Transformation3D cut2_transform(0., 0., 9.);
  auto *cut2 = MakeLeakedStandalonePlacedVolume(
      "test-bool-polycone-section-cut2", new vecgeom::GenericUnplacedCone(0., 6., 0., 2., 3., 0., vecgeom::kTwoPi),
      &cut2_transform);

  // Each cutter is an exact cone-section match to one host polycone section.
  // The result keeps the remaining sections and stresses coplanar Boolean
  // subtraction boundaries without making the whole solid void.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-exact-polycone-sections",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, minus_first, cut2));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionPolyhedronExactZSlabTestSolid()
{
  // Polyhedron-left subtraction stress case: exercise a zero-thickness Boolean
  // hand-off where the right solid exactly removes one complete z segment.
  constexpr int sides                  = 8;
  constexpr int host_num_z             = 4;
  const vecgeom::Precision phi_start   = -0.5 * vecgeom::kTwoPi / sides;
  const vecgeom::Precision host_z[]    = {-6., -2., 2., 6.};
  const vecgeom::Precision host_rmin[] = {0., 0., 0., 0.};
  const vecgeom::Precision host_rmax[] = {5., 5., 5., 5.};
  auto *host = MakeLeakedPolyhedronNode("test-bool-polyhedron-z-slab-host", phi_start, vecgeom::kTwoPi, sides,
                                        host_num_z, host_z, host_rmin, host_rmax);

  constexpr int cutter_num_z             = 2;
  const vecgeom::Precision cutter_z[]    = {-2., 2.};
  const vecgeom::Precision cutter_rmin[] = {0., 0.};
  const vecgeom::Precision cutter_rmax[] = {5., 5.};
  auto *cutter = MakeLeakedPolyhedronNode("test-bool-polyhedron-z-slab-cutter", phi_start, vecgeom::kTwoPi, sides,
                                          cutter_num_z, cutter_z, cutter_rmin, cutter_rmax);

  // The cutter exactly removes one full z slab of the left polyhedron, so the
  // remaining Boolean boundary is dominated by shared coplanar z faces.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-polyhedron-exact-z-slab",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, host, cutter));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionPolyhedronExactRadialShellTestSolid()
{
  // Polyhedron-left subtraction stress case: exercise coincident side faces by
  // subtracting a shell with the same outer polyhedron boundary as the host.
  constexpr int sides                  = 8;
  constexpr int num_z                  = 2;
  const vecgeom::Precision phi_start   = -0.5 * vecgeom::kTwoPi / sides;
  const vecgeom::Precision z[]         = {-5., 5.};
  const vecgeom::Precision host_rmin[] = {0., 0.};
  const vecgeom::Precision host_rmax[] = {5., 5.};
  auto *host = MakeLeakedPolyhedronNode("test-bool-polyhedron-radial-shell-host", phi_start, vecgeom::kTwoPi, sides,
                                        num_z, z, host_rmin, host_rmax);

  const vecgeom::Precision shell_rmin[] = {3., 3.};
  const vecgeom::Precision shell_rmax[] = {5., 5.};
  auto *shell = MakeLeakedPolyhedronNode("test-bool-polyhedron-radial-shell-cutter", phi_start, vecgeom::kTwoPi, sides,
                                         num_z, z, shell_rmin, shell_rmax);

  // The right solid is a hollow polyhedron sharing the host outer shell exactly;
  // the subtraction result should behave like the remaining inner prism.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-polyhedron-exact-radial-shell",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, host, shell));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionPolyhedronSideCutTubeTestSolid()
{
  // Polyhedron-left subtraction stress case: cover a curved right operand that
  // cuts through one flat polyhedron side and creates polyhedron/tube edge loops.
  constexpr int sides                = 8;
  constexpr int num_z                = 2;
  const vecgeom::Precision phi_start = -0.5 * vecgeom::kTwoPi / sides;
  const vecgeom::Precision z[]       = {-5., 5.};
  const vecgeom::Precision rmin[]    = {0., 0.};
  const vecgeom::Precision rmax[]    = {5., 5.};
  auto *host = MakeLeakedPolyhedronNode("test-bool-polyhedron-side-cut-tube-host", phi_start, vecgeom::kTwoPi, sides,
                                        num_z, z, rmin, rmax);

  vecgeom::Transformation3D tube_transform(4.3, 0., 0.);
  auto *tube = MakeLeakedStandalonePlacedVolume("test-bool-polyhedron-side-cut-tube-cutter",
                                                new vecgeom::GenericUnplacedTube(0., 1., 6., 0., vecgeom::kTwoPi),
                                                &tube_transform);

  // The tube crosses the left side enough to make a real opening, avoiding the
  // zero-angle contact of exact tangency while keeping hard mixed-boundary edges.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-polyhedron-side-cut-tube",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, host, tube));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionPolyhedronPhiSeamRotatedTestSolid()
{
  // Polyhedron-left subtraction stress case: cover transformed cutter seams so
  // Boolean methods cannot accidentally rely on aligned constituent frames.
  constexpr int sides             = 10;
  constexpr int num_z             = 2;
  const vecgeom::Precision z[]    = {-5., 5.};
  const vecgeom::Precision rmin[] = {0., 0.};
  const vecgeom::Precision rmax[] = {5., 5.};
  auto *host =
      MakeLeakedPolyhedronNode("test-bool-polyhedron-phi-rot-host", 0., vecgeom::kTwoPi, sides, num_z, z, rmin, rmax);

  vecgeom::Transformation3D cutter_transform(0., 0., 0., 0., 0., 27.);
  auto *cutter = MakeLeakedPolyhedronNode("test-bool-polyhedron-phi-rot-cutter", 0., 0.45 * vecgeom::kPi, 4, num_z, z,
                                          rmin, rmax, &cutter_transform);

  // Rotating the phi-sector cutter makes its seam planes non-aligned with the
  // left polyhedron sides while preserving shared z and radial extents.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-polyhedron-phi-seam-rotated",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, host, cutter));
}

inline std::unique_ptr<vecgeom::VPlacedVolume> MakeBooleanSubtractionPolyhedronNearCoincidentTestSolid()
{
  // Polyhedron-left subtraction stress case: cover near-coincident but distinct
  // cutter faces, which is the tolerance-sensitive complement of exact sharing.
  constexpr int sides                  = 8;
  constexpr int host_num_z             = 3;
  const vecgeom::Precision phi_start   = -0.5 * vecgeom::kTwoPi / sides;
  const vecgeom::Precision host_z[]    = {-5., 0., 5.};
  const vecgeom::Precision host_rmin[] = {0., 0., 0.};
  const vecgeom::Precision host_rmax[] = {5., 5., 5.};
  auto *host = MakeLeakedPolyhedronNode("test-bool-polyhedron-near-coincident-host", phi_start, vecgeom::kTwoPi, sides,
                                        host_num_z, host_z, host_rmin, host_rmax);

  constexpr int cutter_num_z             = 2;
  const vecgeom::Precision eps           = 20. * vecgeom::kConeTolerance;
  const vecgeom::Precision cutter_z[]    = {-5. + eps, -eps};
  const vecgeom::Precision cutter_rmin[] = {0., 0.};
  const vecgeom::Precision cutter_rmax[] = {5. - eps, 5. - eps};
  auto *cutter = MakeLeakedPolyhedronNode("test-bool-polyhedron-near-coincident-cutter", phi_start, vecgeom::kTwoPi,
                                          sides, cutter_num_z, cutter_z, cutter_rmin, cutter_rmax);

  // The cutter is deliberately offset by only a few numerical tolerances, so
  // boundary walking must not confuse near-coincident surfaces with exact ones.
  return MakeStandalonePlacedTestSolid(
      "test-boolean-subtraction-polyhedron-near-coincident",
      new vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction>(vecgeom::kSubtraction, host, cutter));
}

} // namespace test
} // namespace vecgeom

#endif
