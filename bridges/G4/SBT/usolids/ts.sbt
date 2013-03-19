
# performance scalability test

/performance/maxPoints 10000
/performance/repeat 1

#/solid/file ts/sphere.stl
/solid/file ts/sphere-small.stl # OK
/solid/file ts/tank-8.7k.stl # OK
/solid/file ts/f1-373k.stl # stl file wrong
/solid/file ts/bottle-1.2k.stl # getpolyhedron does not work
/solid/file ts/porsche-151k.stl # two points are wrong ...
/solid/file ts/tire_v-72.3k.stl # inside + normal ok, rest to test
/solid/file ts/mannequin-head-13.3k.stl # OK
/solid/file ts/ktoolcor-3.8k.stl # OK
/solid/file ts/ktoolcav-4k.stl # not able to create points on surface
/solid/file ts/\liver-38.1k.stl # OK
/solid/file ts/red_pepper-30k.stl # OK
/solid/file ts/3d_face-163k.stl # wrong mesh? getpylehedron warnings + problems in inside
/solid/file ts/knot-76.4k.stl # !! many points on inside wrong, g4ts contruction giving warnings
/solid/file ts/brain-gear-76.4k.stl # THIS SOLID SEEMS NOT CLOSED !!! many points on inside wrong 
/solid/file ts/pump-178k.stl # !! many errors Inside, SafetyFromOutSide, DistancetoOut and In, see ./pump-178k/, POLYHEDRON WARNINGS
/solid/file ts/ship-828.stl # !! THIS FILE IS NOT COMPLETELY ENCLOSED => some points on normal wrong
/solid/file ts/decoration-256k.stl # fails many points at inside http://carver3d.com/content/10-sample , differences in SafetyFromOutside, rest is ok ./decoration-256k/
/solid/file ts/anno-245k.stl # errors in inside, getpolyhedron problems
/solid/file ts/sunduk-290k.stl
/solid/file ts/anya1-300k.stl
/solid/file ts/carburator-500k.st
/solid/file ts/dogskull-1.2m.stl
/solid/file ts/dragonfusion-900k.stl
/solid/file ts/spaceinvadermagnet-376.stl # OK
/solid/file ts/scissorshandlepolygons-500k.stl
/solid/file ts/screwingboxperfecttop-3k.stl
#/solid/file ts/key-1.1k.stl # OK
/solid/file ts/humanoid-96.stl # OK
/solid/file ts/finfet-8.3k.stl # OK
/solid/file ts/finfet-27.6k.stl # OK
/solid/file ts/skull-121k.stl # works inside, normal rest to test

#/solid/G4TessellatedSolidFromSTLFile			1000000

/solid/file ts/orb-497k.gdml # OK
/solid/file ts/foil-2.5k.gdml # OK

/solid/file ts/foil-72.9k.gdml # OK
/solid/file ts/foil-37.2k.gdml # OK
/solid/file ts/foil-112k.gdml # OK
/solid/file ts/foil-164k.gdml # OK
/solid/file ts/key-1.1k.gdml # OK

/solid/file ts/hollow.gdml

/solid/G4TessellatedSolidFromGDMLFile -1


/performance/errorFileName log/hollow2/sbt.log

/control/execute usolids/performance.sbt

exit


/performance/maxPoints 1000
/performance/repeat 1000

#/solid/G4MultiUnion 3
#/solid/G4TessellatedSolidTransform 1
#/solid/file ts/foil-164k
#/solid/G4TessellatedSolidFromPlainFile 1
/solid/file ts/foil-164k.gdml
/solid/G4TessellatedSolidFromGDMLFile 100000
/performance/errorFileName log/tessellatedsolid-test-t100/sbt.log

/control/execute usolids/performance.sbt

exit

























/performance/maxPoints 1000
/performance/repeat 1

/solid/file ts/key-1.1k.gdml
/solid/G4TessellatedSolidFromGDMLFile 10000
/performance/errorFileName log/tessellatedsolid-test2-t10k/sbt.log

/control/execute usolids/performance.sbt

exit










/performance/maxPoints 10
/performance/repeat 1

/solid/file ts/orb-497k.gdml
/solid/G4TessellatedSolidFromGDMLFile 10000                             
/performance/errorFileName log/tessellatedsolid-sphere-t10k/sbt.log

/control/execute usolids/values.sbt

exit









 
/performance/maxInsidePercent 80
# maxOutsidePercent must be 0, otherwise errors are produced in Geant4
/performance/maxOutsidePercent 0
/performance/method DistanceToOut
/performance/run 

exit

/control/execute usolids/values.sbt

exit

/solid/G4MultiUnion 2
/performance/maxPoints 10000
/performance/repeat 1
/performance/errorFileName log/multiunion-2-t10k/sbt.log
/control/execute usolids/test.sbt


/test/errorFileName log/trd.a1.log
/test/run

#
# --- trd.a2.log
# Up the ante ans generate points on a grid
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.a2.log
/test/run
#
# --- trd.b1.log
# Adjust just x
#
#/test/gridSizes 0 0 0 m
/solid/G4trd 0.5 1.5 1 1 1
/test/errorFileName  log/trd.b1.log
/test/run
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.b2.log
/test/run
#
# --- trd.c1.log
# Adjust x and y
#
/test/gridSizes 0 0 0 m
/solid/G4trd 0.5 1.5 0.25 1 1
/test/errorFileName  log/trd.c1.log
/test/run
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.c2.log
/test/run
#
#
# --- trd.d1.log
# extreme case
#
/test/widths 1 0.00002 1 m
/test/gridSizes 0 0 0 m
/solid/G4trd 0.000001 1 0.00001 0.00002 1
/test/errorFileName  log/trd.d1.log
/test/run
#
/test/gridSizes 0.2 0.2 0.2 m
/test/errorFileName  log/trd.d2.log
/test/run
#
exit 


?
Polyhedron::SetReferences: List 133834 is not empty
Polyhedron::SetReferences: List 133838 is not empty
Polyhedron::SetReferences: List 133849 is not empty
Polyhedron::SetReferences: List 133853 is not empty
Polyhedron::SetReferences: List 133888 is not empty
Polyhedron::SetReferences: List 133890 is not empty
Polyhedron::SetReferences: List 133894 is not empty
Polyhedron::SetReferences: List 133905 is not empty
Polyhedron::SetReferences: List 133909 is not empty
Polyhedron::SetReferences: List 133944 is not empty
Polyhedron::SetReferences: List 133946 is not empty
Polyhedron::SetReferences: List 133950 is not empty
Polyhedron::SetReferences: List 133961 is not empty
Polyhedron::SetReferences: List 133965 is not empty
Polyhedron::SetReferences: List 134000 is not empty

? serialization, vertices too close to each other are united, is it ok?
	coincident = (GetVertex(i)-right.GetVertex(j)).Mag2() < tolerance;

? bridge
