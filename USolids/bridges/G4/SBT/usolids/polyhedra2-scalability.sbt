#
# GEANT4 SBT Script to test G4Polyhedra
# DCW 25/3/99 First try
# DCW 23/5/99 Add voxel tests
#
# Increment the numbers below to **really** consume CPU time

/performance/repeat 100

/solid/G4Polyhedra2 0 180 3 2 (0,1) (1,1) (2,3)

/performance/errorFileName log/polyhedra2-3-2-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

/solid/G4Polyhedra2 0 180 5 4 (-1.0,0,1.0,2.0) (1, 1, 1, 1) (2,3,3,2)

/performance/errorFileName log/polyhedra2-5-4-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

/solid/G4Polyhedra2 0 180 7 6 (-1.0,0,1,2,3,4) (1, 1, 1, 1, 1, 1) (2,3,3,4,3,2)

/performance/errorFileName log/polyhedra2-7-6-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

/solid/G4Polyhedra2 0 180 9 8 (-1.0,0,1,2,3,4,5,6) (1, 1, 1, 1,1,1,1,1) (2,3,3,4,3,4,3,2)

/performance/errorFileName log/polyhedra2-9-8-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

/solid/G4Polyhedra2 0 180 11 10 (-1.0,0,1,2,3,4,5,6,7,8) (1, 1, 1, 1,1,1,1,1,1,1) (2,3,3,4,3,4,3,4,3,2)

/performance/errorFileName log/polyhedra2-11-10-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

/performance/repeat 100

/solid/G4Polyhedra2 0 180 21 20 (-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18) (1, 1, 1, 1,1,1,1,1,1,1, 1, 1, 1, 1,1,1,1,1,1,1) (2,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,2)

/performance/errorFileName log/polyhedra2-21-20-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

/solid/G4Polyhedra2 0 180 51 50 (-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49) (1, 1, 1, 1,1,1,1,1,1,1, 1, 1, 1, 1,1,1,1,1,1,1,1, 1, 1, 1,1,1,1,1,1,1,1, 1, 1, 1,1,1,1,1,1,1,1, 1, 1, 1,1,1,1,1,1,1) (2,3,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4,3,2)

/performance/errorFileName log/polyhedra2-51-50-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt
