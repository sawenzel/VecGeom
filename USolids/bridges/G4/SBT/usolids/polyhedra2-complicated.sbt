
/performance/maxPoints 10000
/performance/repeat 10

# Build a much more complicated polyhedra
#
/solid/G4Polyhedra 0 360 6 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 

/performance/errorFileName log/polyhedra-complicated-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

# Build a much more complicated polyhedra, now with a slice
#
/solid/G4Polyhedra 0 90 3 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 

/performance/errorFileName log/polyhedra-complicated-slice-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

# Build a much more complicated polyhedra, now with a thin slice
/solid/G4Polyhedra -1 2 2 17 (0.0,0.2,0.3,0.32,0.32,0.4,0.4,0.5,0.5,0.8,0.8,0.9,0.9,0.8,0.8,0.3,0.0) (-0.5,-0.5,-1.1,-1.1,-0.4,-0.4,-1.0,-1.0,-0.4,-1.0,0.0,0.0,0.2,0.2,1.0,0.0,1.0) 

/performance/errorFileName log/polyhedra-complicated-thin-slice-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

exit
