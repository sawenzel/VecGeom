
/performance/maxPoints 100000
/performance/repeat 10

# One of my old favorites, with a few sharp turns

/solid/G4Polyhedra2 0 270 6 6 (-0.6,0.0,-1.0,0.5,0.5,1.0) (0.5,0.5,0.4,0.4,0.8,0.8) (0.6,0.6,1.0,1.0,1.0,1.1)

/performance/errorFileName log/polyhedra-turns-p10k/polyhedrap.a1.log
/control/execute usolids/performance.sbt

exit
