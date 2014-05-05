
/performance/differenceTolerance 0.02

# does not make sense to test safety from outside for inside points
/performance/maxInsidePercent 10
/performance/maxOutsidePercent 80
/performance/method SafetyFromOutside
/performance/run

/performance/maxInsidePercent 25
/performance/maxOutsidePercent 50
/performance/method Inside
/performance/run

/performance/maxInsidePercent 10
/performance/maxOutsidePercent 80
/performance/method DistanceToIn
/performance/run

/performance/maxInsidePercent 80
# maxOutsidePercent must be 0, otherwise errors are produced in Geant4
/performance/maxOutsidePercent 10
/performance/method DistanceToOut
/performance/run 

/performance/maxInsidePercent 10
/performance/maxOutsidePercent 10
/performance/method Normal
/performance/run

/performance/maxInsidePercent 80
# does not make sense to test safety from inside for outside points
/performance/maxOutsidePercent 10
/performance/method SafetyFromInside
/performance/run

