#Simulating dynamics on a 5x5 Manhattan-like grid using the SUMO package.

#routemanip file is to manipulate xml files and add stopped vehicles randomly, as well as analyze the subsequent output (aggregated.xml)


1)Generate grid net file:
	netgenerate --grid --grid.number=3 --grid.length=200 --output-file=grid.net.xml
	then go to netedit, make all intersections as traffic lights, add TLS

2)Generate random trips:
	randomTrips.py -n grid.net.xml -o grid.trips.xml

3) Use DUAROUTER to convert trips to routes
	duarouter --net-file grid.net.xml --trip-files grid.trips.xml -o grid.rou.xml

4) Make cfg file for SUMO, with .net, .rou files, and other inputs


All together now:

netgenerate --grid --grid.number=5 --grid.length=200 --output-file=grid.net.xml
randomTrips.py -n grid.net.xml -e 10000 -o grid.trips.xml
duarouter --net-file grid.net.xml --trip-files grid.trips.xml -o grid.rou.xml

*****
Stopping vehicles at edge:
add this to a particular vehicle in grid.rou.xml that passes through say B2B1 - <stop lane="B2B1_0" until="10000"/>
to stop at particular position wrt edge, do <stop lane="B3B4_0" endPos="5" until="10000"/>. Choose values for endpos between 5 & 175 (10 & 170 just to be safe)

Consider toying with time spent stopped, vs recovery time - presumably recovery is much longer if vehicles are stopped for too long

TIP - to avoid teleportation, run from terminal as: sumo-gui -c grid.sumocfg --time-to-teleport=10000 --no-internal-links
no internal links above, so that left turn lane is not blocked

NEW: sumo-gui -c grid.sumocfg --no-internal-links
realized we want teleportation to avoid situations where 1 vehicle blocks entire grid

*****
Adding traffic lights:
edit grid.net.xml in NETEDIT, add traffic lights at all nodes


******
Adding lanes:
edit grid.net.xml in NETEDIT, and make edges have 3 lanes


******
Only outputting last time
make an additional file with aggregated output
