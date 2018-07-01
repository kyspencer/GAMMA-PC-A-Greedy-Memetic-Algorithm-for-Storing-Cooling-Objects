# GAMMA-PC-A-Greedy-Memetic-Algorithm-for-Storing-Cooling-Objects

This data repository includes the input and output for the results presented in 
K.Y. Spencer, P.V. Tsvetkov, and J.J. Jarrell, "A Greedy Memetic Algorithm for a 
Multiobjective Dynamic Bin Packing Problem for Storing Cooling Objects," *Journal of
Heuristics*.

**Repository Structure** </br>
```
README.md 
license.txt 
Analysis/ 
Results_Dynamic/ 
Results_Static/ 
SampleScripts/ 
```


The **Analysis/** folder contains plots of the results as well as the scripts used to 
generate the 3D empirical attainment functions. </br>
```
Analysis/eaf3D/ 
Analysis/SBSBPP500/     (the static problem) 
Analysis/Cookies24/     (the toy dynamic problem) 
Analysis/Cookies1000/   (the full dynamic problem) 
```

The **Results_Dynamic/** folder contains the raw data used to generate the problems, the
results from each of the algorithms, and an analysis of the toy problem structure.
Each of the folders for the algorithmic results contain folders for both the toy and 
the full dynamic problem. </br>
```
Results_Dyanmic/GAMMA-PC/ 
Results_Dyanmic/MOMA/ 
Results_Dyanmic/NSGA-II/ 
Results_Dyanmic/RawData/ 
Results_Dyanmic/ToyProblem/
```

The **Results_Static/** folder contains the raw data used to generate the problems and the 
results from each of the algorithms considered. </br>
```
Results_Static/GAMMA-PC/ 
Results_Static/MOEPSO/ 
Results_Static/MOMA/ 
Results_Static/MOMAD/ 
Results_Static/NSGA-II/ 
Results_Static/RawData/ 
```

The **SampleScripts/** folder contains some of the Python scripts that were used to 
run the algorithms. This folder is not structured as a Python package. Future work
might improve the organization of this folder to be easily retrievable by other
users, but please use caution if borrowing from this folder for now. The primary 
purpose of this repository is to share the data referenced in the published journal 
article. The comments within the scripts also need improvement.

*Please note that the data in the RawData folders was generated using 2DCPackGen. 
You can find more information about this program from the following article:*

Silva E, Oliveira JF, Wäscher G "2DCPackGen: A problem generator for two-dimensional
rectangular cutting and packing problems," *European Journal of Operational
Research*, vol. 237, pp. 846–856 (2014).

**Acronyms:**
* eaf = Empirical Attainment Function
* GRASP = Greedy Randomized Adaptive Search Procedure
* GAMMA-PC = GRASP-enabled Adaptive Multiobjective Memetic Algorithm with Partial Clustering
* MOEPSO = MultiObjective Evolutionary Particle Swarm Optimization
* MOMA = MultiObjective Memetic Algorithm
* MOMAD = MultiObjective Memetic Algorithm based on Decomposition
* NSGA-II = nondominated sorting genetic algorithm II
* SBSBPP = Single Bin Size Bin Packing Problem
