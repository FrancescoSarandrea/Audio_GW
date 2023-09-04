# Pre-processing of Glitches
This repository contains the scripts used to reconstruct the Glitches with BayesWave.

**H1.cache** contains the information about the .gwf file of the signal. It assumes that the file is in the same repository.

**example_GW** runs the BayesWave pipeline. It outputs the reconstructed signal and graphs into the outputDir folder. 

**megaplot_simple** prints the recontructed glitch signal. It is a slightly modified version of the megaplot.py script which is included in the standard BayesWave pipeline.
