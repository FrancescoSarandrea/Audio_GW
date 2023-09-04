repodir=/home/fsarandrea/anaconda3/envs/igwn-py39/bin
start_time=1165491287
event_time=1165491287.76172


${repodir}/BayesWave --checkpoint  --ifo H1 --psdlength 4.0 --H1-cache H1.cache --H1-channel H1:GWOSC-4KHZ_R1_STRAIN --bayesLine  --updateGeocenterPSD  --waveletPrior  --Dmax 100 --glitchOnly  --Niter 1000 --trigtime $start_time --segment-start $start_time --srate 1024.0 --seglen 4.0 --window 1.0 --H1-flow 30.0 --psdstart $start_time --outputDir outputDir2 --dataseed 1234

${repodir}/BayesWavePost --ifo H1 --psdlength 4.0 --0noise  --bayesLine  --dataseed 1234 --trigtime $start_time --segment-start $start_time --srate 1024.0 --seglen 4.0 --window 1.0 --H1-flow 30.0 --psdstart $start_time --outputDir outputDir --H1-cache interp:outputDir/H1_fairdraw_asd.dat

${repodir}/megaplot_simple.py -o outputDir2 -s $start_time -t $event_time


