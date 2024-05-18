# External datasets

Here we document how to obtain the external datasets used in the project. For each dataset, files used in the project are commented here.

## F-TRACT dataset

Availible on EBRAINS in several versions, here we use the latest F-TRACT_P_11_v2307 (responses until 200 ms) and F-TRACT_P_01_v2307 (50 ms). We use DKT and MNI-HCP-MMP1 parcellations.

Jedynak, M., Boyer, A., Lemaréchal, J.-D., Trebaul, L., Tadel, F., Bhattacharjee, M., Chanteloup-Forêt, B., Deman, P., Tuyisenge, V., Ayoubian, L., Hugues, E., Saubat-Guigui, C., Zouglech, R., Reyes-Mejia, G. C., Tourbier, S., Hagmann, P., Adam, C., Barba, C., Bartolomei, F., … F-TRACT Consortium. (2023). F-TRACT: a probabilistic atlas of anatomo-functional connectivity of the human brain (F-TRACT_P_01_v2307) [Data set]. EBRAINS. https://doi.org/10.25493/5AM4-J3F

```
F-TRACT
├── data-descriptor_eaa26e226384.pdf
├── <parcellation>
│   └── <parcellation>.txt			# list of ROI names in the order that is used in the matrices
│   └── probability.txt.gz			# matrix of response probabilities, one row per stimulated ROI
│   └── amplitude__median.txt.gz	# matrix of response median amplitudes, one row per stimulated ROI
│   └── onset_delay__median.txt.gz	# matrix of response onset delays, one row per stimulated ROI
│   └── peak_delay__median.txt.gz	# matrix of response peak delays, one row per stimulated ROI
│   └── N_implantations.txt.gz	    # matrix of implantations, m_ij denoted how many times were electrodes implantet to both i and j
│   └── N_stimulations.txt.gz	    # matrix of stimulations, m_ij denoted how many times respones was recorded at j while i was stimulated
│   ....
└── ....
F-TRACT_short                       # same structure, 50 ms responses
└── ....
```

## PyTepFit 

TMS-EEG data in the PyTepFit study were taken from an open dataset at https://figshare.com/articles/dataset/TEPs-_SEPs/7440713

data: Biabani M Fornito A Mutanen TP Morrow J Rogasch NC (2019) figshare ID TEPs-_SEPs/7440713. TEPs-PEPs.
PyTepFit: Davide Momi, Zheng Wang, John D Griffiths (2023) TMS-evoked responses are driven by recurrent large-scale network dynamics eLife 12:e83232 https://doi.org/10.7554/eLife.83232 
    

## Rosen-Halgren dataset

Availible on Zenodo, we use the version 3.2. Associated paper decribing the data availible here: https://doi.org/10.1523/ENEURO.0416-20.2020

Rosen, B. Q., & Halgren, E. (2021). A whole-cortex probabilistic diffusion tractography connectome (3.2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10150880

```
rosen_halgren
├── averageConnectivity_Fpt.mat				# structural connectivity matrix consisting of the 360 HCP-MMPS1.0 atlas parcels (streamline counts fractionally scaled yielding the log probability) together with ROI names
├── averageConnectivity_tractLengths.mat	# matrix of average tract lengths between ROIs
└── ....
```

## Mica-Mics dataset

Availible at the Canadian Open Neuroscience Platform (CONP): https://n2t.net/ark:/70798/d72xnk2wd397j190qv We used glasser360 and schaefer200 parcellation.

Royer, J., Rodriguez-Cruces, R., Tavakol, S., Lariviere, S., Herholz, P., Li, Q., Vos de Wael, R., Paquola, C., Benkarim, O., Park, B., Lowe, A.J., Margulies, D.S., Smallwood, J., Bernasconi, A., Bernasconi, N., Frauscher, B., Bernhardt, B.C., 2021. An open MRI dataset for multiscale neuroscience. bioRxiv 2021.08.04.454795. https://doi.org/10.1101/2021.08.04.454795

Downloaded using datalad with this sequence of commands:
```
datalad install https://github.com/CONP-PCNO/conp-dataset.git
cd conp-dataset
datalad install projects/mica-mics
cd projects/mica-mics/MICs_release/derivatives/micapipe
datalad get */ses-01/dwi/*glasser360*.txt
cp */ses-01/dwi/*<parcellation><number of ROIs>*.txt <path_to_my_mica_dir>/<parcellation><number of ROIs>/
```

Annotations downloaded form GitHub: https://github.com/MICA-MNI/micapipe/blob/master/parcellations/

```
mica-mics
├── <parcellation><number of ROIs>
│   └── sub-HC<subject number xxx>_ses-01_space-dwinative_atlas-<parcellation><number of ROIs>_desc-edgeLength.txt	# edge length matrix for subject xxx
│   └── sub-HC<subject number xxx>_ses-01_space-dwinative_atlas-<parcellation><number of ROIs>_desc-sc.txt			# structural connectivity matrix for subject xxx
|   └── ....
├── schaefer200_annot
|   └── lh.schaefer-200_mics.annot # download from https://github.com/MICA-MNI/micapipe/blob/master/parcellations/lh.schaefer-200_mics.annot
|   └── rh.schaefer-200_mics.annot # download from https://github.com/MICA-MNI/micapipe/blob/master/parcellations/rh.schaefer-200_mics.annot
└── ....
```




## Domhof dataset

Availible on EBRAINS: https://doi.org/10.25493/NVS8-XS5 Downloaded parcellations 070-DesikanKilliany.zip as DKT, 100-Schaefer17Networks.zip as Schaefer, 150-Destrieux.zip as Dexterious.

Domhof, J. W. M., Jung, K., Eickhoff, S. B., & Popovych, O. V. (2022). Parcellation-based structural and resting-state functional brain connectomes of a healthy cohort (v1.1) [Data set]. EBRAINS. https://doi.org/10.25493/NVS8-XS5

```
domhof
├── <parcellation>
│   └── 0ImageProcessing
│   └── 1StructuralConnectivity
│	│	└── <xxx>					# subject number
│	│		└── Counts.csv			#
│	│		└── Lengths.csv			#
│   └── 2FunctionalConnectivity
└── ....
```

# Other external resources

