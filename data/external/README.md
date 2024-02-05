# External datasets

Here we document how to obtain the external datasets used in the project.

## TMS-EEG Annen et al.

1. Head over to the EBRAINS KG: https://doi.org/10.25493/G8E3-DQE
2. Request data access and download the data (whole bucket)
3. Extract to annen_tms_eeg_ebrains

```
├── data-descriptor_a3dade301e9f.pdf
├── participants.csv
├── sub-S01
│   └── sub-S01.nxe
│   ....
└── sub-S12
    └── sub-S12.nxe
```



Raimondo, F., Wolff, A., Sanz, L. R. D., Barra, A., Cassol, H., Carrière, M., Laureys, S., & Gosseries, O. (2020). TMS-EEG perturbation in patients with disorders of consciousness [Data set]. Human Brain Project Neuroinformatics Platform. 


## TMS-EEG Nieus et al

1. Head over to the EBRAINS KG: EBRAINS. https://doi.org/10.25493/5TNA-R5P
2. Request data access and download the data (whole bucket)
3. Extract to nieus_tms_eeg_ebrains

```
├── data-descriptor.pdf
├── dataset_description.json
├── derivatives
│   └── epochs
│       ├── sub-01
│       │   ├── eeg
│       │   │   ├── sub-01_task-tmseeg_coordsystem.json
│       │   │   ├── sub-01_task-tmseeg_electrodes.tsv
│       │   │   ├── sub-01_task-tmseeg_run-01_channels.tsv
│       │   │   ├── sub-01_task-tmseeg_run-01_epochs.json
│       │   │   ├── sub-01_task-tmseeg_run-01_epochs.npy
│       │   │   └── sub-01_task-tmseeg_run-01_epochs.tsv
│       │   └── sub-01_scans.tsv
│       ...
│       └── sub-06
│           ├── ...
├── participants.json
├── participants.tsv
├── raw
│   ├── sub-01
│   │   └── anat
│   │       ├── sub-01_T1w.json
│   │       └── sub-01_T1w.nii
│   ...
│   └── sub-06
│       └── ...
└── TMSEEGlesion_README.md
```

Nieus, T., Casarotto, S., Viganò, A., & Massimini, M. (2021). Results for complexity measures and a read-out of the state of cortical circuits after injury [Data set]. EBRAINS. https://doi.org/10.25493/5TNA-R5P


## F-TRACT

Grows on EBRAINS in several versions, here we use the latest one in the two flavors:
- F-TRACT_P_11_v2307: responses until 200ms
- F-TRACT_P_01_v2307: responses until 50ms

Jedynak, M., Boyer, A., Lemaréchal, J.-D., Trebaul, L., Tadel, F., Bhattacharjee, M., Chanteloup-Forêt, B., Deman, P., Tuyisenge, V., Ayoubian, L., Hugues, E., Saubat-Guigui, C., Zouglech, R., Reyes-Mejia, G. C., Tourbier, S., Hagmann, P., Adam, C., Barba, C., Bartolomei, F., … F-TRACT Consortium. (2023). F-TRACT: a probabilistic atlas of anatomo-functional connectivity of the human brain (F-TRACT_P_01_v2307) [Data set]. EBRAINS. https://doi.org/10.25493/JYVR-WJ7
