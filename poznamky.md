# Diplomka -- poznámky

- chat: https://chat.ebrains.eu/home
- uspořádní repozitáře: https://drivendata.github.io/cookiecutter-data-science/

## Termíny a formality

- [kompletní pokyny pro diplomky na FI MUNI](https://www.fi.muni.cz/files/studijni/bp-dp-na-fi-v280423.pdf)
    - pozor! na rozdíl od MFF UK doporučený (40) a minimální (30) počet normostran (normostrana ~1800 znaků)
- [termíny na webu FI](https://www.fi.muni.cz/studies/master/dates.html.cs)
    - **zadání** 18. 3. 2024 (už musí být kompletní a potvrzení v IS)
        - přihlásit se k tématu musím už do 4. 3. 2024!
    - **odevzdání** 20. 5. 2024
    - uzavření studia 11. 6. 2024
    - **státnice a obhajoby** od 17. 6. 2024 do 21. 6. 2024
- související předměty
    - SDIPR 20 kreditů, zapsat lze v dalším semestru vše naráz
        - přihlásit se k tématu v IS
    - SOBHA a SZMGR, zápis do 4. 3. 2024
- [LaTeX šablona](https://external_relations.pages.fi.muni.cz/document_templates/fithesis/#fithesis)
- příklad obhájené práce z FI MUNI: <https://is.muni.cz/th/olo1j/>
- [instrukce pro vedoucí a oponenty](https://www.fi.muni.cz/studies/bachelor-diploma-theses-advisors-opponents.html)
- existuje nějaké stipendium za diplomku v angličtině, nominuje vedoucí (relevantní pro jarní semestr 2024)
    - [stipendijní řád, Článek 16 - Stipendijní program na podporu závěrečných prací v anglickém jazyce](https://is.muni.cz/auth/do/fi/uredni_deska/predpisy/opatreni/Opatreni_dekana_13_2023_stipendijni_programy.pdf#page=12)

## Mohlo by se hodit

- úvod do EEG: https://www.ebme.co.uk/articles/clinical-engineering/introduction-to-eeg
- https://raamana.github.io/neuropredict/
- hezké návody na zpracování EEG dat v Pythonu, jsou tam i odkazy na videa: https://neuraldatascience.io/intro.html
- functional connectivity v Pythonu (uvažují MRI)
    - https://carpentries-incubator.github.io/SDC-BIDS-fMRI/07-functional-connectivity-analysis/index.html
    - https://peerherholz.github.io/workshop_weizmann/advanced/functional_connectivity.html#extracting-times-series-to-build-a-functional-connectome
- PCI python: https://github.com/noreun/pypci/blob/master/pci.py
- https://siibra-python.readthedocs.io/en/latest/autoapi/siibra/features/connectivity/index.html#module-siibra.features.connectivity
- https://netneurotools.readthedocs.io/en/latest/index.html
- https://dartbrains.org/content/Parcellations.html
- https://github.com/Davi1990/DissNet

## Metacentrum

- [OnDemand](https://ondemand.grid.cesnet.cz/pun/sys/dashboard/batch_connect/sys/bc_meta_jupyter/session_contexts/new)
- [Metacentrum wiki](https://wiki.metacentrum.cz/wiki/Main_Page)

#### Virtualenv pro OnDemand

```
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
python -m venv __venv__
source __venv__/bin/activate
pip install jupyter
module rm py-pip
python -m ipykernel install --user --name=__venv__ --env VIRTUAL_ENV "$VIRTUAL_ENV"
mv $VIRTUAL_ENV/bin/python $VIRTUAL_ENV/bin/python.orig
cat > $VIRTUAL_ENV/bin/python <<END
#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
PATH=$VIRTUAL_ENV/bin:$PATH
exec $VIRTUAL_ENV/bin/python.orig "\$@"
END
chmod a+x /storage/brno2/home/katterrina/__venv__/bin/python
```

## Datasety

- https://search.kg.ebrains.eu/instances/f16e449d-86e1-408b-9487-aa9d72e39901
    - Parcellation-based structural and resting-state functional brain connectomes of a healthy cohort (200 lidí z HCP)
    - alternativa? https://search.kg.ebrains.eu/?category=Dataset&q=structural%20connectivity%20schaefer%20200%207#3f179784-194d-4795-9d8d-301b524ca00a
- https://search.kg.ebrains.eu/instances/1570d4e5-8cc5-44b2-bfaa-f91274fe0bf3
    - F-TRACT (EEG epilepsie) 
- https://search.kg.ebrains.eu/instances/164b7564-1730-4b51-9c98-308005b620dd
    - Parcellation-based resting-state blood-oxygen-level-dependent (BOLD) signals of a healthy cohort (v1.0)

### Co jsem našla, zeptat se Honzy nebo dál prozkoumat

#### Strukturní data

- https://www.nature.com/articles/s41597-022-01596-9 - mají celou pipeline včetně kódu, změnit parcelaci by snad mělo být jednoduché
- https://github.com/MICA-MNI/micapipe - už tam mají Schaefer přímo v seznamu parcelací, tak by mělo být jednoduché ho použít

#### Článek *The spectral features of EEG responses to transcranial magnetic stimulation of the primary motor cortex depend on the amplitude of the motor evoked potentials*

- využívají bootstrap, stimulovali různá místa, zveřejnili data (<https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0184910&type=printable>)
    - data nejsou source reconstructed, ale jinak vypadají dobře, byly by tam ty stimulace jinde než v M1

#### Asi k ničemu

- https://openneuro.org/datasets/ds002094/versions/1.0.0 nějaká data bez popisků?
- https://gin.g-node.org/CIMeC/TMS-EEG_brain_connectivity_BIDS
- https://doi.org/10.1089/brain.2016.0462 data k tomu tady nejsou, ale kdybychom je našli, tak by to asi bylo super
- vůbec netuším, co to je, ale píšou, že data poskytnou na request https://github.com/thaivinhnguyen/EEGTMSfMRI
- https://github.com/Song-Yufei/tms-eeg-using-optimized-sham-outside-M1 píšou, že data někde na Zenedo
- https://www.nature.com/articles/s42003-020-0764-0#data-availability data pošlou na "reasonable request"
    - asi to není k ničemu, protože to není source-reconstructed? 
- https://github.com/Davi1990/EEG-and-DWI-metrics?tab=readme-ov-file - z článku od Momiho
    - matlab a nevím, jestli tam jsou i data nebo jenom kód (protože to je matlab, tak jsem to zatím neotevírala)
- https://zenodo.org/records/4990628 - asi ne, není source reconstructed
- http://www.tmslab.org/netconlab-perturbation.php - ze článku TMS-EEG: Individualized perturbation of the human connectome reveals reproducible biomarkers of network dynamics relevant to cognition

## Přístup k datům z HCP přímo

- přes prohlížeč to nějak nefunguje...
- [s3cmd](https://askubuntu.com/questions/202072/what-is-a-good-amazon-s3-client)
    - nakonfigutováno, přístupové údaje uloženy v config file
    - dokumentace `s3cmd --help` nebo <https://s3tools.org/usage>

## Poznámky, co bych mohla dělat

- zkusit podle notebooku od Honzy namapovat tu SC z F-TRACTu na TMS-EEG
- stáhnout pro F-TRACT přímo jejich data, ne ta z EBRAINS
- prozkoumat path transitivity - <https://www.biorxiv.org/content/10.1101/2023.02.09.527639v2.full.pdf>
- zjistit, jak využít **bootstrap** pro detekci ROI s významnou odezvou
    - v tomto případě asi nic moc, protože nemám raw měření
- začít psát text
    - co je to TMS/EEG, co je to strukturní konektivita
- prostudovat pořádně ten článek, který stimuloval i jiné lokace než M1
- protřídit a uhladit notebooky, přidat grafy a popisky
    - opravit ten notebook na source reconstruction
    - notebooky by možná mohly být užitečné někomu, kdo bude mít podobné problémy :D
- protřídit a opoznámkovat články v Zoteru
- dodělat statistiku pro F-TRACT tak, jak ji má fakt F-TRACT
    - parciální korelace - done, ale moc to nefunguje
    - bootstrap, connected a unconnected region pairs
- kouknout na Zenodo, které posílal Honza (ne na moc dlouho, asi z toho nic nebude)
- zachovat ve strukturní konektivitě vždy stejnou density
- https://pingouin-stats.org/build/html/generated/pingouin.distance_corr.html#pingouin.distance_corr

## Zjištění a co jsem zkusila

- z netneurotools se dají taky stáhnout nějaké FC a SC, ale nepíšou tam, v jaké je to parcelaci
    - povedlo se mi vystopovat zdroj a není to kompatibilní
- zamyslet se, jestli bych nemohla nějak použít modularitu (to by bylo cool :D)
    - zkusit predikci šíření pomocí modularity (modul kde to je 0, ostatní 1) - samo o sobě nefunguje, možná v kobinaci s něčím
- nějaké články
    - rešerše Brain network communication: concepts, models and applications
        - nejlepší zdroj k teorii k diplomce
        - přečíst ještě jednou pořádně a rovnou z toho dělat výpisky na text práce
            - ve zdrojích mají u článků napsáno, co tam zjistili <3
    - The Prediction of Brain Activity from Connectivity: Advances and Applications
        - zajímavé, ale predikují activation maps a taky mají data všechna per subject, ne průměrnou strukturní a funkční konektiivtu

## Oficiální zadání

The primary concern of the thesis are the spatio-temporal patterns of brain activity evoked by direct stimulation. The objective is to explore suitable methodologies rooted in complex network analysis and compare their applicability in empirical and simulated electroencephalography recordings of responses to stimulation. The entry point to this topic will be the recent results on metrics based on network communication models capturing some of the relationship between structural connectivity and stimulus response. The original work, done on invasive intracranial data in a large cohort, serves as a basis for evaluating the extent to which these observations can be replicated in noninvasive stimulation and recordings. The student will engage with openly available datasets, namely the summary data of the Functional Brain Tractography project (F-TRACT), which provides the response probabilities and amplitudes between brain regions from intracranial recordings. Additionally, the student will utilize open transcranial magnetic stimulation datasets available in EBRAINS and other data sharing platforms. The student will familiarize herself with the aforementioned datasets and the network communication models and methods to characterize stimulus response complexity. The main focus of the student's work will be on the implementation and modification of the network communication models methodology, followed the iterative application to the non-invasive data. Subsequently, she will assess any eventual discrepancies to the results from invasive datasets, designing the appropriate approach and statistical methodology for the evaluation. Given the intricacy and depth of the background topic, the student will collaborate with experts at CEITEC MU to seek guidance on the respective aspects of the data and the surrounding neuroscientific context.