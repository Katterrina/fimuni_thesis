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

## Dlaší datasety

- https://search.kg.ebrains.eu/instances/f16e449d-86e1-408b-9487-aa9d72e39901
    - Parcellation-based structural and resting-state functional brain connectomes of a healthy cohort (200 lidí z HCP)
    - alternativa? https://search.kg.ebrains.eu/?category=Dataset&q=structural%20connectivity%20schaefer%20200%207#3f179784-194d-4795-9d8d-301b524ca00a
- https://search.kg.ebrains.eu/instances/1570d4e5-8cc5-44b2-bfaa-f91274fe0bf3
    - F-TRACT (EEG epilepsie) 
- https://search.kg.ebrains.eu/instances/164b7564-1730-4b51-9c98-308005b620dd
    - Parcellation-based resting-state blood-oxygen-level-dependent (BOLD) signals of a healthy cohort (v1.0)

### Co jsem našla, zeptat se Honzy nebo dál prozkoumat

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

## Přístup k datům z HCP

- přes prohlížeč to nějak nefunguje...
- [s3cmd](https://askubuntu.com/questions/202072/what-is-a-good-amazon-s3-client)
    - nakonfigutováno, přístupové údaje uloženy v config file
    - dokumentace `s3cmd --help` nebo <https://s3tools.org/usage>

## Aktuální otázky a TODO

- teď dělám korelaci jednoho stimulovaného místa se strukturní konektivitou - když bych udělala to samé s F-Tractem, taky jen po jednotlivých řádcích, vyšlo by mi to stále korelované?
    - zkusila jsem, vyšlo
- path transitivity - <https://www.biorxiv.org/content/10.1101/2023.02.09.527639v2.full.pdf>


## Oficiální zadání

- TODO zkopírovat z ISu

This thesis will explore methods for complex network based analysis of spatio-temporal patterns of propagation of brain activity evoked by non-invasive stimulation methods. In particular, the relationship of the responses to transcranial magnetic stimulation with the brain structural connectivity and the maps of structural and functional properties of the cortex will be explored using techniques for both static network analysis and communication dynamics. The work will be performed on open datasets.