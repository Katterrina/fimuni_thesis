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

## Dlaší datasety

- https://search.kg.ebrains.eu/instances/f16e449d-86e1-408b-9487-aa9d72e39901
- https://search.kg.ebrains.eu/instances/1570d4e5-8cc5-44b2-bfaa-f91274fe0bf3
- https://search.kg.ebrains.eu/instances/164b7564-1730-4b51-9c98-308005b620dd
exec $VIRTUAL_ENV/bin/python.orig "\$@"
END
chmod a+x /storage/brno2/home/katterrina/__venv__/bin/python
```
