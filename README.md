# nooddrinkwater_locaties_distributie

Hey Hackathon deelnemer! Hierbij de repository die we belooft hadden, hieronder een overzicht van de opbouw van de repository.
Hij is opgezet in de vorm van de data-science cookie cutter. Hoe dan ook, je vind onze notebook 'location_picker' in de 'notebooks' folder.

In deze notebook wordt alles geregeld, van het installeren van de package die nodig zijn voor deze tool, het downloaden en inladen van alle opensource data tot aan het genereren van geadviseerde locaties.
In het notebook zijn een aantal variabelen gevuld met "[VUL HIER x IN]", hier moet je bijvoorbeeld aangeven voor welke gemeente je gaat draaien, ik wil je vragen op "VUL HIER" te zoeken en alle velden aan te vullen voordat je het notebook draait. Dit voorkomt dat je tegen problemen aanloopt.

Om te voorkomen dat jullie allen los data gaan opvragen via API's hebben wij het een en ander klaargezet in een Sharepoint omgeving: https://ifvportal.sharepoint.com/sites/hackathonnipvnooddrinkwater/SitePages/Home.aspx

**LET OP: de eerste keer dat je het nodebook draait kan het zijn dat het vrij lang duurt, dit komt omdat alle data nog verzamelt en geprepareerd moet worden. Om dit te voorkomen hebben we wat bestanden in Sharepoint klaargezet**
**LET OP: in notebooks/eda_support_files/CONSTANTS.py vind je een pad wat je moet invullen voor jouw pad**

```
├── README.md          <- The top-level README for developers using this project.
|
├── data               <- THIS IS WHERE YOU WILL FIND ALL DATA YOU WILL GET AND USE IN THE NOTEBOOKS
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- HERE IS WHERE YOU FIND THE NOTEBOOK TO RUN FOR
│   └── eda_support_files       <- THESE ARE THE SUPPORT FILES OF THE NOTEBOOK, ALL IN PYTHON
│       └── modelling           <- THESE ARE THE SUPPORT FILES OF THE NOTEBOOK, SPECIFICALLY FOR THE OPTIMISATION ALGORITHM PART
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
```

--------

