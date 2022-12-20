In questa cartella sono contenuti i risultati delle computazioni.

La cartella ./static_energy_profiles/ contiene i risultati delle istanze con profili
energetici statici.
I file sono nella forma:
    result_{dataset}_{numero time-slots}_{istanza}_{profilo energetico}.csv.

La cartella ./distances_based_energy_profiles/ contiene invece tutti i risultati delle
istanze con profili energetici basati sulla distanza delle facility dal centro.
I file sono suddivisi in base a come sono stati computati, quindi se usando il modello
normale oppure tramite un'euristica.
I file sono nella forma:
    result_{dataset}_{numero time-slots}_{istanza}_{profilo energetico riferimento}_{funzione distanza}.csv
ed ognuno è contenuto nella cartella che ha come nome il mese utilizzato per l'irraggiamento.

In entrambe le cartelle è presente un file stats.csv che contiene le statistiche di
computazione delle varie istanze.
