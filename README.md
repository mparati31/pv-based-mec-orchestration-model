# pv-based-mec-orchestration-model

Questo repository contiene il lavoro di tesi svolto per la laurea triennale in informatica.
La tesi è reperibile al [seguente link](https://github.com/mparati31/Tesi).

## Struttura

La cartella [/data](https://github.com/mparati31/pv-based-mec-orchestration-model/tree/main/data) contiene i dati usati per generare le istanze di test, gli *split points* per gli algoritmi euristici e i risultati ottenuti.
Il codice è contenuto nella cartella [/pycode](https://github.com/mparati31/pv-based-mec-orchestration-model/tree/main/pycode), suddiviso in due parti:

- [dataset](https://github.com/mparati31/pv-based-mec-orchestration-model/tree/main/pycode/dataset): contiene i moduli per caricare i dati e generare le istanze di test. Il tutto è accessibile usando le funzioni presenti nel modulo [api.py](https://github.com/mparati31/pv-based-mec-orchestration-model/blob/main/pycode/dataset/api.py);
- [src](https://github.com/mparati31/pv-based-mec-orchestration-model/tree/main/pycode/src): contiene le classi che rappresentano le istanze e i risultati, mentre la cartella
  [models](https://github.com/mparati31/pv-based-mec-orchestration-model/tree/main/pycode/src/models) implementa i modelli di ottimizzazione e gli algoritmi risolutivi euristici.

