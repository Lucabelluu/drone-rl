# StableHover — Reinforcement Learning per l'hovering di un quadrirotore

**Autore:** Luca Bellu (matricola 60/79/00295)
**Corso:** Deep Learning (SC/0081) — Università degli Studi di Cagliari, A.A. 2025/2026

## Panoramica

StableHover è uno studio sperimentale di Reinforcement Learning sul controllo del volo stazionario (hovering) di un quadrirotore, condotto nell'ambiente di simulazione gym-pybullet-drones. Il lavoro è organizzato attorno a tre domande di ricerca che ne scandiscono le fasi: il confronto tra una famiglia di algoritmi puramente reattivi (on-policy) e una basata sulla memorizzazione dell'esperienza passata (off-policy); l'effetto del reward shaping sulla fluidità del volo e il suo costo in termini di rapidità nel raggiungere il target; la robustezza del modello a turbolenze esterne e il ruolo della domain randomization in addestramento.

Gli algoritmi di apprendimento (PPO e SAC) provengono dalla libreria stable-baselines3; l'ambiente, la dinamica del drone e le convenzioni di osservazione e azione provengono da gym-pybullet-drones. Il contributo originale del progetto è nel disegno sperimentale, nelle modifiche all'ambiente necessarie a un confronto corretto, nella metodologia di valutazione e nell'analisi dei risultati, descritti nelle sezioni seguenti.

## Domande di ricerca

1. **Confronto tra algoritmi (Memoria vs. Reattività):** In un task di hovering spaziale, un algoritmo off-policy basato sulla memorizzazione dell'esperienza passata garantisce un tasso di schianti (Crash Rate) inferiore e una convergenza più stabile rispetto a un algoritmo on-policy puramente reattivo?
2. **Il "prezzo" della fluidità (Reward Shaping):** Applicando la tecnica del Reward Shaping all'algoritmo vincitore della Domanda 1, l'introduzione di una penalità quadratica per le variazioni brusche di potenza dei motori produce un volo stazionario significativamente più fluido, e quanto incide questo vincolo sul tempo necessario a raggiungere il target?
3. **Robustezza e Domain Randomization (L'aggiunta delle turbolenze):** Il modello ottimizzato per la fluidità (vincitore Domanda 2) è in grado di resistere a turbolenze esterne stocastiche (vento) introdotte in fase di valutazione, o è necessario addestrare una nuova policy iniettando disturbi fisici casuali durante il training per evitare la perdita di assetto?

## Ambiente e installazione

Il progetto è stato sviluppato su macOS con Apple Silicon, con gestione dell'ambiente tramite conda (miniforge) e Python 3.10. La 3.10 è la versione raccomandata dalla documentazione di gym-pybullet-drones.

L'ambiente di simulazione è una dipendenza di terzi e viene installato in una cartella separata dal progetto, per mantenere il repository contenente solo codice originale.

**Prerequisiti:** conda (miniforge o miniconda) e git.

### Procedura (macOS Apple Silicon)

Su Apple Silicon l'installazione di pybullet via pip fallisce in compilazione con i compilatori recenti di macOS; si installa quindi il binario pre-compilato da conda-forge prima delle altre dipendenze.

```bash
# 1. Ambiente isolato con Python 3.10
conda create -n drone-rl python=3.10
conda activate drone-rl

# 2. Clona il simulatore (dipendenza di terzi) in una cartella separata
mkdir -p ~/tools && cd ~/tools
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
git checkout 9bc12bc          # versione esatta usata nel progetto

# 3. pybullet pre-compilato PRIMA del resto (evita l'errore di compilazione clang)
conda install -c conda-forge pybullet

# 4. Strumento di build richiesto dal pacchetto
pip install poetry-core

# 5. Installa il simulatore senza ricompilare le dipendenze (pybullet è già presente)
pip install -e . --no-build-isolation --no-deps

# 6. Restanti dipendenze (pybullet escluso, già installato via conda)
pip install "numpy>=2.2" "scipy>=1.15" "transforms3d>=0.4" "matplotlib>=3.10" "gymnasium>=1.2" "stable-baselines3>=2.8" "control>=0.10.2" "pytest>=9.0" "jupyter"

# 7. Verifica
python -c "import gym_pybullet_drones, stable_baselines3, gymnasium; print('OK')"
```

I percorsi (`~/tools/`) e il nome dell'ambiente (`drone-rl`) sono le scelte adottate in questo progetto e possono essere adattati. Su piattaforme diverse da Apple Silicon è in genere sufficiente la procedura ufficiale del simulatore (`git clone` seguito da `pip install -e .`); la sequenza sopra è quella verificata per macOS Apple Silicon, dove il metodo standard fallisce.

Il file `requirements.txt` riporta le versioni esatte dei pacchetti con cui il progetto è stato eseguito. Nota: conda installa pybullet 3.2.5, leggermente precedente alla 3.2.7 indicata dal pacchetto; la differenza produce solo un avviso, verificato innocuo.

## Struttura della repository

```
drone-rl/
├── src/
│   ├── envs/
│   │   └── hover_terminal.py     # variante dell'ambiente con schianto = stato terminale
│   ├── train.py                  # addestramento di una singola policy (1 algoritmo, 1 seed)
│   └── evaluate.py               # valutazione: Crash Rate sotto perturbazioni graduate
├── scripts/
│   ├── run_dq1.sh                # addestra i 6 modelli della Domanda 1
│   ├── run_eval_dq1.sh           # valutazione a condizioni nominali
│   ├── run_eval_sweep_dq1.sh     # valutazione su 5 livelli di severità
│   └── aggregate_eval.py         # raccoglie i riepiloghi JSON in un unico CSV
├── experiments/
│   └── dq1/                      # artefatti dei run e risultati aggregati della Domanda 1
├── notebooks/
│   └── dq1_confronto_ppo_sac.ipynb   # analisi dei risultati della Domanda 1
├── results/
│   ├── figures/                  # figure finali (.png)
│   └── tables/                   # tabelle finali (.csv)
├── assets/                       # logo, proposta di progetto, immagini di calibrazione e setup
├── requirements.txt
├── DIARIO.md                     # diario di progetto: decisioni e motivazioni in ordine cronologico
└── README.md
```

Il codice è separato per responsabilità. `src/` contiene la logica riutilizzabile: l'ambiente di simulazione e i due programmi a riga di comando per addestramento e valutazione, ciascuno parametrico su algoritmo e seed. `scripts/` contiene gli script di orchestrazione che compongono gli esperimenti di una domanda di ricerca, lanciando i programmi di `src/` con le combinazioni previste dal disegno sperimentale. La separazione tiene distinti il "come si esegue una singola operazione" (in `src/`) e il "quali operazioni compongono un esperimento" (in `scripts/`).

Gli artefatti generati dagli esperimenti — modelli, normalizzatori, valutazioni periodiche, riepiloghi — sono raccolti sotto `experiments/`, organizzati per domanda di ricerca e per run. Le figure e le tabelle definitive, prodotte dai notebook a partire da quegli artefatti, sono esportate sotto `results/` come file riutilizzabili. I notebook in `notebooks/` svolgono esclusivamente l'analisi: leggono gli artefatti e ne ricavano le visualizzazioni, senza eseguire addestramenti.

I modelli addestrati e i file di log non sono versionati, perché pesanti e rigenerabili dall'esecuzione degli esperimenti; figure e tabelle finali sono invece incluse nel repository.

## Metodologia sperimentale

Le scelte metodologiche seguenti sono comuni a tutte le domande di ricerca e garantiscono confronti equi e riproducibili.

Ogni configurazione è addestrata con **tre semi casuali** (0, 1, 2) e i risultati sono riportati come media e deviazione standard sui tre semi: la deviazione standard misura quanto un esito dipenda dal seme, cioè la riproducibilità dell'addestramento. Le condizioni messe a confronto ricevono sempre lo **stesso budget** di addestramento e non si applica early stopping: tutti i modelli vengono addestrati per il budget pieno, così che eventuali instabilità tardive non restino nascoste. Le valutazioni usano il **modello finale** di ogni run, non il checkpoint migliore, per non selezionare l'istante più favorevole.

Le condizioni di valutazione sono generate con un seme fisso e sono **identiche per tutti i modelli** confrontati: a parità di scenario, le differenze osservate dipendono dalla policy e non dalla variabilità campionaria. Ogni run salva un `config.json` con algoritmo, seme, budget, versioni delle librerie e hash del commit, per legare ciascun risultato al codice che lo ha prodotto.

Il confronto tra algoritmi è inteso a livello di **famiglie** (on-policy e off-policy), con PPO e SAC come rappresentanti: le conclusioni riguardano le famiglie, non i due algoritmi specifici presi isolatamente.

## Domanda 1 — Confronto PPO vs SAC

### Impostazione

Il confronto oppone PPO (on-policy, reattivo) e SAC (off-policy, dotato di replay buffer) sul task di hovering di `HoverAviary`, con osservazione cinematica (`kin`) e azione sui giri dei quattro motori (`rpm`), per un budget di 500.000 passi e tre semi.

Il task richiede una modifica all'ambiente. In `HoverAviary` l'uscita dall'inviluppo di volo sicuro è segnalata come troncamento (`truncated`); per un algoritmo off-policy questo è scorretto, perché induce a stimare un valore futuro non nullo per stati di schianto, da cui una divergenza delle stime di valore. La variante `HoverAviaryTerminal` (in `src/envs/`) tratta lo schianto come stato terminale (`terminated`), eliminando il problema. La correzione è applicata a entrambi gli algoritmi.

Le osservazioni sono normalizzate con statistiche correnti (medie e varianze), condizione necessaria alla stabilità di SAC su questo ambiente; la ricompensa è lasciata grezza per mantenere i ritorni interpretabili e confrontabili. Per SAC il coefficiente di entropia è fissato a 0.1, poiché la sua regolazione automatica diverge in questo contesto; PPO usa i valori predefiniti. Il funzionamento dettagliato è documentato nel codice e nei commenti dei rispettivi file.

Poiché a condizioni nominali entrambi gli algoritmi raggiungono lo 0% di schianti (la metrica non discrimina), la robustezza è misurata sottoponendo i modelli a condizioni iniziali perturbate di severità crescente su cinque livelli (offset di posizione, inclinazione e impulso di velocità a inizio episodio), con limiti che evitano partenze già oltre la soglia di schianto.

### Riproduzione

Dalla radice del progetto, con l'ambiente attivo:

```bash
conda activate drone-rl

# 1. Addestramento dei 6 modelli (~110 min su MacBook Air M5: PPO ~3 min/run, SAC ~33 min/run)
caffeinate -i bash scripts/run_dq1.sh

# 2. Valutazione di robustezza su 5 livelli di severità (pochi minuti)
caffeinate -i bash scripts/run_eval_sweep_dq1.sh

# 3. Aggregazione dei riepiloghi in un unico CSV
python scripts/aggregate_eval.py

# 4. Esecuzione del notebook di analisi: rigenera ed esporta figure e tabelle in results/
jupyter nbconvert --to notebook --execute --inplace notebooks/dq1_confronto_ppo_sac.ipynb
```

Le figure e le tabelle finali sono già incluse in `results/`. Per rigenerare solo le figure è sufficiente il passo 4, che legge i dati già versionati (`evaluations.npz` e `eval_sweep_results.csv`); per rigenerare l'intera catena a partire dai modelli occorre ripetere i passi 1–4, poiché i modelli addestrati non sono versionati.

### Risultati

![Crash Rate in funzione della severità](results/figures/dq1_crashrate_vs_severity.png)

![Curve di convergenza](results/figures/dq1_convergence.png)

SAC presenta un Crash Rate inferiore a ogni livello di severità, con bande tra semi che non si sovrappongono a quelle di PPO (a severità 3, circa 8% contro 48%) e una variabilità tra semi più contenuta. Sul fronte della convergenza i due algoritmi raggiungono lo stesso livello di prestazione a regime (ritorno ~470), ma SAC vi arriva prima — supera la soglia di ricompensa di 400 a circa 60.000 passi contro i circa 110.000 di PPO — e con una fase di salita più stabile tra semi.

La risposta alla domanda è affermativa su entrambi gli assi considerati: l'algoritmo off-policy ottiene un Crash Rate inferiore e una convergenza più stabile. Il vantaggio si colloca nella velocità di convergenza, nella stabilità e nella robustezza, non nella prestazione finale a regime, che è equivalente per le due famiglie. SAC è quindi il modello adottato come base per la Domanda 2.

## Crediti e riferimenti

Il progetto si appoggia a due strumenti open source di terzi, di cui utilizza le implementazioni senza modificarne il funzionamento interno:

- **gym-pybullet-drones** — ambiente di simulazione del quadrirotore, dinamica fisica (basata su PyBullet) e convenzioni di osservazione e azione. Repository ufficiale: https://github.com/utiasDSL/gym-pybullet-drones (versione utilizzata: commit `9bc12bc`). Riferimento: J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, A. P. Schoellig, "Learning to Fly—a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control", IROS 2021.
- **Stable-Baselines3** — implementazioni degli algoritmi di Reinforcement Learning PPO e SAC utilizzati nel confronto. Riferimento: A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations", Journal of Machine Learning Research, 2021.

Il contributo originale di questo progetto consiste nel disegno sperimentale, nella variante dell'ambiente con terminazione corretta per gli algoritmi off-policy, nella metodologia di valutazione della robustezza, nel reward shaping e nell'analisi dei risultati.

L'ambiente di simulazione e le librerie di terzi sono distribuiti dai rispettivi autori sotto le proprie licenze; si rimanda ai repository ufficiali per i termini.