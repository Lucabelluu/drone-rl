# Diario di progetto — drone-rl

Progetto sperimentale Deep Learning (SC/0081), UniCA 2026.
Ambiente: gym-pybullet-drones (Reinforcement Learning).

## Domande di ricerca

1. Confronto tra algoritmi (Memoria vs. Reattività): In un task di hovering spaziale, un algoritmo off-policy basato sulla memorizzazione dell'esperienza passata garantisce un tasso di schianti (Crash Rate) inferiore e una convergenza più stabile rispetto a un algoritmo on-policy puramente reattivo?

2. Il "prezzo" della fluidità (Reward Shaping): Applicando la tecnica del Reward Shaping all'algoritmo vincitore della Domanda 1, l'introduzione di una penalità quadratica per le variazioni brusche di potenza dei motori produce un volo stazionario significativamente più fluido, e quanto incide questo vincolo sul tempo necessario a raggiungere il target?

3. Robustezza e Domain Randomization (L'aggiunta delle turbolenze): Il modello ottimizzato per la fluidità (vincitore Domanda 2) è in grado di resistere a turbolenze esterne stocastiche (vento) introdotte in fase di valutazione, o è necessario addestrare una nuova policy iniettando disturbi fisici casuali durante il training per evitare la perdita di assetto?

---

## [15-06-2026] — Setup iniziale

- Hardware: MacBook Air M5, macOS.
- Git 2.50.1 installato. Account GitHub presente.
- Command Line Tools presenti (compilazione dipendenze native OK).
- Cartella progetto: /Users/lucabellu/drone-rl (rinominata senza spazi).

### Decisioni prese
- Ambiente di training: locale primario (Mac), Colab di riserva per run lunghi/paralleli.
  Motivo: PyBullet è CPU-bound, la GPU di Colab non accelera la simulazione. Su Colab → runtime CPU.
- Python di sistema (3.9) NON toccato. Si userà un ambiente isolato con Python 3.11.
- Versionamento: git locale + repo GitHub pubblico (sarà il canale di consegna).
- Diario ufficiale: questo file, versionato nel repo.

### Fase corrente
- Prova semplice richiesta dal prof: verificare che l'ambiente gym-pybullet-drones funzioni.
- Linee guida specifiche del progetto: ancora da ricevere (verranno caricate a breve).

## 15-06-2026 — Versionamento git inizializzato

- Inizializzato repository git locale nella cartella drone-rl (`git init`).
- Creato `.gitignore` per escludere: ambiente virtuale, cache Python (__pycache__),
  file di sistema macOS (.DS_Store), checkpoint notebook, file modello/dati pesanti.
- Primo commit effettuato: "Setup iniziale: diario di progetto e gitignore".
- Identità git configurata: Luca Bellu / l.bellu05@gmail.com.
- Stato: repository locale attivo. GitHub remoto ancora da collegare.

## 15-06-2026 — Repository collegato a GitHub

- Creato repository remoto pubblico: github.com/Lucabelluu/drone-rl (vuoto, senza README/licenza per evitare conflitti).
- Collegato il locale al remoto: `git remote add origin <url>` (remote chiamato "origin").
- Primo push effettuato: `git push -u origin main`. Branch main locale agganciato a origin/main.
- Autenticazione GitHub via Personal Access Token (classic, scope "repo"), memorizzato nel Portachiavi macOS. Token NON versionato.
- Stato: progetto versionato in locale e sincronizzato su GitHub. Pronto per il setup dell'ambiente Python.

## 15-06-2026 — Ambiente Python isolato creato

- Installato Miniforge3 (conda/mamba) per Apple Silicon (arm64) in ~/miniforge3.
  Motivo: su Mac M-series conda gestisce meglio le librerie scientifiche con componenti
  native (es. PyBullet), riducendo errori di compilazione rispetto a venv+pip puro.
- Creato ambiente conda dedicato "drone-rl" con Python 3.11.15.
  Motivo della versione: 3.11 è il punto di equilibrio tra supporto e stabilità per lo
  stack RL (gym-pybullet-drones, stable-baselines3); le 3.12/3.13 possono dare problemi
  di compatibilità con librerie scientifiche.
- Python di sistema (3.9) lasciato intatto. Ambiente di lavoro: sempre `conda activate drone-rl`.
- Stato: ambiente pronto. Prossimo passo: installazione di gym-pybullet-drones.

## 15-06-2026 — Correzione versione Python + strategia implementativa

- CORREZIONE: ambiente ricreato con Python 3.10.20 (era 3.11).
  Motivo: la documentazione ufficiale di gym-pybullet-drones (utiasDSL) raccomanda Python 3.10.
  Allinearsi alla versione ufficiale riduce il rischio di incompatibilità.

- STRATEGIA (da mail del prof, 15-06-2026): il progetto NON richiede di re-implementare
  gli algoritmi RL da zero. Approccio di qualità = comprendere, estendere e modificare
  implementazioni esistenti rispetto alle 3 domande di ricerca.
- Repository di riferimento per i controllori RL: safe-control-gym (learnsyslab),
  che include già PPO (on-policy) e DDPG (off-policy) — direttamente utili alla Domanda 1.
- Divisione del lavoro:
  * gym-pybullet-drones = l'ambiente/simulatore. Si usa, non si studia a fondo (tranne
    modifiche mirate, es. vento per Domanda 3).
  * safe-control-gym = il codice dei controllori RL. QUESTO va studiato, capito e modificato.
- Prossimo: verificare il funzionamento dell'ambiente con script di esempio (richiesta del prof).
- Nota: sfruttare le sessioni di mentoring (giovedì 09-11) per i punti critici, come suggerito dal prof.

## 15-06-2026 — Installazione ambiente e prova di funzionamento (RICHIESTA PROF)

### Installazione gym-pybullet-drones
- Clonato repo ufficiale utiasDSL/gym-pybullet-drones in ~/tools/ (FUORI dal repo di progetto,
  per tenere il repo pulito: è dipendenza di terzi, gestita via requirements, non codice nostro).
- Versione installata: gym-pybullet-drones 2.1.0.

### Problema incontrato e risoluzione (rilevante per l'esame)
- `pip install -e .` falliva: pybullet non compilava (errore clang exit code 1).
- Causa: clang di sistema molto recente (Apple clang 21.0.0, macOS darwin25) troppo severo
  per il codice C++ datato di pybullet.
- Soluzione: installato pybullet PRE-COMPILATO via conda (`conda install -c conda-forge pybullet`),
  evitando la compilazione locale. Poi installato gym-pybullet-drones con `--no-deps` e le restanti
  dipendenze a mano via pip, saltando pybullet.
- Warning residuo: pybullet 3.2.5 (conda) vs 3.2.7 richiesto. Verificato sul campo che è innocuo:
  gli script di esempio funzionano correttamente.

### Dipendenze principali installate
- pybullet 3.2.5 (conda), numpy 2.2, gymnasium 1.2.3, stable-baselines3 2.8.0, torch 2.12.0,
  scipy, matplotlib, control, transforms3d, pytest.

### Prova di funzionamento (test richiesto dal prof)
- Eseguito `examples/pid.py`: finestra 3D aperta correttamente, droni in volo controllati da PID,
  grafici finali generati. Log di posizione/assetto/velocità regolari.
- ESITO: ambiente RL pienamente funzionante. Pronti per la proposta di progetto.

## 15-06-2026 — Seconda prova: esempio RL (learn.py)

- Eseguito `examples/learn.py`: addestramento RL completato senza errori.
- L'esempio usa PPO (on-policy). Osservata curva Episode Reward: converge verso ~470 ma con
  forti instabilità intermedie (crolli a ~150 e recuperi). Coerente con l'ipotesi della Domanda 1
  (on-policy = convergenza meno stabile). NB: singolo run, non significativo statisticamente.
- Comportamento del drone addestrato: hover a z≈1.0 raggiunto e mantenuto (x,y≈0).
- Osservati gli RPM dei 4 motori oscillanti → rilevante per la Domanda 2 (fluidità / reward shaping).
- ESITO: ambiente RL pienamente operativo, inclusa la pipeline di training e visualizzazione.
- Salvati screenshot dei due grafici in assets/ per riferimento futuro (curva reward + stati drone).