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

## 16-06-2026 — DQ1: algoritmi, ruolo di safe-control-gym, definizione di Crash Rate

### Ambiente e strumenti (deciso)
- Ambiente unico del progetto: gym-pybullet-drones, task hovering (volo stazionario a
  z = 1.0 m). Già installato e funzionante.
- Algoritmi presi da stable-baselines3 (SB3), già installato (v2.8.0). SB3 contiene già
  PPO, SAC, DDPG, TD3 → nessuna nuova installazione necessaria.
- safe-control-gym: usato SOLO come riferimento di studio per capire come sono
  implementati gli algoritmi RL; NON è il runtime e non se ne copia il codice.
  Motivo: è un ambiente DIVERSO (le linee guida ammettono un solo ambiente) e per
  rispettare la regola "<15% di codice scritto da altri".

### Concetti chiave (da saper spiegare all'esame)
- policy = la strategia dell'agente: vista la situazione (posizione, assetto, velocità
  del drone) decide l'azione (potenza ai motori). "Imparare" = migliorare la policy.
- on-policy = impara solo dall'esperienza appena vissuta, poi la scarta (nessuna
  memoria) → "reattivo".
- off-policy = tiene un replay buffer (memoria delle esperienze passate) e lo ripassa
  più volte, riusando anche esperienze vecchie → "con memoria".
- Conseguenza pratica: l'off-policy di solito impara con MENO passi di simulazione
  (più efficiente coi dati) → vantaggio su Mac CPU-bound.

### DQ1 — algoritmi scelti (deciso)
- on-policy: PPO (Proximal Policy Optimization). Migliora la strategia a piccoli passi
  prudenti → stabile. È già l'algoritmo usato in learn.py.
- off-policy: SAC (Soft Actor-Critic). Stabile, efficiente coi dati, già usato come
  baseline off-policy sull'hovering nel paper dell'ambiente.
- Scelto SAC al posto di DDPG (che il prof aveva citato) perché DDPG è noto per essere
  instabile: come unico campione off-policy rischierebbe di far apparire l'off-policy
  "meno stabile" per un suo difetto, falsando l'ipotesi della DQ1. La DQ1 dice "un
  algoritmo off-policy" senza nominare DDPG → scelta libera e motivabile.
- Cautela onesta: SAC differisce da PPO anche per la "morbidezza" (mantiene un po' di
  casualità nelle azioni), quindi il confronto non isola PERFETTAMENTE la sola variabile
  on/off-policy. Accettabile per un progetto universitario, purché dichiarato.

### DQ1 — metrica 1: Crash Rate (deciso)
- Un episodio = un tentativo di volo. Finisce con due "bandierine" Gymnasium:
  terminated (fine legata al compito) oppure truncated (episodio tagliato).
- "truncated" mescola due casi diversi: (a) drone fuori controllo = troppo lontano dal
  target o inclinato oltre l'angolo di sicurezza → SCHIANTO; (b) tempo scaduto mentre
  volava ancora bene → NON schianto.
- Definizione: Crash Rate = (episodi finiti in schianto) / (episodi di valutazione
  totali), in %. Misurato su N episodi (es. 50–100) con la policy addestrata.
- NON si conta semplicemente "truncated == True": si guarda lo stato finale per capire
  se era fuori controllo (schianto) o solo tempo scaduto.
- I valori soglia esatti (distanza/angolo) si prenderanno dai limiti di sicurezza già
  presenti dentro HoverAviary in fase di implementazione (così non sono arbitrari).
- Buona pratica: registrare per ogni episodio il motivo della fine (schianto / tempo
  scaduto / target raggiunto) per trasparenza.

  ### DQ1 — metrica 2: convergenza stabile (deciso)
- Curva di apprendimento = ricompensa media per episodio (asse Y) lungo il tempo di
  addestramento (asse X). Convergenza = la curva sale e si assesta in alto.
- Convergenza stabile = curva regolare che resta su; instabile = forti oscillazioni,
  cali bruschi o crollo dopo un buon valore (es. già osservato con PPO: ~470 con
  crolli a ~150 e recuperi).
- "Stabilità" = poca variabilità, con due facce: quanto la curva traballa dentro un
  singolo addestramento, e quanto cambia il risultato rifacendo l'addestramento con
  casualità diversa.
- seed = numero che fissa la casualità di un run (posizione iniziale, esplorazione,
  inizializzazione della rete). Un solo run può essere fortuna o sfortuna: per misurare
  la stabilità servono più run con seed diversi.
- Definizione operativa, due numeri:
  1. stabilità tra run = deviazione standard della prestazione finale tra i seed
     (deviazione standard = quanto i numeri sono sparpagliati; piccola = stabile);
  2. regolarità entro il run = deviazione standard della ricompensa nell'ultimo tratto
     di addestramento, mediata sui seed (piccola = curva liscia, senza crolli).
- Presentazione standard: curva media dei seed con una fascia ± una deviazione standard
  attorno (fascia stretta = convergenza stabile).
- Principio deciso: addestramento multi-seed obbligatorio per questa metrica
  (minimo 3, idealmente 5).

  ### DQ1 — formulazione finale e piano sperimentale (deciso)
Formulazione definitiva:
"In un task di hovering (volo stazionario a quota fissa), un algoritmo off-policy basato
sulla memorizzazione dell'esperienza passata ottiene un tasso di schianti (Crash Rate)
inferiore e una convergenza più stabile rispetto a un algoritmo on-policy puramente reattivo?"
(Rispetto alla prima stesura: "garantisce" → "ottiene" perché una domanda di ricerca osserva
una tendenza, non garantisce; "hovering spaziale" → "hovering (volo stazionario a quota fissa)".
La domanda resta generale: PPO/SAC stanno nei metodi, non nella domanda.)

Piano sperimentale:
- Metodi: confronto PPO (on-policy) vs SAC (off-policy), entrambi da stable-baselines3, stesso
  ambiente HoverAviary, STESSO budget di addestramento (stesse interazioni con l'ambiente) per
  un confronto equo.
- Attività: addestrare PPO e SAC con più seed (min 3, idealmente 5), stessi passi; salvare le
  curve di apprendimento; valutare ogni policy su N episodi (es. 50–100) registrando il motivo
  di fine di ciascun episodio.
- Metriche: Crash Rate (definito sopra) + stabilità di convergenza (definita sopra).
- Ipotesi: SAC ottiene Crash Rate inferiore e convergenza più stabile di PPO; la conclusione
  dipende dal supporto dei dati su più seed.
- Nota di metodo: "stesso budget" = pari numero di passi; SAC, più efficiente coi dati, può
  avvantaggiarsi, ma è un vantaggio legittimo dell'off-policy (da dichiarare).