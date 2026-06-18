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

  ### Spazio d'azione del progetto (deciso, vale per DQ1–DQ2–DQ3)
- Scelto lo spazio d'azione `rpm`: la policy comanda i 4 motori in modo INDIPENDENTE
  (4 valori di RPM), non `one_d_rpm` (un solo valore uguale per tutti).
- Motivi:
  * DQ2: la penalità è sulle "variazioni di potenza dei motori" → l'azione deve ESSERE la
    potenza dei 4 motori.
  * DQ3: per resistere al vento il drone deve inclinarsi, e per inclinarsi servono potenze
    diverse sui 4 motori. Con `one_d_rpm` (tutti uguali) può solo salire/scendere e NON
    potrebbe contrastare una raffica laterale → la DQ3 sarebbe senza risposta possibile.
  * La stessa policy attraversa DQ1→DQ2→DQ3: lo spazio d'azione si sceglie una volta sola.
- Costo: addestrare 4 motori indipendenti è molto più lento (ore, non minuti).
  Mitigazione: Colab di riserva per i run pesanti; se serve, 3 seed invece di 5.

### DQ2 — Reward Shaping per la fluidità (deciso)
Formulazione (invariata, già ben posta):
"Applicando il Reward Shaping all'algoritmo vincitore della DQ1, l'introduzione di una
penalità quadratica per le variazioni brusche di potenza dei motori produce un volo
stazionario significativamente più fluido, e quanto incide questo vincolo sul tempo
necessario a raggiungere il target?"

- Tecnica: alla ricompensa del compito si somma un termine -λ·||aₜ − aₜ₋₁||², cioè la
  differenza al quadrato tra i comandi ai 4 motori in due istanti consecutivi. "Quadratica"
  = le variazioni grandi sono punite molto più delle piccole → l'agente impara a non dare
  strattoni ai motori. È la tecnica standard di action smoothness, semplice, e aderente alla
  domanda. (Scartate: penalità sull'ampiezza = controlla l'energia non la fluidità;
  jerk/CAPS = troppo complesse; L1 = meno adatta e cambierebbe "quadratica".)
- Variabile studiata: λ (peso della penalità). Sweep su più valori incluso λ=0 (= vincitore
  DQ1 senza shaping) per tracciare il trade-off. Valori esatti da calibrare sulla scala
  della ricompensa.
- Applicata a: il vincitore della DQ1 (PPO o SAC, deciso dai risultati).
- Metriche: fluidità = media delle variazioni quadratiche dei comandi ai motori tra istanti
  consecutivi (più bassa = più liscio); tempo al target = settling time = passi finché il
  drone arriva e RESTA vicino al target (|z−1|<ε mantenuto per N passi).
- Analisi: curva del trade-off fluidità vs tempo al target al crescere di λ (= il "prezzo").
- Confronto su più seed, per poter dire "significativamente".

### DQ3 — Robustezza e Domain Randomization (deciso)
Formulazione (invariata, già ben posta):
"Il modello ottimizzato per la fluidità (vincitore DQ2) è in grado di resistere a turbolenze
esterne stocastiche (vento) introdotte in fase di valutazione, o è necessario addestrare una
nuova policy iniettando disturbi fisici casuali durante il training per evitare la perdita di
assetto?"

- Vento: forza esterna applicata al drone via PyBullet (applyExternalForce), in una sottoclasse
  di HoverAviary. Modello stocastico CORRELATO nel tempo (Ornstein-Uhlenbeck): raffiche che
  crescono e calano. Motivo: un vento costante farebbe imparare solo un offset fisso (banale e
  irrealistico); rumore per-passo sarebbe troppo brusco. Direzione casuale nel piano orizzontale
  (è lì che serve il controllo d'assetto), intensità espressa in % del peso del drone, calibrata
  con uno spike (forza al limite in cui la policy non protetta inizia a fallire a volte).
- Il vento NON è osservato dalla policy: vede solo lo stato del drone e reagisce agli effetti
  (robustezza reattiva realistica).
- Confronto: Policy A = vincitrice DQ2 (allenata in aria calma) testata nel vento (zero-shot)
  vs Policy B = stesso algoritmo e stessa reward della DQ2 + vento iniettato in addestramento
  (domain randomization). Unica differenza A vs B: vento in training.
- Valutazione: A e B sugli STESSI episodi di vento. Test su vento in-distribution e su vento
  più forte mai visto (out-of-distribution) per misurare la generalizzazione. Figura chiave:
  curva di robustezza (tasso di fallimento vs intensità del vento) per A e B.
- Metriche: Crash Rate / perdita di assetto sotto vento (riuso definizione DQ1) + deriva dal
  target (errore di posizione RMS). Confronto su più seed.
- Ipotesi collegata a DQ2: la fluidità (correzioni morbide) potrebbe ridurre la reattività al
  vento; si verifica se il domain randomization riesce comunque a rendere robusta la policy.

  ## 17-06-2026 — Spazio d'azione, DQ2, DQ3, proposta consegnata

### Spazio d'azione del progetto (vale per DQ1–DQ2–DQ3)
- Scelto `rpm`: la policy comanda i 4 motori in modo indipendente (NON `one_d_rpm`,
  dove tutti i motori riceverebbero lo stesso valore).
- Motivi:
  * DQ2: la penalità è sulle "variazioni di potenza dei motori" → l'azione DEVE essere
    la potenza dei 4 motori.
  * DQ3: per contrastare il vento il drone deve inclinarsi, e per inclinarsi servono
    spinte diverse sui 4 motori. Con `one_d_rpm` la DQ3 sarebbe impossibile.
  * La stessa policy attraversa DQ1→DQ2→DQ3, quindi lo spazio d'azione si sceglie una
    volta sola.
- Costo: addestramento più lento (ore, non minuti). Mitigazione: Colab per i run
  pesanti; se necessario, 3 seed invece di 5.

### DQ2 — Reward Shaping per la fluidità
- Tecnica: alla ricompensa del compito si somma -λ·||aₜ − aₜ₋₁||², ovvero la
  differenza quadratica tra i comandi ai motori in due istanti consecutivi.
- "Quadratica": le variazioni grandi sono punite molto più delle piccole, quindi
  l'agente impara a non dare strattoni ai motori (= fluidità).
- Variabile studiata: λ (peso della penalità). Sweep su più valori incluso λ=0
  (= vincitore DQ1 senza shaping), per tracciare il trade-off fluidità ↔ tempo al
  target.
- Applicata al vincitore della DQ1.
- Metriche: fluidità = media delle variazioni quadratiche dei comandi tra istanti
  consecutivi (più bassa = più liscio); tempo al target = settling time = numero di
  passi finché |z−1|<ε è mantenuto per N passi.
- Multi-seed obbligatorio per poter parlare di differenza "significativa".
- Alternative scartate: penalità sull'ampiezza dell'azione (controlla l'energia,
  non la fluidità); L1 (cambia il senso di "quadratica"); jerk/CAPS (troppo complesse
  rispetto alla domanda).

### DQ3 — Robustezza e Domain Randomization
- Vento: forza esterna applicata via PyBullet (applyExternalForce) in una sottoclasse
  di HoverAviary.
- Modello stocastico correlato nel tempo (Ornstein-Uhlenbeck): raffiche che crescono
  e calano. Motivi: un vento costante farebbe imparare un offset banale e irrealistico;
  rumore per-passo sarebbe troppo brusco e poco fisico.
- Direzione casuale nel piano orizzontale (è lì che serve il controllo d'assetto),
  intensità espressa in % del peso del drone, calibrata con uno spike (intensità al
  limite in cui la policy non protetta inizia a fallire qualche volta).
- Il vento NON è osservato dalla policy: vede solo lo stato del drone e reagisce agli
  effetti (robustezza reattiva realistica, come un drone vero).
- Confronto:
  * Policy A = vincitrice DQ2 (allenata in aria calma) testata nel vento (zero-shot).
  * Policy B = stesso algoritmo e stessa reward della DQ2 + vento iniettato in
    addestramento (domain randomization).
  * Unica differenza A↔B: presenza di vento in training.
- Valutazione: A e B sugli STESSI episodi di vento. Test su vento in-distribution
  (= stessa intensità del training di B) e out-of-distribution (= più forte di
  qualsiasi visto), per misurare la generalizzazione. Figura chiave: curva di
  robustezza (tasso di fallimento vs intensità del vento) per A e B.
- Metriche: Crash Rate / perdita di assetto sotto vento (riuso definizione DQ1) +
  deriva dal target (errore RMS di posizione).
- Ipotesi collegata a DQ2: la fluidità (correzioni morbide) potrebbe ridurre la
  reattività al vento; si verifica se il domain randomization riesce comunque a
  rendere robusta la policy fluida.

### Proposta di progetto consegnata
- Nome: StableHover. Settimana: S09 Deep Reinforcement Learning.
- Ambiente: gym-pybullet-drones (repo: github.com/utiasDSL/gym-pybullet-drones),
  task HoverAviary, azione `rpm` a 4 motori indipendenti.
- D3 marcata come responsible DL (robustezza/sicurezza sotto disturbi).
- Consegnata via Moodle. Mandata mail di rettifica per refuso nel link al repository
  dell'ambiente (era learnsyslab, repo corretto utiasDSL). Versione v2 della proposta
  in `assets/`.

  ## 17-06-2026 — Lettura di HoverAviary: reward, fine episodio, Crash Rate operativa

Letto `gym_pybullet_drones/envs/HoverAviary.py` della v2.1.0 installata.
Verifica di affidabilità: `cat` dal Mac IDENTICO al file raw del branch main su
GitHub → per questo file nessun disallineamento web/installato. (I genitori
BaseRLAviary/BaseAviary verranno comunque verificati a parte.)

### Costanti dell'ambiente (lette, non decise)
- Target di hover: TARGET_POS = [0, 0, 1] (z = 1 m). Durata episodio:
  EPISODE_LEN_SEC = 8 s. Frequenze: fisica 240 Hz (pyb_freq), controllo 30 Hz
  (ctrl_freq) → 8 passi di fisica per ogni decisione, 240 decisioni per episodio.
  Il numero 240 (passi di controllo/episodio) è l'unità temporale per il settling
  time della DQ2 e per l'asse X dei grafici.
- Vettore di stato (indici rilevanti): state[0:3] = x,y,z; state[7] = roll;
  state[8] = pitch; state[9] = yaw (NON usato dai controlli di fine episodio).

### _computeReward (punto d'iniezione della DQ2)
- Ricompensa per passo: max(0, 2 − ||TARGET_POS − pos||⁴). Massimo 2/passo
  (solo sul target); la quarta potenza la fa crollare in fretta (a 1 m vale 1,
  a ~1.19 m è già 0). Massimo teorico per episodio ≈ 240 × 2 = 480 → coerente
  con il ~470 osservato con learn.py (conferma indiretta della versione).
- Decisione: il termine di fluidità della DQ2 (−λ·||aₜ−aₜ₋₁||²) si sommerà QUI,
  in una sottoclasse di HoverAviary che override-a _computeReward.

### _computeTerminated (quasi sempre falso → conseguenza metodologica)
- Restituisce True solo se distanza dal target < 0.0001 m (0.1 mm): in pratica
  non capita mai. Conseguenza: gli episodi finiscono quasi sempre per "truncated".
- Conferma la scelta già nel diario: la Crash Rate NON si legge da truncated==True,
  ma dallo STATO FINALE dell'episodio.

### _computeTruncated → definizione VERA di schianto (DECISA)
- L'ambiente tronca per due famiglie di cause:
  (a) FUORI CONTROLLO = SCHIANTO: |x|>1.5 OR |y|>1.5 OR z>2.0 OR |roll|>0.4 OR
      |pitch|>0.4 (0.4 rad ≈ 22.9° = soglia di perdita d'assetto);
  (b) TEMPO SCADUTO = SOPRAVVISSUTO: oltre 8 s di volo.
- DEFINIZIONE OPERATIVA di Crash Rate (DQ1):
  schianto = al passo finale è violata almeno una delle 5 soglie geometriche/
  d'assetto; sopravvivenza = episodio arrivato al limite di 8 s senza violarle.
  Crash Rate = schianti / episodi di valutazione totali (%).
- Riuso in DQ3: le soglie |roll|>0.4 e |pitch|>0.4 sono anche l'indicatore di
  "perdita di assetto" sotto vento. Strumento unico, definito una volta.

### Limitazione nota e sua gestione (DECISA, per onestà metodologica)
- La soglia su z è solo superiore (z>2.0): un drone che cadesse perfettamente
  verticale verso il pavimento (z→0) potrebbe non essere troncato come schianto.
  In pratica un drone che precipita si inclina quasi sempre → scatta roll/pitch.
- Decisione: si ancora la Crash Rate alle 5 soglie NATIVE dell'ambiente (limiti
  veri, non arbitrari) e in più si logga lo stato finale + il motivo di fine di
  ogni episodio, così eventuali atterraggi a terra mascherati da timeout sono
  rilevabili a posteriori e riportati con trasparenza. Difendibilità + tracciabilità.

  ## 17-06-2026 — Lettura di BaseRLAviary: azione rpm, buffer, aₜ della DQ2, osservazione

Letto `gym_pybullet_drones/envs/BaseRLAviary.py` (genitore di HoverAviary) dalla
fonte installata sul Mac. Qui sono definiti: spazio d'azione, traduzione azione→RPM,
buffer delle azioni, spazio e contenuto dell'osservazione.

### Spazio d'azione e traduzione azione → RPM (act = rpm)
- L'agente NON emette RPM diretti. Emette 4 numeri normalizzati in [−1, +1] (uno per
  motore); _actionSpace è un Box(low=−1, high=+1) di dimensione 4.
- Traduzione in _preprocessAction: rpm = HOVER_RPM · (1 + 0.05 · action), dove
  HOVER_RPM è il regime di giri che bilancia esattamente la gravità (hover a regime).
  Quindi: action=0 → hover; action=+1 → +5% RPM; action=−1 → −5% RPM.
- Conseguenza (autorità di controllo): l'agente modula solo il ±5% attorno all'hover.
  Banda stretta, ottima per l'hovering (azione ben condizionata, niente strattoni
  enormi), ma è un vincolo fisico da tenere presente per la DQ3.

### DQ2 — definizione operativa di aₜ nella penalità di fluidità (DECISA)
- Penalità DQ2: −λ·||aₜ − aₜ₋₁||². Decisione: aₜ = AZIONE NORMALIZZATA in [−1,1]⁴
  (l'output della policy), NON gli RPM.
- Equivalenza (motivazione): poiché rpm = HOVER_RPM·(1+0.05·a), si ha
  rpm_t − rpm_{t−1} = (0.05·HOVER_RPM)·(aₜ − aₜ₋₁), quindi
  ||Δrpm||² = (0.05·HOVER_RPM)²·||Δa||². Penalizzare gli RPM o l'azione normalizzata
  è IDENTICO a meno di una costante, e quella costante viene riassorbita da λ (che
  facciamo variare nello sweep). La scelta resta quindi fedele al testo della DQ2
  ("variazioni di potenza dei motori").
- Perché l'azione normalizzata e non gli RPM: (a) è ciò che la policy controlla
  direttamente; (b) è già disponibile nel action_buffer → zero codice extra;
  (c) essendo limitata in [−1,1], λ ha una scala interpretabile e pulita.

### Meccanismo del buffer delle azioni (come si ottengono aₜ e aₜ₋₁)
- self.action_buffer è una deque di lunghezza ctrl_freq//2 = 15 (≈ 0,5 s di storico),
  riempita di zeri all'avvio. Ogni _preprocessAction vi appende l'azione corrente.
- Nel futuro override di _computeReward: aₜ = action_buffer[-1], aₜ₋₁ = action_buffer[-2].
  La differenza è quindi già pronta, senza variabili di stato aggiuntive.
- Transitorio al primo passo: aₜ₋₁ = 0 (buffer inizializzato a zeri) → penalità iniziale
  trascurabile.
- Da confermare leggendo BaseAviary: l'ordine esatto delle chiamate dentro step()
  (_preprocessAction prima di _computeReward). Il meccanismo del buffer è comunque questo.

### Osservazione della policy (obs = kin) e premessa della DQ3 (CERTIFICATA dal codice)
- _computeObs restituisce 72 dimensioni: 12 valori cinematici [posizione x,y,z; assetto
  roll,pitch,yaw; velocità lineare vx,vy,vz; velocità angolare wx,wy,wz] + storico delle
  ultime 15 azioni (15×4 = 60).
- Nell'osservazione NON compare alcun termine di vento/forza/disturbo esterno: la policy
  vede solo lo stato del proprio corpo e la propria storia di comandi.
- Certificazione DQ3: iniettando il vento via applyExternalForce, la policy NON lo osserva
  direttamente; ne percepisce solo gli effetti (variazioni di posizione/velocità/assetto).
  È la "robustezza reattiva realistica" voluta dalla DQ3 (drone senza sensore di vento a
  bordo). Vincolo implementativo conseguente: la sottoclasse del vento NON deve modificare
  _computeObs, così il vento resta non osservato per costruzione.

### Nota per la DQ3 — la calibrazione del vento è legata all'autorità ±5%
- L'autorità limitata (±5% RPM) fissa la scala del vento "sensato": un vento troppo forte
  sarebbe incontrastabile da qualunque policy e svuoterebbe l'esperimento. Conferma la
  necessità (già prevista) di calibrare l'intensità con uno spike. Il confronto A-vs-B usa
  lo STESSO vento per entrambe → è relativo, quindi il valore assoluto del vento non è il punto.

  ## 17-06-2026 — Lettura di BaseAviary: ordine in step(), iniezione del vento (chiusura lettura ambiente)

Letti dalla fonte installata sul Mac i metodi step(), _updateAndStoreKinematicInformation()
e _physics() di BaseAviary.py. Con questi si chiude la fase di studio dell'ambiente.

### Conclusione DQ2 — aₜ blindato (ordine delle chiamate in step())
- Ordine reale dentro step(): (1) _preprocessAction(action) [appende l'azione al
  action_buffer e la traduce in RPM]; (2) loop fisica, PYB_STEPS_PER_CTRL = 8 sotto-passi
  a 240 Hz; (3) _updateAndStoreKinematicInformation() [rilegge pos/assetto/vel da PyBullet];
  (4) _computeObs / _computeReward / _computeTerminated / _computeTruncated / _computeInfo.
- _preprocessAction gira PRIMA di _computeReward → al momento della reward:
  aₜ = action_buffer[-1] (azione appena applicata), aₜ₋₁ = action_buffer[-2]. La penalità
  DQ2 −λ·||action_buffer[-1] − action_buffer[-2]||² è quindi ben definita. Ipotesi confermata.
- Inoltre lo stato è riletto (punto 3) PRIMA della reward → _computeReward vede la posizione
  DOPO il passo di fisica (corretto: valuta dove il drone è finito, non dov'era).

### Conclusione DQ3 — punto e modo d'iniezione del vento (DECISA)
- _physics() applica le spinte dei motori (forces = KF·rpm²) con
  p.applyExternalForce(..., flags=LINK_FRAME) + coppia di imbardata con applyExternalTorque,
  una volta per ogni sotto-passo di fisica, subito prima di p.stepSimulation(). Motivo della
  ripetizione: in PyBullet una forza esterna vale solo per il successivo stepSimulation() e
  poi si azzera.
- DECISIONE: il vento si inietta con override di _physics() nella sottoclasse della DQ3:
  super()._physics(rpm, nth_drone) [mantiene le spinte dei motori] → avanzamento dello stato
  del vento (Ornstein-Uhlenbeck, a 240 Hz) → p.applyExternalForce sul corpo del drone, forza
  [Fx, Fy, 0] con flags=p.WORLD_FRAME, applicata al baricentro.
- Motivazioni:
  * Idiomatica: l'ambiente già aggiunge forze extra accanto a _physics nello stesso loop
    (_drag, _downwash, _groundEffect). Il vento è un'altra forza di disturbo → stesso pattern.
  * Footprint minimo: non si ricopia step() (resta della libreria), poche righe nostre →
    rispetta la regola "<15% di codice altrui".
  * Frame corretto: i motori sono in LINK_FRAME (frame locale); il vento in WORLD_FRAME
    (direzione fissa nel mondo, indipendente dall'orientamento del drone). Applicato al
    baricentro → nessuna coppia artificiale → spinta laterale pulita: il drone deve inclinarsi
    per generare spinta orizzontale e tenere la posizione (= dinamica voluta dalla DQ3).
- Assunzione di modello (dichiarata): il vento è modellato come UN'UNICA forza netta sul
  corpo, non come carico aerodinamico distribuito sulle superfici. Coerente col diario
  ("forza esterna applicata al drone") e standard per lo studio. Si lega all'autorità ±5%:
  la spinta orizzontale opponibile è limitata → vento sensato = piccola % del peso (coerente
  col piano di calibrazione con spike).

### Stato: lettura dell'ambiente COMPLETA
- Studiati HoverAviary (reward, fine episodio, Crash Rate), BaseRLAviary (azione rpm, buffer,
  osservazione, vento non osservato) e BaseAviary (ordine step(), iniezione forze). Tutte le
  premesse implementative di DQ1/DQ2/DQ3 sono verificate sul codice reale. Prossima fase:
  implementazione (codice nostro), a partire dall'impalcatura sperimentale della DQ1.