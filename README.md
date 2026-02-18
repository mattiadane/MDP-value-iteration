# MDP-value-iteration

## Testo

Un drone avanzato è stato posizionato sul campo per prendere dati critici sulla biodiversità marina in una regione costiera. Il Drone parte dal punto S, posizionato vicino alla costa, e deve navigare fino al punto G, un punto designato ricco di corali marini e vita di mare. Lunga la strada, il drone deve manovrare in modo attento attraverso degli ambienti subacquei dinamici, evitando pericoli e ottimizzando l'uso di energia.
Cosa include l'ambiente

### L'ambiente è così composto:

    (O) Open Water: movimento normale, non sono presenti sfide
    (C) Currents: aree dove il movimento del drone è influenzato da correnti marine che possono portarlo fuori rotta
    (F) Seaweed Forest: vegetazione densa che rallenta il drone che deve usare più energia per muoversi
    (E) Energy Stations: punti specifici dove il drone può ricaricare la batteria, riducendo il costo totale della navigazione

### Dettaglia dell'ambiente

    Rappresentazione della griglia: l'ambiente è rappresentato da una matrice 10 x 10

    S - Start State: Il punto di partenza del drone (0, 0)

    G - Goal State: Il punto di arrivo del drone (9, 7)

    Costo del momvimento: ogni movimento ha come costo base di -0.04

    Pericoli:

        Correnti forti: si entra in una zona con questi risultati in un ambiente stocastico:
            80% di possibilità di rimanere sulla stessa rotta
            10% di possibilità di essere spinto su una cella a sinistra perpendicolare al movimento desiderato all'inizio
            10% di possibilità di essere sprinto su una cella a destra perpendicolare al movimento desiderato all'inizio

        Seaweed Forest: quando si entra in queste zone si deve aggiungere un -0.02 in più di penalty rispetto alla normale penalty del movimento (ex: -0.24)

    Stazioni di energia: provvede ad un +1.0 di reward quando visitato (nonostante questa scelta possa portare il drone lontano dalla cella di goal).

### Visualizzazione ambiente
    
    [['S' 'O' 'O' 'F' 'F' 'F' 'F' 'O' 'O' 'O']
     ['O' 'F' 'C' 'C' 'C' 'O' 'F' 'E' 'F' 'O']
     ['O' 'O' 'F' 'F' 'F' 'O' 'F' 'F' 'F' 'C']
     ['F' 'C' 'F' 'F' 'E' 'C' 'F' 'O' 'F' 'C']
     ['F' 'C' 'F' 'F' 'F' 'C' 'F' 'O' 'F' 'C']
     ['F' 'E' 'F' 'O' 'O' 'O' 'F' 'E' 'F' 'C']
     ['O' 'O' 'O' 'O' 'O' 'O' 'F' 'F' 'F' 'C']
     ['O' 'F' 'F' 'F' 'O' 'O' 'O' 'F' 'F' 'C']
     ['O' 'O' 'O' 'O' 'F' 'F' 'F' 'F' 'F' 'C']
     ['F' 'F' 'F' 'O' 'O' 'O' 'O' 'G' 'O' 'F']]
    
### Codifica delle azioni
    
    Actions encoding:  {0: 'L', 1: 'R', 2: 'U', 3: 'D'}
    Cell type of start state:  S
    Cell type of goal state:  G
    Cell type of cell (0, 3):  F
    Cell type of cell (1, 2):  C
    Cell type of cell (1, 7):  E
