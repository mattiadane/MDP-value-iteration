# MDP Value Iteration – Marine Drone Navigation

## Description

An advanced marine drone has been deployed to collect critical biodiversity data in a coastal region.  
The drone starts from point **S**, located near the shore, and must navigate to point **G**, a designated area rich in coral and marine life.  
Along the way, the drone must carefully maneuver through dynamic underwater environments, avoid hazards, and optimize energy usage.

This project models the environment as a **Markov Decision Process (MDP)** and applies **Value Iteration** to compute the optimal navigation policy.

---

## Environment Description

The environment is represented as a **10 × 10 grid**, where each cell corresponds to a specific underwater condition:

### Cell Types

- **(O) Open Water** – Normal movement, no particular challenges  
- **(C) Currents** – Marine currents may push the drone off its intended path  
- **(F) Seaweed Forest** – Dense vegetation that slows the drone and increases energy cost  
- **(E) Energy Stations** – Recharge points that provide a positive reward  
- **S** – Start state at coordinates **(0, 0)**  
- **G** – Goal state at coordinates **(9, 7)**  

---

## Movement Cost and Rewards

- **Base movement cost:** `-0.04`  
- **Seaweed Forest penalty:** additional `-0.02` (total `-0.24`)  
- **Energy Station reward:** `+1.0` when visited  
- **Goal state:** terminal state with positive reward (defined in code)

---

## Stochastic Transitions (Currents)

When the drone moves into a **Current (C)** cell, the movement becomes stochastic:

- **80%** chance to continue in the intended direction  
- **10%** chance to be pushed **left** (perpendicular to intended direction)  
- **10%** chance to be pushed **right** (perpendicular to intended direction)

This models the unpredictable nature of underwater currents.

---

## Environment Visualization

```
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
```

---

## Actions Encoding

```
{0: 'L', 1: 'R', 2: 'U', 3: 'D'}
```

- **L** – Move Left  
- **R** – Move Right  
- **U** – Move Up  
- **D** – Move Down  

---

## Example Cell Types

- Start state: **S**  
- Goal state: **G**  
- Cell (0, 3): **F** (Seaweed Forest)  
- Cell (1, 2): **C** (Currents)  
- Cell (1, 7): **E** (Energy Station)

---

## Purpose of the Project

The goal is to:

- Model the underwater environment as an MDP  
- Define transition probabilities and rewards  
- Apply **Value Iteration** to compute the optimal policy  
- Visualize the resulting value function and optimal actions  

This allows the drone to navigate efficiently while balancing risk, energy consumption, and environmental constraints.

---

## Collaborators
 - (Alesssandro De Carli)[https://github.com/aledpl5]



