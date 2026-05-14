# Mindustry SIM — Resource Extraction RL Agent

PPO agent with LSTM that learns to build optimal copper mining routes
in a Mindustry game mechanics simulator — no WebSocket, no delays, pure Python.

---

## Why a Simulator?

In the real game, the agent receives observations through a WebSocket mod and waits
for a response from the Java process. This yields roughly 10–30 episodes per hour.
The simulator eliminates all that overhead:

| Mode | Episodes per hour | GPU utilization |
|------|-------------------|-----------------|
| Real game (WebSocket) | ~20 | 10–15% |
| Simulator (this file) | 500–2000+ | 60–80% |

After training, take the weights (`sim_agent.pt`) and fine-tune for a few hundred
episodes in the real game.

---

## What the Agent Learns

- Navigate a randomly generated 256x256 map with water, copper ore, and a core
- Locate copper clusters and place drills (2x2) for maximum ore coverage
- Build conveyor chains with correct direction from drill to core
- Route around water bodies — building on water is not allowed
- Delete bad placements and retry
- Optimize efficiency: fewer buildings, more copper (reward shaping)

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy matplotlib
```

AMD GPU (DirectML):
```bash
pip install torch-directml
```

### 2. Run a test episode

```bash
python mindustry_sim.py --test
```

Prints a terminal map and episode statistics:

```
=== Test episode ===
Steps: 200 | Reward: -12.34 | Copper: 0.00
Drills: 2 | Conveyors: 8

Agent: (134, 89)  Core: (120, 75)

....c....c..~~~~....
..c..c..~~~~.....c..
....~~~~.....XXXX...
..~~~~.......XXXX...
~~~~~........XXXX...
...~~....D>>.XXXX...
....c....D>>>.......
....@...............
```

Legend: `@` agent, `X` core, `D` drill, `>` conveyor, `~` water, `c` copper ore

### 3. Start training

```bash
python mindustry_sim.py --episodes 2000
```

Every 10 episodes the console prints:

```
Ep    10 | avg_r= -45.21 | copper=   0.0 | drills=  0 | conv=   3 | ep/s=12.4 | pol=0.412 val=1.234 ent=3.21
Ep   100 | avg_r=  +8.44 | copper=  12.3 | drills=  2 | conv=  18 | ep/s=14.1 | pol=0.201 val=0.834 ent=2.87
Ep   500 | avg_r= +67.12 | copper= 124.5 | drills=  5 | conv=  43 | ep/s=15.3 | pol=0.089 val=0.312 ent=2.41
```

If `avg_r` and `copper` are trending upward, the agent is learning correctly.

### 4. Resume from checkpoint

```bash
python mindustry_sim.py --episodes 3000 --load
```

### 5. Plot training progress

```bash
python mindustry_sim.py --plot
```

Saves `training_plot.png` — reward and copper delivered per episode,
both with a 50-episode moving average.

---

## Command Line Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 2000 | Number of training episodes |
| `--load` | — | Load `sim_agent.pt` and continue training |
| `--test` | — | Run one random episode with terminal map |
| `--plot` | — | Generate plot from `training_log.csv` |
| `--map-size N` | 256 | Map size N x N |
| `--steps N` | 4000 | Steps per episode |

---

## Simulator Mechanics

### Map

```
Size:    256x256 tiles (configurable)
Water:   ~18% of map — random rivers, lakes, seas (blob generation)
Copper:  ~12% of map — random ore clusters
Core:    4x4 tiles, placed randomly away from edges and water
```

The map is regenerated randomly every episode so the agent cannot
memorize a fixed layout.

### Drill (2x2 tiles)

```
Output rate:   0.09 copper/sec per copper tile covered
               (0.09 for 1 tile up to 0.36 for all 4)
Delivery:      to a random adjacent conveyor from the 8 surrounding tiles
Restrictions:
  - cannot be placed on water (even one tile out of four)
  - cannot overlap existing structures
  - must keep at least 1 tile gap from the core perimeter
Build time:    2 ticks (~0.1 sec)
```

### Conveyor (1x1 tile)

```
Directions:    0=right, 1=up, 2=left, 3=down
Capacity:      3 items
Speed:         4.2 items/sec
Restrictions:  cannot be placed on water or over existing structures
Build time:    1 tick (~0.016 sec)

Only works as part of a connected chain:
  Drill -> Conveyor -> ... -> Conveyor -> Core
  Each tile passes items to the next tile in its facing direction.
  When the end of the chain reaches the core, copper is credited.
```

### Agent

```
Movement:      flies freely (no water collision), speed 8 tiles/sec
View range:    43x43 tiles centered on the agent (radius 21)
Build range:   same as view range
Actions:       one at a time; next action waits until current finishes
```

---

## Neural Network Architecture

```
Map input (8 x 43 x 43)           Scalars (10)
         |                               |
   CNN (3 layers)                  MLP (2 layers)
   + ResBlocks                     + LayerNorm
   -> 7744 features                -> 64 features
         |________________ concat _______|
                            |
                       LSTM (256)
                    /     |      \
              Actor-T  Actor-RX  Actor-RY      Critic
              (type)   (X goal)  (Y goal)      (V(s))
              7 log.   43 log.   43 log.        1 value
```

The CNN processes the local map: water, copper, core, drills, conveyors,
their fill levels and directions.

The LSTM retains memory of previous steps — critical for understanding
"I already tried building there."

The Actor selects actions hierarchically: first WHAT to do, then WHERE (X),
then WHERE (Y).

The Critic estimates how good the current state is, used in GAE advantage estimation.

### Observation Channels (43x43 map)

| Channel | Content |
|---------|---------|
| 0 | Water (1 = water or map boundary) |
| 1 | Copper ore (1 = ore present) |
| 2 | Core (1 = core tile) |
| 3 | Drill (1 = drill tile) |
| 4 | Conveyor (1 = conveyor present) |
| 5 | Conveyor direction (0..3 / 4, normalized) |
| 6 | Conveyor fill level (0..1) |
| 7 | Agent position (1 at center) |

### Scalar Observations (10 values)

| Index | Value |
|-------|-------|
| 0 | Copper in core / 500 (normalized) |
| 1 | Copper delta this step |
| 2 | Build busy fraction (0..1) |
| 3–4 | Vector to core (dx, dy normalized) |
| 5 | Drill count / 20 |
| 6 | Conveyor count / 200 |
| 7–8 | Agent position (x/w, y/h) |
| 9 | Episode progress (step / max_steps) |

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Each step | -0.002 (time penalty) |
| Copper delivered to core | +5.0 x amount |
| Drill placed on copper | +2.0 x (copper tiles / 4) |
| Drill placed with no copper | -3.0 |
| Conveyor connected to a source | +0.5 |
| Conveyor reaches the core | +0.5 |
| Idle for more than 10 steps | -0.1 / step |

The per-step penalty exists so the agent learns not to overbuild —
every structure must pay for itself in copper delivered.

---

## Output Files

```
mindustry_sim.py        <- all code (environment + network + PPO)
sim_agent.pt            <- checkpoint (created automatically)
training_log.csv        <- per-episode log (episode, reward, copper, ...)
training_plot.png       <- training curve (generated by --plot)
```

---

## Transferring Weights to the Real Game

Once the agent reliably delivers copper in the simulator:

```
1. Copy sim_agent.pt -> mindustry_agent.pt
2. In mind_rl_agent.py adjust load_checkpoint() to load weights
   from sim_agent.pt (architectures must match)
3. Run fine-tuning in the real game:
   python mind_rl_agent.py --real --episodes 500 --load
4. For the first 100 episodes reduce lr by 10x for stable fine-tuning
```

Make sure the network architecture in the simulator (8 map channels,
10 scalars) matches what mind_rl_agent.py expects before transferring.
Check `map_channels` and `scalar_size` in both files.

---

## Training Tips

- The first 50–100 episodes will have negative avg_r — this is normal,
  the agent is randomly exploring.
- The sign that learning is working: `copper` starts growing around episode 200–500.
- If the agent is stuck: increase `entropy_coef` to 0.05 for more exploration.
- Smaller map is faster: `--map-size 128` cuts episode time by roughly 4x.
- Longer episodes help: `--steps 6000` gives the agent more time to build a full chain.
- On AMD with DirectML, CPU training can be faster than GPU for this model size
  due to the overhead of transferring small batches.