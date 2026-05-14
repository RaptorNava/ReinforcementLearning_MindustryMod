# Mindustry RL Agent

A reinforcement learning agent that learns to play Mindustry using PPO with LSTM.
The agent learns to mine copper, build conveyor chains, and navigate the map efficiently.

The project has three components that work together:

```
mindustry_sim.py      Fast Python simulator for training (no game required)
mind_rl_agent.py      Main PPO agent that connects to the real game
mod/                  Java WebSocket mod that bridges the game and the agent
```

The intended workflow is: train in the simulator until the agent is stable,
then transfer the weights and fine-tune in the real game.

---

## Repository Structure

```
Mindrl/
  mind_rl_agent.py        PPO agent for the real game (WebSocket)
  mindustry_sim.py        Fast simulator for training without the game
  mindustry_sim0.py       Earlier simulator version (reference)
  mindustry_rl_agent.py   Earlier agent version (reference)
  dashboard.py            Live training dashboard
  test.py                 Checkpoint inspection utility
  mod/                    Java mod source (MindustryMod.java + MindustryBridge.java)
  mindustry_agent.pt      Real-game agent checkpoint
  sim_agent.pt            Simulator agent checkpoint
  training_log.csv        Per-episode training log
  training_plot.png       Training curve image
  media/                  GIFs and screenshots for documentation
```

---

## Quick Start

### Option A: Train in the simulator (recommended first step)

No game installation needed. Thousands of episodes per hour.

```bash
pip install torch numpy matplotlib

python mindustry_sim.py --episodes 2000
```

### Option B: Train in the real game

Requires Mindustry running with the WebSocket mod installed.

```bash
pip install torch numpy websockets

# Start Mindustry, load a map, then:
python mind_rl_agent.py
```

---

## Part 1: Simulator (`mindustry_sim.py`)

The simulator replicates Mindustry's resource extraction mechanics in pure Python
with no rendering and no networking overhead.

### Why use it

| Mode | Episodes per hour | GPU utilization |
|------|-------------------|-----------------|
| Real game (WebSocket) | ~20 | 10-15% |
| Simulator | 500-2000+ | 60-80% |

The bottleneck in the real game is the WebSocket round-trip to the Java process.
The simulator eliminates this entirely.

### What is simulated

**Map (256x256 tiles)**
- Water: ~18% of map, random rivers, lakes, seas. Building on water is not allowed.
- Copper ore: ~12% of map, random clusters.
- Core: 4x4 tiles, placed randomly away from edges and water.
- Map is regenerated randomly every episode so the agent cannot memorize layouts.

**Drill (2x2 tiles)**
- Output rate: 0.09 copper/sec per copper tile covered (max 0.36 for all 4 tiles).
- Delivers items to a random adjacent conveyor from the 8 surrounding tiles.
- Cannot be placed on water, over existing structures, or within 1 tile of the core.
- Build time: 2 ticks (~0.1 sec).

**Conveyor (1x1 tile)**
- Four directions: right, up, left, down.
- Capacity: 3 items. Speed: 4.2 items/sec.
- Cannot be placed on water or over existing structures.
- Build time: 1 tick (~0.016 sec).
- Only delivers copper when forming an unbroken chain from a drill to the core.

**Agent**
- Flies freely across the map (no water collision).
- View range and build range: 43x43 tiles centered on the agent (radius 21).
- One action at a time; the next action waits until the current build finishes.

### Observation space

Map tensor: 8 channels x 43x43 tiles (centered on agent)

| Channel | Content |
|---------|---------|
| 0 | Water (1 = water or map boundary) |
| 1 | Copper ore |
| 2 | Core tiles |
| 3 | Drill tiles |
| 4 | Conveyor tiles |
| 5 | Conveyor direction (0..3 normalized) |
| 6 | Conveyor fill level (0..1) |
| 7 | Agent position (1 at center pixel) |

Scalar vector: 10 values

| Index | Value |
|-------|-------|
| 0 | Copper in core / 500 |
| 1 | Copper delta this step |
| 2 | Build busy fraction (0..1) |
| 3-4 | Vector to core (dx, dy normalized) |
| 5 | Drill count / 20 |
| 6 | Conveyor count / 200 |
| 7-8 | Agent position (x/w, y/h) |
| 9 | Episode progress (step / max_steps) |

### Action space

7 discrete action types, each with a relative (rx, ry) target within view range:

| Type | Action |
|------|--------|
| 0 | Move toward (rx, ry) |
| 1 | Build drill at (rx, ry) |
| 2 | Build conveyor facing right |
| 3 | Build conveyor facing up |
| 4 | Build conveyor facing left |
| 5 | Build conveyor facing down |
| 6 | Delete structure at (rx, ry) |

### Reward structure

| Event | Reward |
|-------|--------|
| Each step | -0.002 |
| Copper delivered to core | +5.0 x amount |
| Drill placed on copper tiles | +2.0 x (copper tiles / 4) |
| Drill placed with no copper | -3.0 |
| Conveyor connected to source | +0.5 |
| Conveyor reaches core | +0.5 |
| Idle more than 10 steps | -0.1 per step |

The per-step penalty ensures the agent learns not to overbuild.
Every structure must pay for itself in copper delivered to the core.

### Commands

```bash
# Train from scratch
python mindustry_sim.py --episodes 2000

# Resume from checkpoint
python mindustry_sim.py --episodes 3000 --load

# Run one random episode with terminal map printout
python mindustry_sim.py --test

# Generate training plot from log
python mindustry_sim.py --plot

# Smaller map for faster iteration
python mindustry_sim.py --episodes 2000 --map-size 128

# Longer episodes give more time to build a full chain
python mindustry_sim.py --episodes 1000 --steps 6000
```

### Output files

```
sim_agent.pt          Checkpoint (saved every 50 episodes)
training_log.csv      Per-episode log: episode, reward, copper, drills, conveyors, losses
training_plot.png     Training curve with 50-episode moving average
```

---

## Part 2: Real Game Agent (`mind_rl_agent.py`)

The main agent connects to Mindustry through the WebSocket mod and trains directly
in the live game environment.

### Philosophy (v4)

Previous versions had instability issues: val_loss exploding to 900+, agent
getting stuck in corners, episodes never ending. Version 4 fixes this with
three principles:

1. **Single primary signal**: core proximity. The agent always benefits from
   being near the base. This is a dense, continuous signal that never disappears.
2. **Running reward normalization** (Welford online algorithm). Guarantees
   val_loss stays below 10 at all times regardless of raw reward scale.
3. **Small, linear penalties** only. No large fixed penalties that cause instability.

### Observation space

Map tensor: 11 channels x 32x32 tiles (centered on agent)

| Channel | Content |
|---------|---------|
| 0 | Floor tile type (id / 255) |
| 1 | Copper ore |
| 2 | Core tiles |
| 3 | Drill tiles |
| 4 | Conveyor tiles |
| 5 | Wall tiles |
| 6 | Enemy positions (normalized count) |
| 7 | Building health (0..1) |
| 8 | Conveyor direction (0..3 / 4) |
| 9 | Agent position (1 at center) |
| 10 | Items in building / 100 |

Scalar vector: 16 values from the Java mod

| Index | Value |
|-------|-------|
| 0 | Unit health (0..1) |
| 1 | Tile X / 100 |
| 2 | Tile Y / 100 |
| 3 | Items in stack / 50 |
| 4 | Delta copper this step |
| 5 | isStuck (0 or 1) |
| 6 | Core direction X (normalized) |
| 7 | Core direction Y (normalized) |
| 8 | Nearest copper dist / 32 |
| 9 | Drill count / 20 |
| 10 | Enemy count / 10 |
| 11 | Core health (0..1) |
| 12 | Edge danger (0..1) |
| 13-15 | Reserved (zeros) |

### Action space

4 discrete action types with absolute (x, y) tile coordinates (32x32 grid):

| Type | Mapped to | Action |
|------|-----------|--------|
| 0 | idle | No build action |
| 1 | mechanicalDrill | Place drill |
| 2 | conveyor | Place conveyor |
| 3 | delete | Remove structure |

Plus a continuous 2D movement vector (vx, vy) sampled from a Normal distribution.

### Reward structure (v4)

| Event | Reward |
|-------|--------|
| Each step | -0.01 |
| Proximity to core | +1.0 x (1 - core_dist) |
| Moving closer to core | +3.0 x delta_distance |
| Moving away from core | -2.0 x delta_distance |
| Idle beyond grace period | -0.5 per step |
| Copper delivered | +5.0 x delta_copper |

All rewards are normalized by running standard deviation (Welford) and
clipped to [-10, 10] before being stored in the rollout buffer.

### Neural network

```
Map input (11 x 32 x 32)          Scalars (16)
         |                               |
   CNN (3 layers)                  MLP (2 layers)
   + ResBlocks + GroupNorm         + LayerNorm
   -> 4096 features                -> 128 features
         |________________ concat _______|
                            |
                  Manual LSTM (256 hidden)
                  (compatible with DirectML)
                    /     |      \
              Actor-T  Actor-X  Actor-Y      Critic
              (type)   (x tile) (y tile)     (V(s))
              4 log.   32 log.  32 log.      1 value
                            |
                     Move head (2D Normal)
                     continuous vx, vy
```

The LSTM is implemented manually (without `nn.LSTM`) because PyTorch's built-in
LSTM does not support backward on DirectML. The manual implementation uses
separate `nn.Linear` layers for input and hidden gates.

### Commands

```bash
# Train (connects to running Mindustry instance)
python mind_rl_agent.py

# Train for a fixed number of PPO updates
python mind_rl_agent.py --updates 500

# Use a different checkpoint file
python mind_rl_agent.py --checkpoint my_run.pt

# Disable reward normalization
python mind_rl_agent.py --no-norm
```

### Inspect a checkpoint

```bash
python test.py
```

Prints episode count, total steps, best reward, and model layer names.

---

## Part 3: Java WebSocket Mod

The mod opens a WebSocket server inside the game on port 6789.
The Python agent connects to this server and exchanges JSON messages
once per game step.

### Files

```
mod/
  MindustryMod.java     Mod entry point, handles game lifecycle events
  MindustryBridge.java  WebSocket server, action handling, state serialization
```

### Protocol

Every step the Python agent sends:

```json
{
  "vx": 0.4,
  "vy": -0.2,
  "type": 1,
  "x": 14,
  "y": 18,
  "delete": false
}
```

| Field | Description |
|-------|-------------|
| vx, vy | Movement vector, scaled by 15 in Java |
| type | Block index: 0=air, 1=drill, 2=conveyor, 3=wall, 4=turret |
| x, y | Target tile relative to agent center (agent is at tile 16,16) |
| delete | If true, remove the block at (x, y) |

The mod responds with a full state JSON:

```json
{
  "map": [[[...], ...], ...],
  "scalars": [0.95, 0.45, 0.32, ...],
  "done": false,
  "ready": true
}
```

A reset message (`{"reset": true}`) triggers `Vars.control.playMap()` and
waits up to 60 attempts (10 ticks each) for the game to become ready before
sending the first observation.

### Buildable blocks

| Index | Block |
|-------|-------|
| 0 | Air (no action) |
| 1 | mechanicalDrill |
| 2 | conveyor |
| 3 | copperWall |
| 4 | duo (turret) |

### State construction details

The mod builds a 43x43 view centered on the agent (11 channels) and 16 scalars
on every step. Key implementation details:

- **Stuck detection**: tracks position over 3 consecutive steps. Sets `scalars[5] = 1`
  if the unit has not moved more than 0.5 pixels in any step.
- **Delta copper**: difference between current and previous core copper count,
  clamped to non-negative. Normalized by dividing by 20.
- **Enemy map**: enemies within 32 tiles are marked on channel 6 at their relative
  grid position. Multiple enemies accumulate additively (clamped to 1).
- **Conveyor rotation**: stored as `rotation / 4` so it fits in 0..1.
- **Edge danger** (`scalars[12]`): distance to the nearest map edge, normalized
  so values near 0 mean the agent is safely inland and near 1 means it is at the edge.
- **Thread safety**: game state is built inside `Time.runTask(2f, ...)` to ensure
  the `Core.app.post()` action has already been applied before the snapshot is taken.

### Building

The mod depends on Java-WebSocket. Add to your `build.gradle`:

```gradle
dependencies {
    implementation 'org.java-websocket:Java-WebSocket:1.5.4'
}
```

Build and install:

```bash
./gradlew build
# Copy the output .jar to your Mindustry/mods/ folder
```

Then start Mindustry, load a map, start a game, and run the Python agent.

---

## Transferring Weights from Simulator to Real Game

The simulator uses 8 map channels and 10 scalars.
The real game agent uses 11 map channels and 16 scalars.
The architectures are intentionally different because the real game provides
richer observations. Direct weight transfer is not straightforward.

Recommended approach:

1. Train the simulator agent until `copper` averages above 50 per episode.
2. Use the simulator-trained weights to initialize the CNN and LSTM layers only
   (the input conv layer will need reinitialization since channel counts differ).
3. Fine-tune in the real game with a reduced learning rate (divide `lr` by 10).
4. The LSTM and value head transfer cleanly. The actor heads also transfer since
   the action space is the same.

For a fast demo without transfer, just run `mind_rl_agent.py` from scratch.
The v4 reward shaping is stable enough that the agent begins showing useful
behavior within a few hundred episodes.

---

## Device Notes

Both scripts auto-detect DirectML (AMD GPU) and fall back to CPU.

```python
try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    device = torch.device("cpu")
```

DirectML does not support PyTorch's backward pass through `nn.LSTM`,
which is why both agents implement the LSTM manually using `nn.Linear` layers.
The PPO update step runs on CPU and the model is moved back to the DirectML
device afterward for rollout collection.

On an AMD RX 6550M, the simulator typically achieves 60-70% GPU utilization.
The real game agent stays at 10-15% because the bottleneck is the WebSocket
round-trip, not compute.

---

## Training Tips

- Run `--test` first to confirm the simulator generates valid maps before starting training.
- In the simulator, `copper` should start growing around episode 200-500.
  If it does not, increase `entropy_coef` to 0.05 in `SimConfig`.
- In the real game, watch `reward_std` in the PPO log line. If it grows above 5.0
  the reward scale is unstable; reduce `r_near_base` or `r_approach_base`.
- `val_loss` above 1.0 for many consecutive updates indicates the critic is struggling.
  Reduce `lr` or increase `rollout_steps`.
- Shorter rollouts (`rollout_steps = 256`) mean more frequent updates but noisier gradients.
  Longer rollouts (`rollout_steps = 1024`) are more stable but slower to adapt.
- The LSTM hidden state is reset at the start of each episode. If episodes are very
  short (under 100 steps) the LSTM has little time to build useful context.
  Consider increasing `max_steps` or `max_updates` accordingly.