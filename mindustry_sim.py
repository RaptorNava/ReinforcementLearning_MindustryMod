"""
mind_rl_agent.py — PPO агент для Mindustry (через WebSocket мод)

КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ v3:
  - Катастрофический штраф за нахождение у края/угла карты (экспоненциальный рост)
  - Штраф за стояние на месте растёт быстро и не имеет потолка (только мягкий cap)
  - Добавлен "diversity bonus" — агент получает бонус за посещение новых клеток
  - Убрана нормализация наград (мешает агенту чувствовать масштаб катастрофы)
  - Штраф за угол карты — отдельная компонента, очень большая
  - Эксплорация поощряется: бонус за уникальные посещённые тайлы
  - Entropy coef повышен для борьбы с преждевременной сходимостью
"""

import asyncio
import argparse
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────
@dataclass
class Config:
    # Наблюдения
    map_channels: int = 11
    scalar_obs_size: int = 16
    map_size: int = 32          # тайлов на сторону карты (для нормализации)

    # Действия
    n_action_types: int = 4     # 0=idle, 1=drill, 2=conv, 3=delete
    action_type_map: tuple = (0, 1, 2, 5)
    n_action_x: int = 32
    n_action_y: int = 32

    # PPO гиперпараметры
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02  # ↑ повышен: нужно больше исследования
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 1024

    # Архитектура
    lstm_hidden: int = 256
    move_std: float = 0.4       # чуть больше шума в движении

    # ── Основные награды ──────────────────────────────────────
    r_step_pen: float = -0.002      # маленький штраф за каждый шаг
    r_copper_mult: float = 200.0    # большой бонус за добычу меди

    # ── Движение и исследование ───────────────────────────────
    move_tile_epsilon: float = 0.5  # порог "сдвинулся"

    # Штраф за стояние на месте (нет движения)
    no_move_grace_steps: int = 5    # шагов прощения
    r_no_move_base: float = -5.0    # базовый штраф
    r_no_move_per_step: float = -3.0 # растёт с каждым шагом застревания
    r_no_move_cap: float = -120.0   # жёсткий потолок (катастрофа)

    # Бонус за посещение новых тайлов (эксплорация)
    r_explore_new_tile: float = 2.0
    explore_grid_size: int = 4      # группируем тайлы в клетки 4×4

    # ── Штраф за угол/край карты ──────────────────────────────
    # Работает как: penalty = r_edge_base * (edge_factor ** r_edge_power)
    # edge_factor = насколько близко к краю (0..1, 1 = самый край)
    r_edge_base: float = -2.0
    r_edge_power: float = 3.0       # экспоненциальный рост у края
    edge_margin: float = 0.15       # начинает штрафовать при dist_to_edge < 15% карты
    # Дополнительный огромный штраф за попадание в угол
    r_corner_penalty: float = -50.0
    corner_margin: float = 0.08     # угол = одновременно < 8% по X и Y

    # ── Штраф за удалённость от ядра ─────────────────────────
    away_core_threshold: float = 0.55
    away_core_grace_steps: int = 20
    r_away_core_base: float = -1.0
    r_away_core_per_step: float = -0.15
    r_away_core_cap: float = -30.0

    # Бонус за приближение к ядру
    r_core_progress: float = 1.0
    r_core_proximity_max: float = 0.5  # макс. бонус за близость к ядру

    # ── Строительство ─────────────────────────────────────────
    r_drill_on_copper: float = 8.0
    r_useless_build: float = -2.0
    r_conveyor_to_core: float = 1.0

    # ── Здоровье / враги ──────────────────────────────────────
    r_health_loss: float = -4.0
    r_stuck_pen: float = -3.0
    r_enemy_near: float = -0.8

    # ── Чекпоинт и логирование ────────────────────────────────
    checkpoint_path: str = "mindustry_agent.pt"
    log_interval: int = 10
    save_interval_updates: int = 1
    max_updates: int = 0

    # WebSocket
    ws_host: str = "localhost"
    ws_port: int = 6789


cfg = Config()

try:
    import torch_directml
    device = torch_directml.device()
    print("[Device] DirectML (AMD RX6550M)")
except ImportError:
    device = torch.device("cpu")
    print("[Device] CPU")


# ─────────────────────────────────────────────
#  НЕЙРОСЕТЬ
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(x + self.net(x))


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN (карта 32×32 × 11 каналов)
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.map_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # → 16×16
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            ResBlock(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   # → 8×8
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.Flatten(),                                  # → 4096
        )
        cnn_out = 64 * 8 * 8  # 4096

        # MLP для скалярных признаков
        self.mlp = nn.Sequential(
            nn.Linear(cfg.scalar_obs_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 128),
            nn.ReLU(inplace=False),
        )

        # Ручной LSTM (совместим с DirectML)
        lstm_input = cnn_out + 128
        self.lstm_ih = nn.Linear(lstm_input, cfg.lstm_hidden * 4)
        self.lstm_hh = nn.Linear(cfg.lstm_hidden, cfg.lstm_hidden * 4)

        # Иерархическая голова актора: тип → x → y
        self.actor_type = nn.Linear(cfg.lstm_hidden, cfg.n_action_types)
        self.type_embed = nn.Linear(cfg.n_action_types, 32, bias=False)
        self.actor_x    = nn.Linear(cfg.lstm_hidden + 32, cfg.n_action_x)
        self.x_embed    = nn.Linear(cfg.n_action_x, 32, bias=False)
        self.actor_y    = nn.Linear(cfg.lstm_hidden + 32 + 32, cfg.n_action_y)

        # Непрерывное движение
        self.actor_move_mean = nn.Linear(cfg.lstm_hidden, 2)
        self.move_log_std = nn.Parameter(
            torch.full((2,), np.log(cfg.move_std), dtype=torch.float32)
        )

        # Критик
        self.critic = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_type.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_x.weight,    gain=0.01)
        nn.init.orthogonal_(self.actor_y.weight,    gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def init_state(self, batch=1):
        z = torch.zeros(batch, cfg.lstm_hidden, device=device)
        return (z, z)

    def _lstm_step(self, features, h, c):
        gates = self.lstm_ih(features) + self.lstm_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def act(self, m, s, state):
        if m.dim() == 3: m = m.unsqueeze(0)
        if s.dim() == 1: s = s.unsqueeze(0)

        features = torch.cat([self.cnn(m), self.mlp(s)], dim=-1)
        h, c = self._lstm_step(features, state[0], state[1])

        # Тип действия
        type_logits = self.actor_type(h)
        type_dist   = torch.distributions.Categorical(logits=type_logits)
        chosen_t    = type_dist.sample()

        # X (зависит от типа)
        t_oh = F.one_hot(chosen_t.cpu(), cfg.n_action_types).float().to(device)
        te   = self.type_embed(t_oh)
        x_dist  = torch.distributions.Categorical(
            logits=self.actor_x(torch.cat([h, te], -1))
        )
        chosen_x = x_dist.sample()

        # Y (зависит от типа и X)
        x_oh = F.one_hot(chosen_x.cpu(), cfg.n_action_x).float().to(device)
        xe   = self.x_embed(x_oh)
        y_dist  = torch.distributions.Categorical(
            logits=self.actor_y(torch.cat([h, te, xe], -1))
        )
        chosen_y = y_dist.sample()

        # Непрерывное движение
        move_mean = torch.tanh(self.actor_move_mean(h))
        move_std  = self.move_log_std.exp().expand_as(move_mean)
        move_dist = torch.distributions.Normal(move_mean, move_std)
        move_raw  = move_dist.rsample()

        log_prob = (
            type_dist.log_prob(chosen_t)
            + x_dist.log_prob(chosen_x)
            + y_dist.log_prob(chosen_y)
            + move_dist.log_prob(move_raw).sum(-1)
        )
        entropy = (
            type_dist.entropy()
            + x_dist.entropy()
            + y_dist.entropy()
            + move_dist.entropy().sum(-1)
        )

        return {
            "value":      self.critic(h).squeeze(-1),
            "next_state": (h, c),
            "type":       chosen_t,
            "x":          chosen_x,
            "y":          chosen_y,
            "move":       torch.tanh(move_raw),
            "move_raw":   move_raw,
            "log_prob":   log_prob,
            "entropy":    entropy,
        }

    def evaluate(self, m_batch, s_batch, states_batch,
                 t_batch, x_batch, y_batch, move_raw_batch):
        B = m_batch.size(0)
        features = torch.cat([self.cnn(m_batch), self.mlp(s_batch)], dim=-1)
        h0, c0 = states_batch
        h, c = self._lstm_step(features, h0, c0)

        type_dist = torch.distributions.Categorical(logits=self.actor_type(h))
        t_oh = F.one_hot(t_batch.cpu(), cfg.n_action_types).float()
        te   = self.type_embed(t_oh)
        x_dist = torch.distributions.Categorical(
            logits=self.actor_x(torch.cat([h, te], -1))
        )
        x_oh = F.one_hot(x_batch.cpu(), cfg.n_action_x).float()
        xe   = self.x_embed(x_oh)
        y_dist = torch.distributions.Categorical(
            logits=self.actor_y(torch.cat([h, te, xe], -1))
        )
        move_mean = torch.tanh(self.actor_move_mean(h))
        move_std  = self.move_log_std.exp().expand_as(move_mean)
        move_dist = torch.distributions.Normal(move_mean, move_std)

        log_prob = (
            type_dist.log_prob(t_batch)
            + x_dist.log_prob(x_batch)
            + y_dist.log_prob(y_batch)
            + move_dist.log_prob(move_raw_batch).sum(-1)
        )
        entropy = (
            type_dist.entropy()
            + x_dist.entropy()
            + y_dist.entropy()
            + move_dist.entropy().sum(-1)
        )
        return log_prob, entropy, self.critic(h).squeeze(-1)


# ─────────────────────────────────────────────
#  БУФЕР РОЛЛАУТОВ
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.maps      = []
        self.scalars   = []
        self.hx        = []
        self.cx        = []
        self.types     = []
        self.xs        = []
        self.ys        = []
        self.move_raws = []
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.dones     = []

    def add(self, obs, hx, cx, action, log_prob, value, reward, done):
        self.maps.append(obs["m"])
        self.scalars.append(obs["s"])
        self.hx.append(hx.squeeze(0).detach())
        self.cx.append(cx.squeeze(0).detach())
        self.types.append(action["type"].squeeze(0).detach())
        self.xs.append(action["x"].squeeze(0).detach())
        self.ys.append(action["y"].squeeze(0).detach())
        self.move_raws.append(action["move_raw"].squeeze(0).detach())
        self.log_probs.append(log_prob.squeeze(0).detach())
        self.values.append(value.squeeze(0).detach())
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, last_value, last_done):
        T = len(self.rewards)
        advantages = torch.zeros(T, device=device)
        returns    = torch.zeros(T, device=device)
        gae = 0.0
        next_val  = last_value
        next_done = last_done
        for t in reversed(range(T)):
            mask   = 1.0 - float(self.dones[t])
            delta  = self.rewards[t] + cfg.gamma * next_val * mask - self.values[t].item()
            gae    = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t]    = gae + self.values[t].item()
            next_val  = self.values[t].item()
            next_done = self.dones[t]
        return advantages, returns

    def get_tensors(self):
        return {
            "maps":      torch.tensor(np.array(self.maps),      dtype=torch.float32, device=device),
            "scalars":   torch.tensor(np.array(self.scalars),   dtype=torch.float32, device=device),
            "hx":        torch.stack(self.hx),
            "cx":        torch.stack(self.cx),
            "types":     torch.stack(self.types),
            "xs":        torch.stack(self.xs),
            "ys":        torch.stack(self.ys),
            "move_raws": torch.stack(self.move_raws),
            "log_probs": torch.stack(self.log_probs),
            "values":    torch.stack(self.values),
        }


# ─────────────────────────────────────────────
#  СРЕДА (WebSocket)
# ─────────────────────────────────────────────
class MindustryEnv:
    """
    Скаляры (16 штук):
      0  - здоровье юнита (0..1)
      1  - tileX / 100
      2  - tileY / 100
      3  - кол-во предметов в стеке / 50
      4  - delta copper (добыча за шаг)
      5  - isStuck (0 или 1)
      6  - coreDx (-1..1)
      7  - coreDy (-1..1)
      8  - nearestCopperDist / 32
      9  - drillCount / 20
      10 - enemyCount / 10
      11 - coreHealth (0..1)
      12..15 - зарезервировано
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._q = queue.Queue()
        self._connected = threading.Event()
        self._ws = None
        threading.Thread(target=self._run_loop, daemon=True).start()
        ok = self._connected.wait(timeout=20.0)
        if not ok:
            raise RuntimeError("Не удалось подключиться к Mindustry за 20 секунд.")
        self._reset_state()

    def _reset_state(self):
        self._prev_health      = 1.0
        self._prev_tile_x      = None
        self._prev_tile_y      = None
        self._stationary_steps = 0
        self._away_core_steps  = 0
        self._prev_core_dist   = None
        # Карта посещённых клеток (для бонуса за исследование)
        self._visited_cells: set = set()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        import websockets

        async def connect():
            while True:
                try:
                    async with websockets.connect(
                        f"ws://{cfg.ws_host}:{cfg.ws_port}",
                        ping_interval=20,
                        ping_timeout=10,
                    ) as ws:
                        self._ws = ws
                        self._connected.set()
                        while True:
                            raw = await ws.recv()
                            self._q.put(json.loads(raw))
                except Exception as e:
                    print(f"[WS] reconnecting... ({e})")
                    self._connected.clear()
                    await asyncio.sleep(2)

        self._loop.run_until_complete(connect())

    def _send(self, payload: dict):
        asyncio.run_coroutine_threadsafe(
            self._ws.send(json.dumps(payload)), self._loop
        ).result(timeout=5.0)

    def _drain_queue(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _obs_from_response(self, resp):
        expected_map_shape = (cfg.map_channels, cfg.map_size, cfg.map_size)
        m = np.asarray(
            resp.get("map", np.zeros(expected_map_shape)),
            dtype=np.float32,
        )
        if m.shape != expected_map_shape:
            m = np.zeros(expected_map_shape, dtype=np.float32)

        s = np.asarray(resp.get("scalars", [0.0] * cfg.scalar_obs_size), dtype=np.float32)
        if s.shape != (cfg.scalar_obs_size,):
            fixed = np.zeros(cfg.scalar_obs_size, dtype=np.float32)
            fixed[:min(cfg.scalar_obs_size, s.size)] = s.reshape(-1)[:cfg.scalar_obs_size]
            s = fixed

        return {"m": m, "s": s}

    # ── Расчёт награды ────────────────────────────────────────
    def _compute_reward(self, s, m, env_act, act_x, act_y):
        r = cfg.r_step_pen

        # 1. Добыча меди
        r += float(s[4]) * cfg.r_copper_mult

        # 2. Застрял (isStuck от мода)
        if float(s[5]) > 0.5:
            r += cfg.r_stuck_pen

        # ── Позиция агента ────────────────────────────────────
        tile_x = float(s[1]) * 100.0
        tile_y = float(s[2]) * 100.0

        # 3. Штраф за стояние на месте
        if self._prev_tile_x is not None:
            moved = (
                abs(tile_x - self._prev_tile_x) > cfg.move_tile_epsilon
                or abs(tile_y - self._prev_tile_y) > cfg.move_tile_epsilon
            )
            if moved:
                self._stationary_steps = 0
            else:
                self._stationary_steps += 1

            over = max(0, self._stationary_steps - cfg.no_move_grace_steps)
            if over > 0:
                # Растёт линейно, но быстро
                pen = cfg.r_no_move_base + cfg.r_no_move_per_step * over
                r += max(cfg.r_no_move_cap, pen)

        # 4. Бонус за исследование новых тайлов
        cell_key = (
            int(tile_x) // cfg.explore_grid_size,
            int(tile_y) // cfg.explore_grid_size,
        )
        if cell_key not in self._visited_cells:
            self._visited_cells.add(cell_key)
            r += cfg.r_explore_new_tile

        self._prev_tile_x = tile_x
        self._prev_tile_y = tile_y

        # 5. Штраф за край/угол карты
        #    tile_x/tile_y нормированы через /100, а карта cfg.map_size тайлов
        #    Используем нормированные координаты (0..1)
        nx = tile_x / (cfg.map_size * 1.0 + 1e-6)
        ny = tile_y / (cfg.map_size * 1.0 + 1e-6)
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))

        # Расстояние до ближайшего края (0 = на краю, 0.5 = в центре)
        dist_x = min(nx, 1.0 - nx)
        dist_y = min(ny, 1.0 - ny)
        dist_edge = min(dist_x, dist_y)

        if dist_edge < cfg.edge_margin:
            # Насколько мы внутри опасной зоны (0..1, 1 = прямо на краю)
            edge_factor = 1.0 - (dist_edge / cfg.edge_margin)
            edge_pen = cfg.r_edge_base * (edge_factor ** cfg.r_edge_power)
            r += edge_pen

        # Угловой штраф: близко к краю сразу по обеим осям
        if dist_x < cfg.corner_margin and dist_y < cfg.corner_margin:
            r += cfg.r_corner_penalty
            print(f"[CORNER] агент в углу! tile=({tile_x:.1f},{tile_y:.1f}) pen={cfg.r_corner_penalty}")

        # 6. Ядро: прогресс и близость
        core_dx = float(s[6]) if len(s) > 6 else 0.0
        core_dy = float(s[7]) if len(s) > 7 else 0.0
        core_dist = float(np.clip(np.sqrt(core_dx**2 + core_dy**2), 0.0, 1.0))

        # Бонус за близость к ядру
        proximity = max(0.0, 1.0 - core_dist * 2.0)
        r += proximity * cfg.r_core_proximity_max

        # Бонус за движение к ядру
        if self._prev_core_dist is not None:
            r += (self._prev_core_dist - core_dist) * cfg.r_core_progress
        self._prev_core_dist = core_dist

        # Штраф за длительное удаление от ядра
        if core_dist > cfg.away_core_threshold:
            self._away_core_steps += 1
        else:
            self._away_core_steps = 0

        over_core = max(0, self._away_core_steps - cfg.away_core_grace_steps)
        if over_core > 0:
            away_pen = cfg.r_away_core_base + cfg.r_away_core_per_step * over_core
            r += max(cfg.r_away_core_cap, away_pen)

        # 7. Потеря здоровья
        health_loss = self._prev_health - float(s[0])
        if health_loss > 0.01:
            r += health_loss * cfg.r_health_loss
        self._prev_health = float(s[0])

        # 8. Строительство
        if env_act == 1:  # drill
            tile_xi = max(0, min(31, act_x))
            tile_yi = max(0, min(31, act_y))
            if m.shape[0] > 1 and m[1, tile_yi, tile_xi] > 0.5:
                r += cfg.r_drill_on_copper
            else:
                r += cfg.r_useless_build

        elif env_act == 2:  # conveyor
            tx_rel = act_x - cfg.map_size // 2
            ty_rel = act_y - cfg.map_size // 2
            toward_core = (tx_rel * core_dx + ty_rel * core_dy) > 0
            r += cfg.r_conveyor_to_core if toward_core else -cfg.r_conveyor_to_core

        # 9. Враги рядом
        if len(s) > 10:
            r += float(s[10]) * cfg.r_enemy_near

        return r

    def step(self, action):
        env_act = int(cfg.action_type_map[int(action["type"])])
        act_x   = int(action["x"])
        act_y   = int(action["y"])

        payload = {
            "vx":     float(action["vx"]),
            "vy":     float(action["vy"]),
            "type":   env_act,
            "x":      act_x,
            "y":      act_y,
            "delete": bool(env_act == 5),
        }
        self._send(payload)

        resp = self._q.get(timeout=10.0)
        obs  = self._obs_from_response(resp)
        done = bool(resp.get("done", False))

        reward = self._compute_reward(obs["s"], obs["m"], env_act, act_x, act_y)
        return obs, reward, done

    def reset(self):
        self._drain_queue()
        self._send({"reset": True})
        self._reset_state()

        deadline = time.time() + 20.0
        last_resp = None
        while time.time() < deadline:
            try:
                resp = self._q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                break
            last_resp = resp
            if resp.get("ready", False) and not resp.get("done", False):
                obs = self._obs_from_response(resp)
                s = obs["s"]
                self._prev_health  = float(s[0])
                self._prev_tile_x  = float(s[1]) * 100.0
                self._prev_tile_y  = float(s[2]) * 100.0
                core_dx = float(s[6]) if len(s) > 6 else 0.0
                core_dy = float(s[7]) if len(s) > 7 else 0.0
                self._prev_core_dist = float(np.clip(np.sqrt(core_dx**2 + core_dy**2), 0.0, 1.0))
                return obs

        raise RuntimeError(
            f"Mindustry не вернула ready после reset. Последний ответ: {last_resp}"
        )


# ─────────────────────────────────────────────
#  PPO ТРЕНЕР
# ─────────────────────────────────────────────
class PPOTrainer:
    def __init__(self, model: ActorCritic):
        self.model     = model
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )

    def update(self, buffer: RolloutBuffer, last_value: float, last_done: bool):
        advantages, returns = buffer.compute_gae(last_value, last_done)

        adv_cpu = advantages.cpu()
        adv_cpu = (adv_cpu - adv_cpu.mean()) / (adv_cpu.std() + 1e-8)

        # backward на CPU (DirectML не поддерживает scatter/backward)
        self.model.cpu()
        data = buffer.get_tensors()
        cpu_data = {k: v.cpu() for k, v in data.items()}
        returns_cpu = returns.cpu()

        T = len(buffer.rewards)
        total_pol = total_val = total_ent = 0.0
        n = 0

        for _ in range(cfg.ppo_epochs):
            idxs = torch.randperm(T)
            for start in range(0, T, cfg.batch_size):
                b = idxs[start : start + cfg.batch_size]

                log_prob_new, entropy, value_new = self.model.evaluate(
                    cpu_data["maps"][b],
                    cpu_data["scalars"][b],
                    (cpu_data["hx"][b], cpu_data["cx"][b]),
                    cpu_data["types"][b],
                    cpu_data["xs"][b],
                    cpu_data["ys"][b],
                    cpu_data["move_raws"][b],
                )

                ratio = (log_prob_new - cpu_data["log_probs"][b]).exp()
                adv_b = adv_cpu[b]

                loss_pol = -torch.min(
                    ratio * adv_b,
                    ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b,
                ).mean()

                val_old     = cpu_data["values"][b]
                val_clipped = val_old + (value_new - val_old).clamp(-cfg.clip_eps, cfg.clip_eps)
                loss_val = torch.max(
                    (value_new - returns_cpu[b]) ** 2,
                    (val_clipped - returns_cpu[b]) ** 2,
                ).mean()

                loss = (
                    loss_pol
                    + cfg.value_loss_coef * loss_val
                    - cfg.entropy_coef    * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pol += loss_pol.item()
                total_val += loss_val.item()
                total_ent += entropy.mean().item()
                n += 1

        self.model.to(device)
        self.scheduler.step()
        return {
            "loss_pol": total_pol / n,
            "loss_val": total_val / n,
            "entropy":  total_ent  / n,
        }


# ─────────────────────────────────────────────
#  ЧЕКПОИНТ
# ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, episode, total_steps, total_updates, best_reward):
    torch.save({
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "episode":       episode,
        "total_steps":   total_steps,
        "total_updates": total_updates,
        "best_reward":   best_reward,
    }, cfg.checkpoint_path)
    print(f"[Checkpoint] saved: updates={total_updates}, ep={episode}, steps={total_steps}")


def load_checkpoint(model, optimizer, scheduler):
    if not os.path.exists(cfg.checkpoint_path):
        return 0, 0, 0, -float("inf")
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    total_updates = int(ckpt.get("total_updates", 0))
    print(f"[Checkpoint] loaded: updates={total_updates}, ep={ckpt['episode']}, steps={ckpt['total_steps']}")
    return ckpt["episode"], ckpt["total_steps"], total_updates, ckpt["best_reward"]


# ─────────────────────────────────────────────
#  ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ
# ─────────────────────────────────────────────
def train(max_updates=None):
    print("=" * 60)
    print("  Mindustry RL Agent — PPO+LSTM v3 (anti-corner)")
    print(f"  Device: {device}")
    print("=" * 60)

    env     = MindustryEnv()
    model   = ActorCritic().to(device)
    trainer = PPOTrainer(model)

    start_ep, total_steps, total_updates, best_reward = load_checkpoint(
        model, trainer.optimizer, trainer.scheduler
    )

    buffer     = RolloutBuffer()
    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_explore = deque(maxlen=100)   # сколько новых клеток за эпизод

    episode    = start_ep
    ep_reward  = 0.0
    ep_len     = 0
    obs        = env.reset()
    lstm_state = model.init_state(batch=1)
    target_updates = cfg.max_updates if max_updates is None else max_updates

    print("\n[Train] Старт обучения...\n")

    try:
        while target_updates <= 0 or total_updates < target_updates:
            model.eval()
            buffer.reset()

            for step in range(cfg.rollout_steps):
                m_t = torch.tensor(obs["m"], dtype=torch.float32, device=device).unsqueeze(0)
                s_t = torch.tensor(obs["s"], dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    out = model.act(m_t, s_t, lstm_state)

                action = {
                    "type":     out["type"],
                    "x":        out["x"],
                    "y":        out["y"],
                    "vx":       float(out["move"][0, 0].cpu()),
                    "vy":       float(out["move"][0, 1].cpu()),
                    "move_raw": out["move_raw"],
                }

                next_obs, reward, done = env.step(action)
                ep_reward  += reward
                ep_len     += 1
                total_steps += 1

                buffer.add(
                    obs=obs,
                    hx=lstm_state[0],
                    cx=lstm_state[1],
                    action=out,
                    log_prob=out["log_prob"],
                    value=out["value"],
                    reward=reward,
                    done=done,
                )

                lstm_state = out["next_state"]
                obs = next_obs

                if done:
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(ep_len)
                    ep_explore.append(len(env._visited_cells))
                    episode   += 1
                    ep_reward  = 0.0
                    ep_len     = 0
                    obs        = env.reset()
                    lstm_state = model.init_state(batch=1)

                    if episode % cfg.log_interval == 0:
                        mean_r = np.mean(ep_rewards)
                        mean_l = np.mean(ep_lengths)
                        mean_e = np.mean(ep_explore)
                        print(
                            f"[Ep {episode:5d} | Steps {total_steps:8d}] "
                            f"reward100={mean_r:+8.2f} | "
                            f"len={mean_l:5.0f} | "
                            f"explore={mean_e:4.0f} cells"
                        )
                        if mean_r > best_reward:
                            best_reward = mean_r
                            print(f"  ★ Новый рекорд: {best_reward:.2f}")

            # Bootstrap последнего состояния
            m_t = torch.tensor(obs["m"], dtype=torch.float32, device=device).unsqueeze(0)
            s_t = torch.tensor(obs["s"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                out_last = model.act(m_t, s_t, lstm_state)
            last_value = float(out_last["value"].squeeze().cpu())

            model.train()
            stats = trainer.update(buffer, last_value, last_done=False)
            total_updates += 1

            print(
                f"[PPO] upd={total_updates} steps={total_steps} | "
                f"pol={stats['loss_pol']:.4f} val={stats['loss_val']:.4f} "
                f"ent={stats['entropy']:.4f}"
            )

            if total_updates % cfg.save_interval_updates == 0:
                save_checkpoint(
                    model, trainer.optimizer, trainer.scheduler,
                    episode, total_steps, total_updates, best_reward
                )

    except KeyboardInterrupt:
        print("\n[Train] Прервано, сохраняем чекпоинт...")
        save_checkpoint(
            model, trainer.optimizer, trainer.scheduler,
            episode, total_steps, total_updates, best_reward
        )
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates",    type=int, default=cfg.max_updates)
    parser.add_argument("--checkpoint", type=str, default=cfg.checkpoint_path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.checkpoint_path = args.checkpoint
    train(max_updates=args.updates)