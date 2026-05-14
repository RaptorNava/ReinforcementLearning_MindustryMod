"""
mind_rl_agent.py — PPO агент для Mindustry (WebSocket мод) v4
=============================================================

ФИЛОСОФИЯ v4: ПРОСТОТА И СТАБИЛЬНОСТЬ
  Проблемы предыдущих версий:
    - val_loss взрывалась до 900+ из-за огромных несбалансированных штрафов
    - Агент забивался в угол потому что там нет штрафов за движение к базе
    - episode=0 значит done никогда не приходит — эпизод бесконечный

  Что делаем:
    1. ЕДИНСТВЕННЫЙ главный сигнал: близость к ядру (core proximity)
       Агенту ВСЕГДА выгодно быть рядом с базой. Это непрерывный плотный сигнал.
    2. Нормализация наград через running mean/std (welford online)
       Гарантирует val_loss < 10 всегда.
    3. Штраф за неподвижность — маленький и стабильный, не взрывается.
    4. Убраны все огромные числа (штрафы -50, -120 и т.д.)
    5. Entropy coef = 0.03 — агент активно исследует

  Скаляры от мода (индексы):
    0  - здоровье юнита (0..1)
    1  - tileX / 100
    2  - tileY / 100
    3  - items в стеке / 50
    4  - delta copper за шаг
    5  - isStuck (0/1)
    6  - coreDx (нормированный вектор к ядру, -1..1)
    7  - coreDy
    8  - nearestCopperDist / 32
    9  - drillCount / 20
    10 - enemyCount / 10
    11 - coreHealth (0..1)
    12..15 - резерв
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


# ═══════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════
@dataclass
class Config:
    # Наблюдения
    map_channels: int    = 11
    scalar_obs_size: int = 16
    map_size: int        = 32

    # Действия
    n_action_types: int  = 4          # 0=idle,1=drill,2=conv,3=delete
    action_type_map: tuple = (0, 1, 2, 5)
    n_action_x: int      = 32
    n_action_y: int      = 32

    # PPO
    lr: float            = 2e-4
    gamma: float         = 0.99
    gae_lambda: float    = 0.95
    clip_eps: float      = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float  = 0.03       # высокий — нужно исследование
    max_grad_norm: float = 0.5
    ppo_epochs: int      = 3
    batch_size: int      = 64
    rollout_steps: int   = 512        # короче роллаут = чаще обновления

    # Архитектура
    lstm_hidden: int     = 256
    move_std: float      = 0.5        # больше шум движения = больше исследования

    # ── Reward shaping ──────────────────────────────────
    # Главный сигнал: близость к ядру
    # r = r_near_base * (1 - core_dist)  ->  0..r_near_base
    # Агент получает максимум когда стоит прямо у базы
    r_near_base: float   = 1.0

    # Бонус за движение БЛИЖЕ к базе (delta distance)
    r_approach_base: float = 3.0

    # Штраф за движение ДАЛЬШЕ от базы
    r_leave_base: float  = -2.0

    # Штраф за неподвижность (маленький, стабильный)
    no_move_grace: int   = 10         # шагов прощения
    r_no_move: float     = -0.5       # за каждый шаг сверх grace

    # Медь (вторичный сигнал)
    r_copper_mult: float = 5.0

    # Маленький штраф за каждый шаг (чтобы не тянул время)
    r_step: float        = -0.01

    # Порог движения (тайлов)
    move_epsilon: float  = 0.3

    # Нормализация наград (running welford)
    reward_norm: bool    = True
    reward_clip: float   = 10.0       # клиппируем нормализованные награды

    # Логирование / чекпоинт
    checkpoint_path: str = "mindustry_agent.pt"
    log_interval: int    = 5
    save_interval: int   = 1
    max_updates: int     = 0

    # WebSocket
    ws_host: str         = "localhost"
    ws_port: int         = 6789


cfg = Config()

try:
    import torch_directml
    device = torch_directml.device()
    print("[Device] DirectML (AMD RX6550M)")
except ImportError:
    device = torch.device("cpu")
    print("[Device] CPU")


# ═══════════════════════════════════════════════════════
#  RUNNING REWARD NORMALIZER (Welford online stats)
# ═══════════════════════════════════════════════════════
class RunningNorm:
    """Нормализует награды через бегущие mean/variance (алгоритм Welford)."""
    def __init__(self, clip=10.0):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 1.0
        self.clip = clip

    def update_and_normalize(self, r: float) -> float:
        self.n += 1
        delta   = r - self.mean
        self.mean += delta / self.n
        delta2  = r - self.mean
        self.M2 += delta * delta2
        var = self.M2 / max(self.n - 1, 1)
        std = max(np.sqrt(var), 1e-6)
        return float(np.clip(r / std, -self.clip, self.clip))

    def state_dict(self):
        return {"n": self.n, "mean": self.mean, "M2": self.M2}

    def load_state_dict(self, d):
        self.n    = d.get("n",    0)
        self.mean = d.get("mean", 0.0)
        self.M2   = d.get("M2",  1.0)


# ═══════════════════════════════════════════════════════
#  НЕЙРОСЕТЬ
# ═══════════════════════════════════════════════════════
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
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(x + self.net(x))


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN: 32x32x11 -> flatten 4096
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.map_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            ResBlock(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 8x8
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.Flatten(),
        )
        cnn_out = 64 * 8 * 8  # 4096

        # MLP для скаляров
        self.mlp = nn.Sequential(
            nn.Linear(cfg.scalar_obs_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 128),
            nn.ReLU(inplace=False),
        )

        # Ручной LSTM (совместим с DirectML: нет scatter/backward)
        lstm_in = cnn_out + 128
        self.lstm_ih = nn.Linear(lstm_in, cfg.lstm_hidden * 4)
        self.lstm_hh = nn.Linear(cfg.lstm_hidden, cfg.lstm_hidden * 4)

        # Иерархический актор: тип -> x -> y
        self.actor_type = nn.Linear(cfg.lstm_hidden, cfg.n_action_types)
        self.type_emb   = nn.Linear(cfg.n_action_types, 32, bias=False)
        self.actor_x    = nn.Linear(cfg.lstm_hidden + 32, cfg.n_action_x)
        self.x_emb      = nn.Linear(cfg.n_action_x, 32, bias=False)
        self.actor_y    = nn.Linear(cfg.lstm_hidden + 64, cfg.n_action_y)

        # Непрерывное движение
        self.actor_move = nn.Linear(cfg.lstm_hidden, 2)
        self.log_std    = nn.Parameter(
            torch.full((2,), np.log(cfg.move_std))
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
        for head in (self.actor_type, self.actor_x, self.actor_y):
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def init_state(self, batch=1):
        z = torch.zeros(batch, cfg.lstm_hidden, device=device)
        return (z, z)

    def _lstm(self, feat, h, c):
        gates = self.lstm_ih(feat) + self.lstm_hh(h)
        ig, fg, gg, og = gates.chunk(4, dim=-1)
        c2 = torch.sigmoid(fg) * c + torch.sigmoid(ig) * torch.tanh(gg)
        h2 = torch.sigmoid(og) * torch.tanh(c2)
        return h2, c2

    def act(self, m, s, state):
        if m.dim() == 3: m = m.unsqueeze(0)
        if s.dim() == 1: s = s.unsqueeze(0)
        feat   = torch.cat([self.cnn(m), self.mlp(s)], -1)
        h, c   = self._lstm(feat, state[0], state[1])

        # Тип действия
        t_dist = torch.distributions.Categorical(logits=self.actor_type(h))
        t      = t_dist.sample()
        te     = self.type_emb(F.one_hot(t.cpu(), cfg.n_action_types).float().to(device))

        # X
        x_dist = torch.distributions.Categorical(
            logits=self.actor_x(torch.cat([h, te], -1))
        )
        x  = x_dist.sample()
        xe = self.x_emb(F.one_hot(x.cpu(), cfg.n_action_x).float().to(device))

        # Y
        y_dist = torch.distributions.Categorical(
            logits=self.actor_y(torch.cat([h, te, xe], -1))
        )
        y = y_dist.sample()

        # Движение
        mean     = torch.tanh(self.actor_move(h))
        std      = self.log_std.exp().expand_as(mean)
        mv_dist  = torch.distributions.Normal(mean, std)
        mv_raw   = mv_dist.rsample()

        lp = (t_dist.log_prob(t)
              + x_dist.log_prob(x)
              + y_dist.log_prob(y)
              + mv_dist.log_prob(mv_raw).sum(-1))
        ent = (t_dist.entropy()
               + x_dist.entropy()
               + y_dist.entropy()
               + mv_dist.entropy().sum(-1))

        return dict(
            value      = self.critic(h).squeeze(-1),
            next_state = (h, c),
            type       = t,
            x          = x,
            y          = y,
            move       = torch.tanh(mv_raw),
            move_raw   = mv_raw,
            log_prob   = lp,
            entropy    = ent,
        )

    def evaluate(self, m_b, s_b, hx_b, cx_b, t_b, x_b, y_b, mv_b):
        feat      = torch.cat([self.cnn(m_b), self.mlp(s_b)], -1)
        h, c      = self._lstm(feat, hx_b, cx_b)

        t_dist    = torch.distributions.Categorical(logits=self.actor_type(h))
        te        = self.type_emb(F.one_hot(t_b.cpu(), cfg.n_action_types).float())
        x_dist    = torch.distributions.Categorical(
            logits=self.actor_x(torch.cat([h, te], -1))
        )
        xe        = self.x_emb(F.one_hot(x_b.cpu(), cfg.n_action_x).float())
        y_dist    = torch.distributions.Categorical(
            logits=self.actor_y(torch.cat([h, te, xe], -1))
        )
        mean      = torch.tanh(self.actor_move(h))
        std       = self.log_std.exp().expand_as(mean)
        mv_dist   = torch.distributions.Normal(mean, std)

        lp  = (t_dist.log_prob(t_b)
               + x_dist.log_prob(x_b)
               + y_dist.log_prob(y_b)
               + mv_dist.log_prob(mv_b).sum(-1))
        ent = (t_dist.entropy()
               + x_dist.entropy()
               + y_dist.entropy()
               + mv_dist.entropy().sum(-1))
        return lp, ent, self.critic(h).squeeze(-1)


# ═══════════════════════════════════════════════════════
#  БУФЕР РОЛЛАУТОВ
# ═══════════════════════════════════════════════════════
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

    def add(self, obs, hx, cx, out, reward, done):
        self.maps.append(obs["m"])
        self.scalars.append(obs["s"])
        self.hx.append(hx.squeeze(0).detach())
        self.cx.append(cx.squeeze(0).detach())
        self.types.append(out["type"].squeeze(0).detach())
        self.xs.append(out["x"].squeeze(0).detach())
        self.ys.append(out["y"].squeeze(0).detach())
        self.move_raws.append(out["move_raw"].squeeze(0).detach())
        self.log_probs.append(out["log_prob"].squeeze(0).detach())
        self.values.append(out["value"].squeeze(0).detach())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def compute_gae(self, last_value, last_done):
        T   = len(self.rewards)
        adv = torch.zeros(T)
        ret = torch.zeros(T)
        gae = 0.0
        nv  = last_value
        for t in reversed(range(T)):
            m      = 1.0 - float(self.dones[t])
            delta  = self.rewards[t] + cfg.gamma * nv * m - self.values[t].item()
            gae    = delta + cfg.gamma * cfg.gae_lambda * m * gae
            adv[t] = gae
            ret[t] = gae + self.values[t].item()
            nv     = self.values[t].item()
        return adv, ret

    def get_tensors(self):
        return dict(
            maps      = torch.tensor(np.array(self.maps),    dtype=torch.float32),
            scalars   = torch.tensor(np.array(self.scalars), dtype=torch.float32),
            hx        = torch.stack(self.hx).cpu(),
            cx        = torch.stack(self.cx).cpu(),
            types     = torch.stack(self.types).cpu(),
            xs        = torch.stack(self.xs).cpu(),
            ys        = torch.stack(self.ys).cpu(),
            move_raws = torch.stack(self.move_raws).cpu(),
            log_probs = torch.stack(self.log_probs).cpu(),
            values    = torch.stack(self.values).cpu(),
        )


# ═══════════════════════════════════════════════════════
#  СРЕДА (WebSocket)
# ═══════════════════════════════════════════════════════
class MindustryEnv:
    def __init__(self):
        self._loop      = asyncio.new_event_loop()
        self._q         = queue.Queue()
        self._connected = threading.Event()
        self._ws        = None
        threading.Thread(target=self._run_loop, daemon=True).start()
        if not self._connected.wait(timeout=20.0):
            raise RuntimeError("Не удалось подключиться к Mindustry за 20 с.")

        self._norm           = RunningNorm(clip=cfg.reward_clip)
        self._prev_tile_x    = None
        self._prev_tile_y    = None
        self._prev_core_dist = None
        self._still_steps    = 0

    # ── WebSocket ────────────────────────────────────────────
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        import websockets

        async def _connect():
            while True:
                try:
                    async with websockets.connect(
                        f"ws://{cfg.ws_host}:{cfg.ws_port}",
                        ping_interval=20, ping_timeout=10,
                    ) as ws:
                        self._ws = ws
                        self._connected.set()
                        while True:
                            self._q.put(json.loads(await ws.recv()))
                except Exception as e:
                    print(f"[WS] reconnect... ({e})")
                    self._connected.clear()
                    await asyncio.sleep(2)

        self._loop.run_until_complete(_connect())

    def _send(self, payload):
        asyncio.run_coroutine_threadsafe(
            self._ws.send(json.dumps(payload)), self._loop
        ).result(timeout=5.0)

    def _drain(self):
        while True:
            try: self._q.get_nowait()
            except queue.Empty: return

    # ── Наблюдение ───────────────────────────────────────────
    def _parse_obs(self, resp):
        shape = (cfg.map_channels, cfg.map_size, cfg.map_size)
        m = np.asarray(resp.get("map", np.zeros(shape)), dtype=np.float32)
        if m.shape != shape:
            m = np.zeros(shape, dtype=np.float32)

        s = np.asarray(resp.get("scalars", [0.0]*cfg.scalar_obs_size), dtype=np.float32)
        if s.shape != (cfg.scalar_obs_size,):
            tmp = np.zeros(cfg.scalar_obs_size, dtype=np.float32)
            tmp[:min(cfg.scalar_obs_size, s.size)] = s.ravel()[:cfg.scalar_obs_size]
            s = tmp
        return {"m": m, "s": s}

    # ── Reward ───────────────────────────────────────────────
    def _reward(self, s) -> float:
        """
        Простой и стабильный reward.
        Главное: агент должен ХОТЕТЬ находиться рядом с базой.
        """
        r = cfg.r_step

        # ── 1. Близость к ядру (главный сигнал) ─────────────
        # s[6], s[7] — нормированный вектор ОТ агента К ядру
        # Длина этого вектора приблизительно равна расстоянию
        core_dx   = float(s[6]) if len(s) > 6 else 0.0
        core_dy   = float(s[7]) if len(s) > 7 else 0.0
        core_dist = float(np.clip(np.sqrt(core_dx**2 + core_dy**2), 0.0, 1.0))

        # Постоянный бонус за близость: максимум у базы, ноль вдали
        r += cfg.r_near_base * (1.0 - core_dist)

        # Бонус/штраф за изменение расстояния до базы
        if self._prev_core_dist is not None:
            delta = self._prev_core_dist - core_dist   # >0 = приблизились
            if delta > 0:
                r += cfg.r_approach_base * delta
            else:
                r += cfg.r_leave_base * abs(delta)
        self._prev_core_dist = core_dist

        # ── 2. Штраф за неподвижность ────────────────────────
        tile_x = float(s[1]) * 100.0
        tile_y = float(s[2]) * 100.0
        if self._prev_tile_x is not None:
            moved = (
                abs(tile_x - self._prev_tile_x) > cfg.move_epsilon
                or abs(tile_y - self._prev_tile_y) > cfg.move_epsilon
            )
            if moved:
                self._still_steps = 0
            else:
                self._still_steps += 1
            over = max(0, self._still_steps - cfg.no_move_grace)
            r += cfg.r_no_move * over      # маленький, линейный, стабильный
        self._prev_tile_x = tile_x
        self._prev_tile_y = tile_y

        # ── 3. Медь (вторичный сигнал) ───────────────────────
        r += float(s[4]) * cfg.r_copper_mult

        return r

    # ── Публичный интерфейс ──────────────────────────────────
    def step(self, action):
        env_act = int(cfg.action_type_map[int(action["type"])])
        payload = {
            "vx":     float(action["vx"]),
            "vy":     float(action["vy"]),
            "type":   env_act,
            "x":      int(action["x"]),
            "y":      int(action["y"]),
            "delete": env_act == 5,
        }
        self._send(payload)

        resp   = self._q.get(timeout=10.0)
        obs    = self._parse_obs(resp)
        done   = bool(resp.get("done", False))
        raw_r  = self._reward(obs["s"])

        # Нормализация: делим на running std
        norm_r = self._norm.update_and_normalize(raw_r) if cfg.reward_norm else raw_r
        return obs, norm_r, done, raw_r

    def reset(self):
        self._drain()
        self._send({"reset": True})
        self._prev_tile_x    = None
        self._prev_tile_y    = None
        self._prev_core_dist = None
        self._still_steps    = 0

        deadline = time.time() + 20.0
        last_resp = None
        while time.time() < deadline:
            try:
                resp = self._q.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                break
            last_resp = resp
            if resp.get("ready", False) and not resp.get("done", False):
                obs = self._parse_obs(resp)
                s = obs["s"]
                core_dx = float(s[6]) if len(s) > 6 else 0.0
                core_dy = float(s[7]) if len(s) > 7 else 0.0
                self._prev_core_dist = float(np.clip(
                    np.sqrt(core_dx**2 + core_dy**2), 0.0, 1.0
                ))
                self._prev_tile_x = float(s[1]) * 100.0
                self._prev_tile_y = float(s[2]) * 100.0
                return obs

        raise RuntimeError(
            f"Mindustry не вернула ready после reset. Последний ответ: {last_resp}"
        )


# ═══════════════════════════════════════════════════════
#  PPO ТРЕНЕР
# ═══════════════════════════════════════════════════════
class PPOTrainer:
    def __init__(self, model: ActorCritic):
        self.model = model
        self.opt   = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self.sched = optim.lr_scheduler.LinearLR(
            self.opt, start_factor=1.0, end_factor=0.1, total_iters=1000
        )

    def update(self, buf: RolloutBuffer, last_val: float, last_done: bool):
        adv, ret = buf.compute_gae(last_val, last_done)

        # Нормализация advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Backward на CPU (DirectML не поддерживает scatter)
        self.model.cpu()
        d   = buf.get_tensors()
        T   = len(buf.rewards)

        pl = vl = el = 0.0
        n  = 0

        for _ in range(cfg.ppo_epochs):
            idx = torch.randperm(T)
            for s in range(0, T, cfg.batch_size):
                b = idx[s: s + cfg.batch_size]

                lp_new, ent, val_new = self.model.evaluate(
                    d["maps"][b], d["scalars"][b],
                    d["hx"][b], d["cx"][b],
                    d["types"][b], d["xs"][b], d["ys"][b],
                    d["move_raws"][b],
                )

                ratio  = (lp_new - d["log_probs"][b]).exp()
                ab     = adv[b]
                lp_pol = -torch.min(
                    ratio * ab,
                    ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * ab,
                ).mean()

                vo  = d["values"][b]
                vc  = vo + (val_new - vo).clamp(-cfg.clip_eps, cfg.clip_eps)
                lp_val = torch.max(
                    (val_new - ret[b]) ** 2,
                    (vc      - ret[b]) ** 2,
                ).mean()

                loss = lp_pol + cfg.value_loss_coef * lp_val - cfg.entropy_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

                pl += lp_pol.item(); vl += lp_val.item(); el += ent.mean().item()
                n  += 1

        self.model.to(device)
        self.sched.step()
        return {"pol": pl/n, "val": vl/n, "ent": el/n}


# ═══════════════════════════════════════════════════════
#  ЧЕКПОИНТ
# ═══════════════════════════════════════════════════════
def save_ckpt(model, trainer, norm, episode, steps, updates, best):
    torch.save({
        "model":   model.state_dict(),
        "opt":     trainer.opt.state_dict(),
        "sched":   trainer.sched.state_dict(),
        "norm":    norm.state_dict(),
        "episode": episode,
        "steps":   steps,
        "updates": updates,
        "best":    best,
    }, cfg.checkpoint_path)
    print(f"[Ckpt] saved ep={episode} steps={steps} upd={updates}")


def load_ckpt(model, trainer, norm):
    if not os.path.exists(cfg.checkpoint_path):
        return 0, 0, 0, -float("inf")
    ck = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    trainer.opt.load_state_dict(ck["opt"])
    if "sched" in ck:  trainer.sched.load_state_dict(ck["sched"])
    if "norm"  in ck:  norm.load_state_dict(ck["norm"])
    print(f"[Ckpt] loaded ep={ck['episode']} steps={ck['steps']} upd={ck['updates']}")
    return ck["episode"], ck["steps"], ck["updates"], ck["best"]


# ═══════════════════════════════════════════════════════
#  ГЛАВНЫЙ ЦИКЛ
# ═══════════════════════════════════════════════════════
def train(max_updates=None):
    print("=" * 60)
    print("  Mindustry RL Agent - PPO+LSTM v4")
    print(f"  Device: {device}")
    print(f"  Reward norm: {cfg.reward_norm}  clip={cfg.reward_clip}")
    print("=" * 60)

    env     = MindustryEnv()
    model   = ActorCritic().to(device)
    trainer = PPOTrainer(model)
    norm    = env._norm      # доступ к нормализатору для сохранения

    ep, steps, updates, best = load_ckpt(model, trainer, norm)

    buf        = RolloutBuffer()
    ep_rewards = deque(maxlen=100)
    ep_raw     = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)

    ep_reward  = 0.0
    ep_raw_r   = 0.0
    ep_len     = 0
    obs        = env.reset()
    lstm_state = model.init_state(1)
    target     = cfg.max_updates if max_updates is None else max_updates

    print("\n[Train] Start...\n")

    try:
        while target <= 0 or updates < target:
            model.eval()
            buf.reset()

            for _ in range(cfg.rollout_steps):
                m_t = torch.tensor(obs["m"], dtype=torch.float32, device=device).unsqueeze(0)
                s_t = torch.tensor(obs["s"], dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    out = model.act(m_t, s_t, lstm_state)

                action = dict(
                    type     = out["type"],
                    x        = out["x"],
                    y        = out["y"],
                    vx       = float(out["move"][0, 0].cpu()),
                    vy       = float(out["move"][0, 1].cpu()),
                    move_raw = out["move_raw"],
                )

                next_obs, reward, done, raw_r = env.step(action)
                ep_reward += reward
                ep_raw_r  += raw_r
                ep_len    += 1
                steps     += 1

                buf.add(obs, lstm_state[0], lstm_state[1], out, reward, done)
                lstm_state = out["next_state"]
                obs = next_obs

                if done:
                    ep_rewards.append(ep_reward)
                    ep_raw.append(ep_raw_r)
                    ep_lengths.append(ep_len)
                    ep += 1

                    if ep % cfg.log_interval == 0 and ep_rewards:
                        mr  = np.mean(ep_rewards)
                        mrr = np.mean(ep_raw)
                        ml  = np.mean(ep_lengths)
                        print(
                            f"[Ep {ep:5d} | {steps:8d} steps] "
                            f"norm_r={mr:+7.3f} | raw_r={mrr:+8.1f} | "
                            f"len={ml:5.0f}"
                        )
                        if mr > best:
                            best = mr
                            print(f"  * New best: {best:.3f}")

                    ep_reward  = 0.0
                    ep_raw_r   = 0.0
                    ep_len     = 0
                    obs        = env.reset()
                    lstm_state = model.init_state(1)

            # Bootstrap
            m_t = torch.tensor(obs["m"], dtype=torch.float32, device=device).unsqueeze(0)
            s_t = torch.tensor(obs["s"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                last_out = model.act(m_t, s_t, lstm_state)
            last_val = float(last_out["value"].squeeze().cpu())

            model.train()
            stats   = trainer.update(buf, last_val, False)
            updates += 1

            # Считаем текущий std нормализатора
            cur_std = np.sqrt(max(norm.M2 / max(norm.n - 1, 1), 1e-12))
            print(
                f"[PPO] upd={updates:4d} steps={steps:8d} | "
                f"pol={stats['pol']:+.4f} val={stats['val']:.4f} ent={stats['ent']:.4f} | "
                f"reward_std={cur_std:.3f}"
            )

            if updates % cfg.save_interval == 0:
                save_ckpt(model, trainer, norm, ep, steps, updates, best)

    except KeyboardInterrupt:
        print("\n[Train] Interrupted, saving...")
        save_ckpt(model, trainer, norm, ep, steps, updates, best)
        raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--updates",    type=int, default=cfg.max_updates)
    p.add_argument("--checkpoint", type=str, default=cfg.checkpoint_path)
    p.add_argument("--no-norm",    action="store_true",
                   help="Disable reward normalization")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.checkpoint_path = args.checkpoint
    if args.no_norm:
        cfg.reward_norm = False
    train(max_updates=args.updates)