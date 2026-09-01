from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TransformerTextEncoder:
    def __init__(
        self,
        model_name: str,
        max_length: int = 64,
        batch_size: int = 32,
        device: str = "auto",
        allow_tfidf_fallback: bool = False,
    ):
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)
        self.allow_tfidf_fallback = bool(allow_tfidf_fallback)
        self.mode = "transformer"
        self.vectorizer = None
        self.tokenizer = None
        self.model = None

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception:
            if not self.allow_tfidf_fallback:
                raise
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.mode = "tfidf_fallback"
            self.vectorizer = TfidfVectorizer(max_features=768, ngram_range=(1, 2))

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        if self.mode == "tfidf_fallback":
            matrix = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
            if matrix.shape[1] < 768:
                matrix = np.pad(matrix, ((0, 0), (0, 768 - matrix.shape[1])))
            return matrix[:, :768]

        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tok = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tok = {k: v.to(self.device) for k, v in tok.items()}
                model_out = self.model(**tok)
                pooled = self._mean_pool(
                    model_out.last_hidden_state, tok["attention_mask"]
                )
                pooled = F.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(outputs)


class DenseGraphAttention(nn.Module):
    """
    Lightweight dense GAT layer used on the small per-pair graph:
    candidate + project + skill nodes.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D], adjacency: [B,N,N], non-negative weights
        bsz, n, _ = x.shape
        h = self.proj(x).view(bsz, n, self.heads, self.head_dim)
        src = (h * self.attn_src).sum(-1)  # [B,N,H]
        dst = (h * self.attn_dst).sum(-1)  # [B,N,H]
        logits = src.unsqueeze(2) + dst.unsqueeze(1)  # [B,N,N,H]
        logits = F.leaky_relu(logits, negative_slope=0.2)

        mask = adjacency.unsqueeze(-1) > 0
        logits = logits.masked_fill(~mask, -1e9)
        weight_bias = torch.log(torch.clamp(adjacency.unsqueeze(-1), min=1e-6))
        logits = logits + weight_bias
        alpha = torch.softmax(logits, dim=2)
        alpha = self.dropout(alpha)

        out = torch.einsum("bijh,bjhd->bihd", alpha, h)
        out = out.reshape(bsz, n, self.heads * self.head_dim)
        return self.norm(out + self.proj(x))


class PairGraphTransformer(nn.Module):
    """
    Candidate–project heterogeneous graph encoder.

    Node 0: candidate
    Node 1: project
    Nodes 2..: skill dimensions

    Local graph attention is followed by a Transformer encoder for global
    interaction modeling.
    """

    def __init__(
        self,
        n_skills: int,
        text_dim: int,
        scalar_dim: int,
        hidden_dim: int = 128,
        graph_heads: int = 4,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.2,
        skill_threshold: float = 0.05,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.n_skills = int(n_skills)
        self.hidden_dim = int(hidden_dim)
        self.scalar_dim = int(scalar_dim)
        self.skill_threshold = float(skill_threshold)
        self.use_transformer = bool(use_transformer)

        self.candidate_proj = nn.Linear(n_skills + text_dim, hidden_dim)
        self.project_proj = nn.Linear(n_skills + text_dim, hidden_dim)
        self.skill_embedding = nn.Embedding(n_skills, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)

        self.gat = DenseGraphAttention(
            hidden_dim, hidden_dim, graph_heads, dropout=dropout
        )

        if self.use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=transformer_layers
            )
        else:
            self.transformer = nn.Identity()

        self.pair_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.prediction_head = FinalPredictionHead(
            latent_dim=hidden_dim, scalar_dim=scalar_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def _build_graph(
        self,
        candidate_skill: torch.Tensor,
        project_skill: torch.Tensor,
        candidate_text: torch.Tensor,
        project_text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = candidate_skill.size(0)
        device = candidate_skill.device
        skill_ids = torch.arange(self.n_skills, device=device)
        skill_base = self.skill_embedding(skill_ids).unsqueeze(0).expand(bsz, -1, -1)

        candidate = self.candidate_proj(
            torch.cat([candidate_skill, candidate_text], dim=1)
        ).unsqueeze(1)
        project = self.project_proj(
            torch.cat([project_skill, project_text], dim=1)
        ).unsqueeze(1)

        skill_strength = 0.5 * (
            candidate_skill.unsqueeze(-1) + project_skill.unsqueeze(-1)
        )
        skill_nodes = skill_base * (0.5 + skill_strength)

        nodes = torch.cat([candidate, project, skill_nodes], dim=1)

        type_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.ones(1, dtype=torch.long, device=device),
                torch.full((self.n_skills,), 2, dtype=torch.long, device=device),
            ]
        )
        nodes = nodes + self.type_embedding(type_ids).unsqueeze(0)

        n_nodes = self.n_skills + 2
        adjacency = torch.eye(n_nodes, device=device).unsqueeze(0).repeat(bsz, 1, 1)
        adjacency[:, 0, 1] = 1.0
        adjacency[:, 1, 0] = 1.0

        c_weights = torch.where(
            candidate_skill > self.skill_threshold,
            candidate_skill,
            torch.zeros_like(candidate_skill),
        )
        p_weights = torch.where(
            project_skill > self.skill_threshold,
            project_skill,
            torch.zeros_like(project_skill),
        )
        adjacency[:, 0, 2:] = c_weights
        adjacency[:, 2:, 0] = c_weights
        adjacency[:, 1, 2:] = p_weights
        adjacency[:, 2:, 1] = p_weights
        return nodes, adjacency

    def encode(
        self,
        candidate_skill: torch.Tensor,
        project_skill: torch.Tensor,
        candidate_text: torch.Tensor,
        project_text: torch.Tensor,
    ) -> torch.Tensor:
        nodes, adjacency = self._build_graph(
            candidate_skill, project_skill, candidate_text, project_text
        )
        local = self.gat(nodes, adjacency)
        global_nodes = self.transformer(local)

        cand = global_nodes[:, 0]
        proj = global_nodes[:, 1]
        diff = torch.abs(cand - proj)
        prod = cand * proj
        return self.pair_projection(torch.cat([cand, proj, diff, prod], dim=1))

    def forward(
        self,
        candidate_skill: torch.Tensor,
        project_skill: torch.Tensor,
        candidate_text: torch.Tensor,
        project_text: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(
            candidate_skill, project_skill, candidate_text, project_text
        )
        logits = self.prediction_head(latent, scalar_features)
        return logits, latent


class FinalPredictionHead(nn.Module):
    def __init__(self, latent_dim: int, scalar_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.scalar_dim = scalar_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, latent: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, scalar], dim=1)).squeeze(1)

    def forward_combined(self, combined: torch.Tensor) -> torch.Tensor:
        return self.net(combined).squeeze(1)


@dataclass
class SupervisedTrainingResult:
    best_state: Dict
    best_val_loss: float
    epochs_trained: int


def train_supervised_matcher(
    model: PairGraphTransformer,
    train_tensors: Tuple[torch.Tensor, ...],
    train_labels: torch.Tensor,
    val_tensors: Tuple[torch.Tensor, ...],
    val_labels: torch.Tensor,
    cfg: Dict,
    device: torch.device,
) -> SupervisedTrainingResult:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(*train_tensors, train_labels.float())
    loader = DataLoader(
        ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs_trained = 0

    for epoch in range(int(cfg["supervised_epochs"])):
        model.train()
        for batch in loader:
            *features, y = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*features)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_features = [x.to(device) for x in val_tensors]
            val_logits, _ = model(*val_features)
            val_loss = float(loss_fn(val_logits, val_labels.float().to(device)).item())

        epochs_trained = epoch + 1
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.get("patience", 4)):
                break

    model.load_state_dict(best_state)
    return SupervisedTrainingResult(best_state, best_loss, epochs_trained)


def infer_matcher(
    model: PairGraphTransformer,
    tensors: Tuple[torch.Tensor, ...],
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    ds = TensorDataset(*tensors)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, latents = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]
            logits, latent = model(*batch)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            latents.append(latent.cpu().numpy())
    return np.concatenate(scores).astype(np.float32), np.vstack(latents).astype(np.float32)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        eps: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.eps = float(eps)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.priority = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        i = self.pos
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = reward
        self.next_state[i] = next_state
        self.done[i] = float(done)
        self.priority[i] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, progress: float, rng: np.random.Generator):
        if self.size < batch_size:
            raise ValueError("Not enough replay samples.")
        p = np.power(self.priority[: self.size] + self.eps, self.alpha)
        p = p / p.sum()
        idx = rng.choice(self.size, size=batch_size, replace=False, p=p)
        beta = self.beta_start + (self.beta_end - self.beta_start) * np.clip(progress, 0, 1)
        weights = np.power(self.size * p[idx], -beta)
        weights = weights / weights.max()
        return (
            idx,
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        values = np.abs(td_errors).astype(np.float32) + self.eps
        self.priority[indices] = values
        self.max_priority = max(self.max_priority, float(values.max()))


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DDQNPERAgent:
    def __init__(self, state_dim: int, cfg: Dict, device: torch.device, seed: int):
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(seed)
        hidden = int(cfg["hidden_dim"])
        self.online = QNetwork(state_dim, hidden).to(device)
        self.target = QNetwork(state_dim, hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=float(cfg["learning_rate"])
        )
        self.replay = PrioritizedReplayBuffer(
            int(cfg["replay_buffer_size"]),
            state_dim,
            alpha=float(cfg["priority_alpha"]),
            beta_start=float(cfg["priority_beta_start"]),
            beta_end=float(cfg["priority_beta_end"]),
            eps=float(cfg["priority_epsilon"]),
        )
        self.steps = 0

    def epsilon(self):
        start = float(self.cfg["epsilon_start"])
        end = float(self.cfg["epsilon_end"])
        horizon = max(1, int(self.cfg["epsilon_decay_steps"]))
        frac = min(1.0, self.steps / horizon)
        return start + frac * (end - start)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, 2))
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(dim=1).item())

    def learn(self, total_steps: int):
        batch_size = int(self.cfg["batch_size"])
        if self.replay.size < max(batch_size, int(self.cfg["warmup_steps"])):
            return None
        if self.steps % int(self.cfg["update_every"]) != 0:
            return None

        sample = self.replay.sample(
            batch_size,
            progress=self.steps / max(1, total_steps),
            rng=self.rng,
        )
        idx, state, action, reward, next_state, done, weights = sample

        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.long, device=self.device)
        r = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online network selects action, target network evaluates it.
            next_action = self.online(ns).argmax(dim=1, keepdim=True)
            next_q = self.target(ns).gather(1, next_action).squeeze(1)
            target = r + float(self.cfg["gamma"]) * (1.0 - d) * next_q

        td = target - q
        loss = (w * F.smooth_l1_loss(q, target, reduction="none")).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online.parameters(), float(self.cfg.get("gradient_clip_norm", 5.0))
        )
        self.optimizer.step()

        self.replay.update_priorities(idx, td.detach().cpu().numpy())

        tau = float(self.cfg["soft_tau"])
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target.parameters(), self.online.parameters()
            ):
                target_param.data.mul_(1.0 - tau).add_(tau * online_param.data)

        return float(loss.item())

    def train_in_environment(self, env) -> List[float]:
        total_steps = int(self.cfg["train_steps"])
        losses = []
        state = env.reset()
        for _ in range(total_steps):
            action = self.act(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            self.replay.add(state, action, reward, next_state, done)
            self.steps += 1
            loss = self.learn(total_steps)
            if loss is not None:
                losses.append(loss)
            state = env.reset() if done else next_state
        return losses

    def score_assign_advantage(self, states: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.online.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                x = torch.as_tensor(
                    states[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                q = self.online(x)
                values.append((q[:, 1] - q[:, 0]).cpu().numpy())
        return np.concatenate(values).astype(np.float32)
