"""LSTM trajectory encoder ported from DBB-EPA LunarLander.

Implements the Encoder ABC using an LSTM that compresses a fixed-length
sequence of (state, action) pairs into a compact embedding. Two auxiliary
decoders are trained jointly to enrich the representation:
  - FC decoder: reconstructs the victim's policy π(a|s)  (CrossEntropyLoss)
  - LSTM decoder: reconstructs environment dynamics T(s'|s,a) (MSELoss)

No pre-training is needed; the encoder is trained online during the
attack loop each time encode() is called with new trajectory data.

Generalises to any environment: state_dim and num_actions are derived
automatically via from_population(), so nothing is hardcoded for a
specific env or policy type.

Reference architecture: DBB-EPA/src_lunarlander/ae/autoencoder.py
"""

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from core.encoder import Encoder

# ------------------------------------------------------------------ #
#  Internal trajectory transition                                      #
# ------------------------------------------------------------------ #

_Transition4 = namedtuple('_Transition4', ('pre_state', 'pre_action', 'state', 'action'))


class _TrajectoryMemory:
    """Circular buffer for _Transition4 namedtuples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: List[_Transition4] = []
        self._pos: int = 0

    def push(
        self,
        pre_state: np.ndarray,
        pre_action: float,
        state: np.ndarray,
        action: float,
    ) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # type: ignore[arg-type]
        self.memory[self._pos] = _Transition4(pre_state, pre_action, state, action)
        self._pos = (self._pos + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.memory)

    def get_recent(self, n: int) -> List[_Transition4]:
        """Return the n most recent transitions in temporal order."""
        size = len(self.memory)
        n = min(n, size)
        if size < self.capacity:
            return self.memory[-n:]
        end = self._pos
        start = end - n
        if start >= 0:
            return self.memory[start:end]
        return self.memory[start:] + self.memory[:end]

    def get_ordered(self) -> List[_Transition4]:
        """Return all transitions oldest-first (corrects wrap-around ordering)."""
        if len(self.memory) < self.capacity:
            return list(self.memory)
        return self.memory[self._pos:] + self.memory[:self._pos]

    def clear(self) -> None:
        self.memory.clear()
        self._pos = 0


# ------------------------------------------------------------------ #
#  Neural network modules                                              #
# ------------------------------------------------------------------ #

class _EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, input_size)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        return hn[-1]  # (batch, hidden_size)


class _DecoderFC(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_units: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.ln1 = nn.LayerNorm(hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = F.relu(self.ln1(self.fc1(x)))
        return self.fc2(x)  # raw logits; CrossEntropyLoss handles softmax


class _DecoderLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, input_size)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out)  # (seq_len, batch, output_size)


class _EncoderDecoder(nn.Module):
    """Encoder + dual decoder for joint training."""

    def __init__(
        self,
        encoder: _EncoderLSTM,
        decoder_fc: _DecoderFC,
        decoder_lstm: _DecoderLSTM,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder_fc = decoder_fc
        self.decoder_lstm = decoder_lstm

    def forward(
        self,
        enc_in: torch.Tensor,       # (seq_len, batch, state_dim+1)
        dec_fc_in: torch.Tensor,    # (batch, seq_len, state_dim)
        dec_lstm_in: torch.Tensor,  # (seq_len, batch, state_dim+1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = enc_in.size(0)
        embedding = self.encoder(enc_in)  # (batch, embedding_dim)

        # Broadcast embedding over seq_len for FC decoder
        z_fc = embedding.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, embedding_dim)
        y_fc = self.decoder_fc(torch.cat([dec_fc_in, z_fc], dim=2))  # (batch, seq_len, num_actions)

        # Broadcast embedding over seq_len for LSTM decoder
        z_lstm = embedding.unsqueeze(0).expand(seq_len, -1, -1)  # (seq_len, batch, embedding_dim)
        y_lstm = self.decoder_lstm(torch.cat([dec_lstm_in, z_lstm], dim=2))  # (seq_len, batch, state_dim)

        return y_fc, y_lstm


# ------------------------------------------------------------------ #
#  Public encoder class                                                #
# ------------------------------------------------------------------ #

class LSTMTrajectoryEncoder(Encoder):
    """LSTM-based trajectory encoder for the attack observation.

    Each call to encode() ingests new (pre_s, pre_a, s, a) trajectory tuples
    from victim_data['trajectories'], trains the encoder if enough data has
    accumulated, and returns the LSTM embedding of the victim's most recent
    trajectory window.

    The target memory is populated once (before the attack loop) via
    populate_target_memory().  The encoder is then trained online by
    comparing actual victim trajectories against the target.

    Args:
        state_dim: Dimensionality of a single state observation.
                   Use 1 for Discrete environments (integer → scalar float),
                   or obs_space.shape[0] for Box environments.
        num_actions: Number of discrete actions.
        embedding_dim: LSTM hidden size and output embedding dimension.
        seq_len: Trajectory window length (number of timesteps per chunk).
        memory_size: Capacity of each victim's circular trajectory buffer.
        n_epochs: Training epochs run each time encode() triggers a train step.
        lr: SGD learning rate for the joint encoder-decoder model.
        num_victims: Number of victim agents K.
                     Total output dimension = K * embedding_dim.
        device: Torch device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        embedding_dim: int,
        seq_len: int,
        memory_size: int,
        n_epochs: int,
        lr: float,
        num_victims: int = 1,
        device: Optional[Any] = None,
    ) -> None:
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._embedding_dim = embedding_dim
        self._seq_len = seq_len
        self._n_epochs = n_epochs
        self._num_victims = num_victims
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        enc_in_size = state_dim + 1                       # [state || action]
        dec_fc_in_size = embedding_dim + state_dim         # [embedding || state]
        dec_lstm_in_size = embedding_dim + state_dim + 1   # [embedding || pre_state || pre_action]

        self._enc_lstm = _EncoderLSTM(enc_in_size, embedding_dim).to(self._device)
        self._dec_fc = _DecoderFC(dec_fc_in_size, num_actions).to(self._device)
        self._dec_lstm = _DecoderLSTM(dec_lstm_in_size, embedding_dim, state_dim).to(self._device)
        self._model = _EncoderDecoder(self._enc_lstm, self._dec_fc, self._dec_lstm).to(self._device)

        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._crit_cls = nn.CrossEntropyLoss()
        self._crit_reg = nn.MSELoss()

        self._victim_memories: List[_TrajectoryMemory] = [
            _TrajectoryMemory(memory_size) for _ in range(num_victims)
        ]
        self._target_memory = _TrajectoryMemory(memory_size)

    # ---------------------------------------------------------------- #
    #  Encoder ABC                                                       #
    # ---------------------------------------------------------------- #

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim * self._num_victims

    def encode(
        self,
        victim_data: List[Dict],
        env_dynamics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Ingest trajectories, train encoder, and return current embedding.

        Args:
            victim_data: List of per-victim result dicts. Each dict should
                contain 'trajectories': a list of (pre_state, pre_action,
                state, action) tuples produced by victim.get_trajectories().
                If a victim's dict has no 'trajectories' key the victim's
                memory is not updated this step.
            env_dynamics: Unused; present for interface compatibility.

        Returns:
            1D float32 array of shape (num_victims * embedding_dim,).
        """
        for k in range(self._num_victims):
            data = victim_data[k] if k < len(victim_data) else {}
            for tup in data.get('trajectories', []):
                pre_s, pre_a, s, a = tup
                self._victim_memories[k].push(
                    self._to_state_array(pre_s),
                    float(pre_a),
                    self._to_state_array(s),
                    float(a),
                )

        parts = []
        for k in range(self._num_victims):
            if (len(self._victim_memories[k]) >= self._seq_len
                    and len(self._target_memory) >= self._seq_len):
                emb = self._embed_single(self._victim_memories[k], self._target_memory)
            else:
                emb = np.zeros(self._embedding_dim, dtype=np.float32)
            parts.append(emb)
        return np.concatenate(parts)

    def get_initial_embedding(self) -> np.ndarray:
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def train_encoder(self) -> None:
        """Train on buffered trajectories. Call once per episode, not per timestep."""
        if (len(self._target_memory) >= self._seq_len
                and any(len(m) >= 2 * self._seq_len for m in self._victim_memories)):
            self._train_step()

    # ---------------------------------------------------------------- #
    #  Factory                                                           #
    # ---------------------------------------------------------------- #

    @classmethod
    def from_population(cls, population: Any, config: Any) -> 'LSTMTrajectoryEncoder':
        """Build an encoder with dimensions inferred from the victim population.

        Args:
            population: VictimSystem instance with .env and .num_victims.
            config: YACS config node containing LSTM_ENCODER sub-node with
                    EMBEDDING_DIM, SEQ_LEN, MEMORY_SIZE, N_EPOCHS, LR keys.

        Returns:
            Configured LSTMTrajectoryEncoder instance.
        """
        env = population.env
        obs_space = env.observation_space
        act_space = env.action_space

        if isinstance(obs_space, spaces.Discrete):
            state_dim = 1
        elif isinstance(obs_space, spaces.Box):
            state_dim = int(np.prod(obs_space.shape))
        else:
            state_dim = 1

        if isinstance(act_space, spaces.Discrete):
            num_actions = act_space.n
        else:
            num_actions = int(np.prod(act_space.shape))

        cfg = config.LSTM_ENCODER
        return cls(
            state_dim=state_dim,
            num_actions=num_actions,
            embedding_dim=cfg.EMBEDDING_DIM,
            seq_len=cfg.SEQ_LEN,
            memory_size=cfg.MEMORY_SIZE,
            n_epochs=cfg.N_EPOCHS,
            lr=cfg.LR,
            num_victims=population.num_victims,
        )

    # ---------------------------------------------------------------- #
    #  Target memory                                                     #
    # ---------------------------------------------------------------- #

    def populate_target_memory(
        self,
        env: Any,
        target_policy: Union[np.ndarray, Callable],
        n_steps: int,
    ) -> None:
        """Populate the target trajectory buffer by running a target policy.

        Args:
            env: Environment instance (reset/step interface).
            target_policy: Either a callable policy_fn(raw_obs) -> int, or a
                           (nS, nA) numpy array where action = argmax(row[state]).
                           For continuous-obs environments pass a callable.
            n_steps: Number of transitions to collect (should be >= 2 * seq_len).
        """
        policy_fn = self._make_policy_fn(target_policy)

        self._target_memory.clear()

        obs = env.reset()
        cur_raw = obs[0] if isinstance(obs, tuple) else obs

        prev_state_arr: Optional[np.ndarray] = None
        prev_action: Optional[float] = None

        for _ in range(n_steps):
            cur_state_arr = self._to_state_array(cur_raw)
            action = int(policy_fn(cur_raw))

            result = env.step(action)
            next_raw = result[0]
            terminated = bool(result[2])
            truncated = bool(result[3]) if len(result) > 3 else False
            done = terminated or truncated

            if prev_state_arr is not None:
                self._target_memory.push(
                    prev_state_arr,
                    float(prev_action),  # type: ignore[arg-type]
                    cur_state_arr,
                    float(action),
                )

            prev_state_arr = cur_state_arr
            prev_action = float(action)

            if done:
                obs = env.reset()
                cur_raw = obs[0] if isinstance(obs, tuple) else obs
                prev_state_arr = None
                prev_action = None
            else:
                cur_raw = next_raw

    # ---------------------------------------------------------------- #
    #  Internal helpers                                                  #
    # ---------------------------------------------------------------- #

    def _to_state_array(self, state: Any) -> np.ndarray:
        """Convert any state representation to a float32 array of shape (state_dim,)."""
        if isinstance(state, (int, np.integer)):
            return np.array([float(state)], dtype=np.float32)
        arr = np.asarray(state, dtype=np.float32).flatten()
        if arr.shape[0] == self._state_dim:
            return arr
        if arr.shape[0] > self._state_dim:
            return arr[:self._state_dim]
        return np.pad(arr, (0, self._state_dim - arr.shape[0]))

    @staticmethod
    def _make_policy_fn(
        target_policy: Union[np.ndarray, Callable]
    ) -> Callable:
        """Return a callable policy_fn(raw_obs) -> int from a matrix or callable."""
        if callable(target_policy):
            return target_policy
        policy_matrix = np.asarray(target_policy)

        def _fn(raw_obs: Any) -> int:
            if isinstance(raw_obs, (int, np.integer)):
                idx = int(raw_obs)
            elif isinstance(raw_obs, (float, np.floating)):
                idx = int(raw_obs)
            else:
                arr = np.asarray(raw_obs).flatten()
                idx = int(arr[0]) if arr.ndim > 0 else int(arr)
            return int(np.argmax(policy_matrix[idx]))

        return _fn

    def _build_enc_tensor(self, transitions: List[_Transition4]) -> torch.Tensor:
        """Stack transitions into an encoder input tensor.

        Returns:
            Tensor of shape (seq_len, 1, state_dim + 1).
        """
        states = torch.tensor(
            np.stack([t.state for t in transitions]), dtype=torch.float32, device=self._device
        )  # (seq_len, state_dim)
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.float32, device=self._device
        ).unsqueeze(1)  # (seq_len, 1)
        return torch.cat([states, actions], dim=1).unsqueeze(1)  # (seq_len, 1, state_dim+1)

    def _build_dec_tensors(
        self, transitions: List[_Transition4]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack transitions into decoder input/target tensors.

        Returns:
            dec_fc_in:     (1, seq_len, state_dim)
            dec_fc_target: (1, seq_len)  long
            dec_lstm_in:   (seq_len, 1, state_dim + 1)
            dec_lstm_tgt:  (seq_len, 1, state_dim)
        """
        states = torch.tensor(
            np.stack([t.state for t in transitions]), dtype=torch.float32, device=self._device
        )  # (seq_len, state_dim)
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.float32, device=self._device
        ).unsqueeze(1)  # (seq_len, 1)
        pre_states = torch.tensor(
            np.stack([t.pre_state for t in transitions]), dtype=torch.float32, device=self._device
        )  # (seq_len, state_dim)
        pre_actions = torch.tensor(
            [t.pre_action for t in transitions], dtype=torch.float32, device=self._device
        ).unsqueeze(1)  # (seq_len, 1)

        dec_fc_in = states.unsqueeze(0)                            # (1, seq_len, state_dim)
        dec_fc_tgt = actions.long().view(1, -1)                    # (1, seq_len)
        dec_lstm_in = torch.cat([pre_states, pre_actions], dim=1).unsqueeze(1)  # (seq_len, 1, state_dim+1)
        dec_lstm_tgt = states.unsqueeze(1)                         # (seq_len, 1, state_dim)

        return dec_fc_in, dec_fc_tgt, dec_lstm_in, dec_lstm_tgt

    def _train_step(self) -> None:
        """Train the encoder on all victims that have >= 2 * seq_len transitions."""
        self._model.train()

        target_ordered = self._target_memory.get_ordered()
        B_seq = target_ordered[:self._seq_len]

        for victim_mem in self._victim_memories:
            ordered = victim_mem.get_ordered()
            n_chunks = len(ordered) // self._seq_len
            if n_chunks < 2:
                continue  # need encoder chunk + decoder chunk

            for _ in range(self._n_epochs):
                for n in range(n_chunks - 1):
                    i_start = n * self._seq_len
                    i_end = i_start + self._seq_len

                    # Encoder inputs: actual trajectory chunk A and target chunk B
                    A_enc = ordered[i_start:i_end]
                    B_enc = B_seq
                    A_enc_in = self._build_enc_tensor(A_enc)  # (seq_len, 1, state_dim+1)
                    B_enc_in = self._build_enc_tensor(B_enc)  # (seq_len, 1, state_dim+1)
                    enc_in = torch.cat([A_enc_in, B_enc_in], dim=1)  # (seq_len, 2, state_dim+1)

                    # Decoder inputs: next seq_len chunk for A, same first chunk for B
                    A_dec = ordered[i_end:i_end + self._seq_len]
                    B_dec = B_seq

                    A_fc_in, A_fc_tgt, A_lstm_in, A_lstm_tgt = self._build_dec_tensors(A_dec)
                    B_fc_in, B_fc_tgt, B_lstm_in, B_lstm_tgt = self._build_dec_tensors(B_dec)

                    # Batch dimension = 2 (A and B)
                    dec_fc_in = torch.cat([A_fc_in, B_fc_in], dim=0)        # (2, seq_len, state_dim)
                    dec_lstm_in = torch.cat([A_lstm_in, B_lstm_in], dim=1)  # (seq_len, 2, state_dim+1)
                    dec_fc_tgt = torch.cat([A_fc_tgt, B_fc_tgt], dim=0).view(-1)       # (2*seq_len,)
                    dec_lstm_tgt = torch.cat([A_lstm_tgt, B_lstm_tgt], dim=1).view(-1) # (2*seq_len*state_dim,)

                    self._optimizer.zero_grad()
                    out_fc, out_lstm = self._model(enc_in, dec_fc_in, dec_lstm_in)
                    # out_fc:   (2, seq_len, num_actions)
                    # out_lstm: (seq_len, 2, state_dim)

                    out_fc = torch.cat([out_fc[0], out_fc[1]], dim=0)             # (2*seq_len, num_actions)
                    out_lstm = torch.cat([out_lstm[:, 0, :], out_lstm[:, 1, :]], dim=0).view(-1)  # (2*seq_len*state_dim,)

                    loss = self._crit_cls(out_fc, dec_fc_tgt) + self._crit_reg(out_lstm, dec_lstm_tgt)
                    loss.backward()
                    self._optimizer.step()

    def _embed_single(
        self,
        victim_memory: _TrajectoryMemory,
        target_memory: _TrajectoryMemory,
    ) -> np.ndarray:
        """Encode the last seq_len transitions from victim_memory.

        Returns:
            Float32 array of shape (embedding_dim,).
        """
        A = victim_memory.get_recent(self._seq_len)
        B = target_memory.get_recent(self._seq_len)

        A_in = self._build_enc_tensor(A)              # (seq_len, 1, state_dim+1)
        B_in = self._build_enc_tensor(B)              # (seq_len, 1, state_dim+1)
        combined = torch.cat([A_in, B_in], dim=1)     # (seq_len, 2, state_dim+1)

        self._enc_lstm.eval()
        with torch.no_grad():
            z = self._enc_lstm(combined)  # (2, embedding_dim)
        return z[0].cpu().numpy().astype(np.float32)  # victim embedding only
