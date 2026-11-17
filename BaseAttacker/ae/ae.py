"""Behaviour-trace autoencoder and the observation encoder built on it.

A behaviour trace is, for every state, the last action the victim was seen taking
(or "not visited yet"). The autoencoder compresses it to a few numbers; train one
for a new environment with `python -m ae.train_ae`.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" Define the Policy_Embedding Network"""
class Policy_Representation(nn.Module):
    def __init__(self, input_size, embedding_size, fc1_units=36, fc2_units=36):
        super(Policy_Representation, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, embedding_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
#         print(f"x_1 = {x}")
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


""" Define the Policy_Imitation Network"""
class Policy_Imitation(nn.Module):
    def __init__(self, input_size, action_size, fc1_units=36, fc2_units=36):
        super(Policy_Imitation, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=1)
        return x


""" Combine Policy_Representation with Policy_Imitation """
class Encoder_Decoder(nn.Module):
    def __init__(self, embedding, imitation, lr):
        super(Encoder_Decoder, self).__init__()
        self.encoder = embedding
        self.decoder = imitation
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x, n_target_states, states):
        #x = torch.cat((s, a),1)
        z = self.encoder(x)

        z = torch.unsqueeze(z, 1)
        x = z.repeat(1,n_target_states,1) #torch.cat(len(s)*[z])
        x = torch.cat((states, x), 2)

        x = self.decoder(x)
        return x


class AutoEncoder():
    def __init__(self, enc_in_size, enc_out_size, dec_in_size, dec_out_size, lr, hidden=36):

        self.lr = lr
        self.Encoder = Policy_Representation(enc_in_size, enc_out_size, hidden, hidden).to(device)
        self.Decoder = Policy_Imitation(dec_in_size, dec_out_size, hidden, hidden).to(device)
        self.Model = Encoder_Decoder(self.Encoder, self.Decoder, self.lr).to(device)

    """ Embedding using Encoder network """
    def Policy_Embedding(self, victim_transitions: np.ndarray) -> np.ndarray:
        self.Model.eval() # prep model for *evaluation*
        batch = torch.FloatTensor(victim_transitions).to(device)
        z = self.Model.encoder(batch).cpu().data.numpy()
        return z

    def save(self, filename):
        torch.save(self.Model.state_dict(), filename + "_AutoEncoder")

    def load(self, filename):
        load_model = self.Model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))
        return load_model


def batch_tensors(actions: np.ndarray):
    """(N, nS) per-state actions -> (encoder input, decoder states, flat targets)."""
    n, nS = actions.shape
    states = np.tile(np.arange(nS, dtype=np.float32)[None, :, None], (n, 1, 1))
    acts = actions.astype(np.float32)[:, :, None]
    x = torch.as_tensor(np.concatenate([states, acts], axis=2), device=device)
    return x, torch.as_tensor(states, device=device), torch.as_tensor(actions, device=device).reshape(-1)


def recon_accuracy(ae: "AutoEncoder", actions: np.ndarray) -> float:
    """Share of (trace, state) pairs whose action the autoencoder reconstructs."""
    ae.Model.eval()
    with torch.no_grad():
        x, states, tgt = batch_tensors(actions)
        pred = ae.Model(x, actions.shape[1], states).argmax(dim=1)
    return float((pred == tgt).float().mean())


def fit_autoencoder(ae: "AutoEncoder", train: np.ndarray, epochs: int = 60,
                    batch_size: int = 256, lr: float = 1e-3, log=None) -> None:
    """Train the autoencoder to reconstruct (N, nS) per-state actions."""
    nS = train.shape[1]
    opt = torch.optim.Adam(ae.Model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        ae.Model.train()
        order = np.random.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), batch_size):
            b = train[order[i:i + batch_size]]
            x, states, tgt = batch_tensors(b)
            probs = ae.Model(x, nS, states)  # decoder already applies softmax
            loss = F.nll_loss(torch.log(probs + 1e-8), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if log is not None and (epoch % 5 == 0 or epoch == 1):
            log(epoch, total / len(train))
    ae.Model.eval()


def default_sizes(n_states: int):
    """(embedding size, hidden width) used for a trace autoencoder of this many states."""
    small = n_states <= 16
    return (5 if small else n_states // 3), (36 if small else 2 * n_states)


def default_ae_name(env) -> str:
    """File stem (under ae/models/) of the trace AE for this environment."""
    shape = getattr(env, 'shape', None)
    if type(env).__name__ == 'Grid3D' and shape is not None:
        rows, cols = shape
        return "real_traces" if (rows, cols) == (4, 4) else f"real_traces_{rows}x{cols}"
    return f"real_traces_{type(env).__name__.lower()}_{int(env.nS)}s{int(env.nA)}a"


def _scale(x, bounds):
    """Map x from [low, high] to [-1, 1]; unchanged if bounds is None."""
    if bounds is None:
        return x
    low, high = bounds
    return (np.asarray(x, dtype=np.float64) - (low + high) / 2.0) / ((high - low) / 2.0)


class AEObservationEncoder(Encoder):
    """Observation = AE(behaviour trace) + a context vector, per victim.

    The behaviour trace (last action the victim took in each state) is compressed by
    an AutoEncoder: pre-trained (model_path = a file) or fitted during the run on the
    traces the attacker itself observes (model_path = "online"; see fit()). The
    context vector is whatever the attack environment
    passes at encode time: the running sum of the attacker's own perturbations
    (default), the environment's dynamics vector (legacy), or nothing.

    Nothing here is specific to one environment: the number of states, the number of
    actions, the embedding size and the hidden width are read from the checkpoint.

    Args:
        n_states: Number of victim states (length of the behaviour trace).
        context_dim: Length of the context vector appended to the embedding (0 = none).
        model_path: Path to pre-trained AE weights, or "online".
        n_actions: Number of victim actions (needed for "online" only).
        online_episodes: "online": attack episodes of exploration to collect traces
            from before the autoencoder is fitted and frozen.
        online_epochs: "online": training epochs of that fit.
        num_victims: Number of victim agents K.
        context_bounds: (low, high) to scale the context to [-1, 1], or None.
        initial_context: Context for the first observation of an episode; if None the
            first observation is all zeros.
    """

    EMBEDDING_SIZE = 5

    ONLINE = "online"

    def __init__(self, n_states: int, context_dim: int, model_path: str, num_victims: int = 1,
                 context_bounds=None, initial_context: Optional[np.ndarray] = None,
                 n_actions: Optional[int] = None, online_episodes: int = 500,
                 online_epochs: int = 60):
        self._n_states = n_states
        self._online = model_path == self.ONLINE
        self._frozen = not self._online
        self._online_episodes, self._online_epochs = online_episodes, online_epochs
        self._traces: List[np.ndarray] = []  # observed traces, until the fit
        self._context_dim = context_dim
        self._num_victims = num_victims
        self._bounds = context_bounds
        self._initial_context = None if initial_context is None else np.asarray(initial_context).flatten()

        if self._online:
            if n_actions is None:
                raise ValueError('model_path="online" needs n_actions')
            emb, hidden = default_sizes(n_states)
            n_classes = n_actions + 1   # victim actions + 1 ("not visited yet")
        else:
            sd = torch.load(model_path, map_location=device, weights_only=True)
            emb = sd['encoder.fc3.weight'].shape[0]
            hidden = sd['encoder.fc1.weight'].shape[0]
            n_classes = sd['decoder.fc2.weight'].shape[0]
            if sd['encoder.fc1.weight'].shape[1] != n_states * 2:
                raise ValueError(
                    f"AE at {model_path} was trained for {sd['encoder.fc1.weight'].shape[1] // 2} states, "
                    f"but this environment has {n_states}. Train one with: python -m ae.train_ae")
        self.EMBEDDING_SIZE = emb
        self._unvisited = n_classes - 1
        self._ae = AutoEncoder(
            enc_in_size=n_states * 2,  # behavior_trace is (nS, 2), flattened = nS*2
            enc_out_size=emb,
            dec_in_size=emb + 1,
            dec_out_size=n_classes,
            lr=0.001,
            hidden=hidden,
        )
        if not self._online:
            self._ae.load(model_path)
        self._ae.Model.eval()

    # --- online fitting (model_path="online") ---

    @property
    def warmup_episodes(self) -> int:
        """Exploration-only attack episodes needed before fit() (0 once fitted)."""
        return 0 if self._frozen else self._online_episodes

    def fit(self) -> float:
        """Fit the autoencoder on the traces observed so far and freeze it.
        Returns the reconstruction accuracy on a held-out tenth of them."""
        data = np.asarray(self._traces, dtype=np.int64)
        perm = np.random.permutation(len(data))
        n_val = max(len(data) // 10, 1)
        val, train = data[perm[:n_val]], data[perm[n_val:]]
        fit_autoencoder(self._ae, train, epochs=self._online_epochs)
        self._frozen, self._traces = True, []
        return recon_accuracy(self._ae, val)

    def save(self, path: str) -> None:
        if self._online:
            torch.save(self._ae.Model.state_dict(), path)

    def load(self, path: str) -> None:
        if self._online:
            self._ae.Model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            self._ae.Model.eval()
            self._frozen, self._traces = True, []

    @classmethod
    def from_population(cls, population, model_path: Optional[str] = None,
                        context_dim: Optional[int] = None, context_bounds=None,
                        initial_context: Optional[np.ndarray] = None,
                        online_episodes: int = 500, online_epochs: int = 60) -> 'AEObservationEncoder':
        """Build with dimensions inferred from the victim population."""
        env = population.env
        if model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(here, "models", default_ae_name(env) + "_AutoEncoder")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No behaviour autoencoder for this environment at {model_path}. "
                    "Train one first with `python -m ae.train_ae --env <name> ...` "
                    "(see README, 'Adding your own environment'), or pass ae_model_path.")
        return cls(
            n_states=int(env.nS),
            context_dim=env.dynamics_dim if context_dim is None else context_dim,
            model_path=model_path,
            num_victims=population.num_victims,
            context_bounds=context_bounds,
            initial_context=initial_context,
            n_actions=int(env.nA),
            online_episodes=online_episodes,
            online_epochs=online_epochs,
        )

    @property
    def embedding_dim(self) -> int:
        return (self.EMBEDDING_SIZE + self._context_dim) * self._num_victims

    def encode(self, victim_data: List[Dict], env_dynamics: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode victim data into the attacker's observation.

        Args:
            victim_data: List of per-victim result dicts (behavior_trace required).
            env_dynamics: The context vector (name kept from the Encoder interface).
                Zeros if None.
        """
        ctx = (
            np.asarray(env_dynamics, dtype=np.float64).flatten()[:self._context_dim]
            if env_dynamics is not None
            else np.zeros(self._context_dim)
        )
        if env_dynamics is not None:
            ctx = _scale(ctx, self._bounds)

        parts = []
        for i in range(self._num_victims):
            data = victim_data[i] if i < len(victim_data) else {}
            trace = np.asarray(
                data.get('behavior_trace', np.column_stack(
                    [np.arange(self._n_states), -np.ones(self._n_states)]))
            ).astype(np.float64).copy()
            # Unvisited states are -1 in the trace; the AE uses the last class for them.
            trace[trace[:, 1] < 0, 1] = self._unvisited
            if not self._frozen and 'behavior_trace' in data:
                self._traces.append(trace[:, 1].astype(np.int64))
            trace = trace.reshape(1, -1)  # (1, nS*2)
            ae_emb = self._ae.Policy_Embedding(trace)[-1]
            parts.append(np.concatenate([ae_emb, ctx]))

        return np.concatenate(parts)

    def get_initial_embedding(self) -> np.ndarray:
        if self._initial_context is not None:
            # All states unvisited + the starting context
            return self.encode([{} for _ in range(self._num_victims)], self._initial_context)
        return np.zeros(self.embedding_dim)
