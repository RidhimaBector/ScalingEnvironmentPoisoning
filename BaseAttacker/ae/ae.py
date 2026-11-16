
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

    """ Training the Encoder-Decoder network """
    def Train(self, epoch, n_epoch, n_batch, batch_size):

        self.Model.train()
        for i_epoch in range(n_epoch):

            total_loss = 0.0 #Assuming always 1 epoch of training at a time
            for i_batch in range(n_batch):

                n_target_states = 16 #6
                states = np.tile(np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), axis=0),(batch_size,1))
                states = np.expand_dims(states, axis=2)
                actions = np.random.choice([0,1,2,3,4], size=(batch_size,n_target_states,1), replace=True)
                batch = np.concatenate((states, actions),2)

                states = torch.FloatTensor(states).to(device) #torch.FloatTensor([states]).view(-1,1).to(device)
                actions = torch.FloatTensor(actions).to(device) #torch.FloatTensor([actions]).view(-1,1).to(device)
                batch = torch.FloatTensor(batch).to(device)
                target = actions.long().reshape(batch_size*n_target_states,1).squeeze(1) #actions.long().view(1,6).squeeze(0)

                """ train """
                # clear the gradients of all optimized variables
                self.Model.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.Model(batch, n_target_states, states)
                # calculate the loss
                loss = self.Model.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.Model.optimizer.step()
                # update running training loss
                #loss_list.append([epoch, i_batch, loss.item()]) #*state.size(0)
                total_loss += loss.item()
                print("Epoch: " + str(epoch) + ", Batch: " + str(i_batch) + ", AE Loss = " + str(loss.item()))

        return total_loss/n_batch
#             print('Epoch: {} \tTraining Loss: {:.6f}'.format(i_epoch+1, train_loss))


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
    a pre-trained AutoEncoder. The context vector is whatever the attack environment
    passes at encode time: the running sum of the attacker's own perturbations
    (default), the environment's dynamics vector (legacy), or nothing.

    Nothing here is specific to one environment: the number of states, the number of
    actions, the embedding size and the hidden width are read from the checkpoint.

    Args:
        n_states: Number of victim states (length of the behaviour trace).
        context_dim: Length of the context vector appended to the embedding (0 = none).
        model_path: Path to pre-trained AE weights.
        num_victims: Number of victim agents K.
        context_bounds: (low, high) to scale the context to [-1, 1], or None.
        initial_context: Context for the first observation of an episode; if None the
            first observation is all zeros.
    """

    EMBEDDING_SIZE = 5

    def __init__(self, n_states: int, context_dim: int, model_path: str, num_victims: int = 1,
                 context_bounds=None, initial_context: Optional[np.ndarray] = None):
        self._n_states = n_states
        self._context_dim = context_dim
        self._num_victims = num_victims
        self._bounds = context_bounds
        self._initial_context = None if initial_context is None else np.asarray(initial_context).flatten()

        sd = torch.load(model_path, map_location=device, weights_only=True)
        emb = sd['encoder.fc3.weight'].shape[0]
        hidden = sd['encoder.fc1.weight'].shape[0]
        n_classes = sd['decoder.fc2.weight'].shape[0]   # victim actions + 1 ("not visited yet")
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
        self._ae.load(model_path)

    @classmethod
    def from_population(cls, population, model_path: Optional[str] = None,
                        context_dim: Optional[int] = None, context_bounds=None,
                        initial_context: Optional[np.ndarray] = None) -> 'AEObservationEncoder':
        """Build with dimensions inferred from the victim population."""
        env = population.env
        if model_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(here, "models", default_ae_name(env) + "_AutoEncoder")
            if not os.path.exists(model_path) and default_ae_name(env) == "real_traces":
                model_path = os.path.join(here, "models", "340240_f-o_AutoEncoder_SftMx")
        return cls(
            n_states=int(env.nS),
            context_dim=env.dynamics_dim if context_dim is None else context_dim,
            model_path=model_path,
            num_victims=population.num_victims,
            context_bounds=context_bounds,
            initial_context=initial_context,
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
            trace = trace.reshape(1, -1)  # (1, nS*2)
            ae_emb = self._ae.Policy_Embedding(trace)[-1]
            parts.append(np.concatenate([ae_emb, ctx]))

        return np.concatenate(parts)

    def get_initial_embedding(self) -> np.ndarray:
        if self._initial_context is not None:
            # All states unvisited + the starting context
            return self.encode([{} for _ in range(self._num_victims)], self._initial_context)
        return np.zeros(self.embedding_dim)


if __name__ == "__main__":

    """ Intialize AutoEncoder """
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)

    ae_enc_in_size = 32 # 12 as 6 target states #SEQ_LEN*2 = 5*2
    ae_enc_out_size = 5 # EMBEDDING_SIZE = 5
    ae_dec_in_size = 6 #1+EMBEDDING_SIZE
    ae_dec_out_size = 5 #action_dim 0,1,2,3,4 - 4 for target states that victim has not visited yet

    ae_args = {
        "enc_in_size": ae_enc_in_size,
        "enc_out_size": ae_enc_out_size,
        "dec_in_size": ae_dec_in_size,
        "dec_out_size": ae_dec_out_size,
        "lr": 0.001, #0.001,
    }

    ae = AutoEncoder(**ae_args)
    print("AE Initialized")

    epoch = 1
    n_epoch = 1
    n_batch = 500
    batch_size = 256
    ae_loss_list = []
    while epoch > 0:
        ae_loss_list.append([epoch, ae.Train(epoch, n_epoch, n_batch, batch_size)])
        ae.save(f"./{epoch}")
        if(epoch % 20 == 0):
            np.savetxt("AE_loss_buffer.csv", np.array(ae_loss_list), delimiter=",")

        epoch = epoch + 1

    """states = np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), axis=1)
    actions = np.random.choice([0,1,2,3,4], size=(16,1), replace=True)
    victim_transitions = np.concatenate((states, actions),1)
    ae.Policy_Embedding(victim_transitions)"""
