import torch
import torch.nn as nn
import numpy as np
class EnvironmentEncoder(EnvironmentEncoderInterface):
    @abstractmethod
    def encode_state(self, environment_state) -> np.ndarray:
        """Encode a given environment state into embedding."""
        pass

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def _build_model(self, input_dim, embed_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    @torch.no_grad()
    def encode_state(self, environment_state):
        state_vector = self._flatten_environment_state(environment_state)
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
        embedding = self.model(state_tensor).numpy().flatten()
        return embedding

    def get_initial_state(self, environment_state=None):
        return np.zeros(self.embedding_dimension)

    def _flatten_environment_state(self, environment_state):
        """Convert environment state to a flat numeric representation."""
        if isinstance(environment_state, np.ndarray):
            return environment_state.flatten()
        elif isinstance(environment_state, list):
            return np.array(environment_state).flatten()
        else:
            raise ValueError(f'Unsupported environment state type: {type(environment_state)}')
