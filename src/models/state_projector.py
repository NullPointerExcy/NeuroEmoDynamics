import torch
import torch.nn as nn
import torch.nn.functional as F

# Mapping from empathetic_dialogues emotion labels -> SNN profile IDs
# Profile IDs: 0=depressed, 1=anxious, 2=healthy, 3=impulsive, 4=resilient
EMOTION_TO_PROFILE = {
    # Depressed cluster
    "sad": 0, "lonely": 0, "devastated": 0, "guilty": 0,
    "ashamed": 0, "disappointed": 0, "nostalgic": 0, "sentimental": 0,
    # Anxious cluster
    "anxious": 1, "nervous": 1, "terrified": 1, "apprehensive": 1,
    "afraid": 1, "embarrassed": 1, "worried": 1,
    # Healthy cluster
    "joyful": 2, "excited": 2, "grateful": 2, "content": 2,
    "hopeful": 2, "caring": 2, "faithful": 2, "prepared": 2,
    "anticipating": 2, "impressed": 2,
    # Impulsive cluster
    "angry": 3, "furious": 3, "annoyed": 3, "disgusted": 3,
    "jealous": 3, "surprised": 3,
    # Resilient cluster
    "confident": 4, "proud": 4, "trusting": 4, "commanding": 4,
}

# Default valence per profile
PROFILE_VALENCE = {0: -0.7, 1: -0.5, 2: 0.7, 3: -0.6, 4: 0.5}

# Expected NT levels per profile: [serotonin, dopamine, norepinephrine]
PROFILE_NT_LEVELS = {
    0: [0.3, 0.4, 0.6],
    1: [0.6, 0.5, 0.8],
    2: [0.8, 0.7, 0.5],
    3: [0.5, 0.6, 0.7],
    4: [0.7, 0.5, 0.6],
}

# Emotion probabilities per profile (sadness, joy, love, anger, fear, surprise)
PROFILE_EMOTION_PROBS = {
    0: [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
    1: [0.1, 0.3, 0.1, 0.4, 0.0, 0.1],
    2: [0.1, 0.3, 0.3, 0.1, 0.1, 0.1],
    3: [0.0, 0.2, 0.1, 0.1, 0.6, 0.0],
    4: [0.1, 0.2, 0.1, 0.1, 0.1, 0.4],
}


class StateVectorProjector(nn.Module):
    """Projects SNN state vectors into Llama3's embedding space as virtual tokens.

    Input vector layout (267 dims):
        - emotional_state: 256 dims (continuous profile blend)
        - serotonin_mean: 1 dim
        - dopamine_mean: 1 dim
        - norepinephrine_mean: 1 dim
        - emotion_probs: 6 dims (softmax over 6 emotion classes)
        - valence: 1 dim (-1 to +1)
        - habituation_factor: 1 dim (-0.3 to 1.0)

    Output: (num_virtual_tokens, hidden_size) embeddings for Llama3.
    """

    STATE_DIM = 267

    def __init__(self, hidden_size=4096, num_virtual_tokens=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_virtual_tokens = num_virtual_tokens

        self.projector = nn.Sequential(
            nn.Linear(self.STATE_DIM, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_virtual_tokens * hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, state_vector):
        """
        Args:
            state_vector: (batch_size, 267) or (267,)

        Returns:
            (batch_size, num_virtual_tokens, hidden_size)
        """
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)

        state_vector = state_vector.to(dtype=self.projector[0].weight.dtype)

        projected = self.projector(state_vector)
        projected = projected.view(-1, self.num_virtual_tokens, self.hidden_size)
        return self.norm(projected)


def build_state_vector(emotional_state, serotonin, dopamine, norepinephrine,
                       emotion_probs, valence, habituation_factor):
    """Assemble individual SNN outputs into a single state vector.

    Args:
        emotional_state: (256,) or (B, 256) continuous profile embedding
        serotonin: (B, 512) or scalar mean
        dopamine: (B, 1024) or scalar mean
        norepinephrine: (B, 256) or scalar mean
        emotion_probs: (6,) or (B, 6) softmax probabilities
        valence: float or (B,) tensor
        habituation_factor: float or (B,) tensor

    Returns:
        (B, 267) state vector ready for StateVectorProjector
    """
    device = emotional_state.device

    if emotional_state.dim() == 1:
        emotional_state = emotional_state.unsqueeze(0)
    B = emotional_state.shape[0]

    # Reduce NT vectors to scalar means
    if isinstance(serotonin, torch.Tensor) and serotonin.dim() > 0:
        sero_mean = serotonin.mean(dim=-1, keepdim=True) if serotonin.dim() == 2 else serotonin.unsqueeze(0)
    else:
        sero_mean = torch.tensor([[serotonin]], device=device).expand(B, 1)

    if isinstance(dopamine, torch.Tensor) and dopamine.dim() > 0:
        dopa_mean = dopamine.mean(dim=-1, keepdim=True) if dopamine.dim() == 2 else dopamine.unsqueeze(0)
    else:
        dopa_mean = torch.tensor([[dopamine]], device=device).expand(B, 1)

    if isinstance(norepinephrine, torch.Tensor) and norepinephrine.dim() > 0:
        ne_mean = norepinephrine.mean(dim=-1, keepdim=True) if norepinephrine.dim() == 2 else norepinephrine.unsqueeze(0)
    else:
        ne_mean = torch.tensor([[norepinephrine]], device=device).expand(B, 1)

    # Emotion probs
    if isinstance(emotion_probs, torch.Tensor):
        if emotion_probs.dim() == 1:
            emotion_probs = emotion_probs.unsqueeze(0).expand(B, -1)
    else:
        emotion_probs = torch.tensor([emotion_probs], device=device).expand(B, -1)

    if isinstance(valence, (int, float)):
        valence_t = torch.tensor([[valence]], device=device).expand(B, 1)
    elif valence.dim() == 0:
        valence_t = valence.view(1, 1).expand(B, 1)
    else:
        valence_t = valence.view(B, 1)

    if isinstance(habituation_factor, (int, float)):
        hab_t = torch.tensor([[habituation_factor]], device=device).expand(B, 1)
    elif habituation_factor.dim() == 0:
        hab_t = habituation_factor.view(1, 1).expand(B, 1)
    else:
        hab_t = habituation_factor.view(B, 1)

    return torch.cat([
        emotional_state,    # (B, 256)
        sero_mean,          # (B, 1)
        dopa_mean,          # (B, 1)
        ne_mean,            # (B, 1)
        emotion_probs,      # (B, 6)
        valence_t,          # (B, 1)
        hab_t,              # (B, 1)
    ], dim=1)               # (B, 267)


def emotion_label_to_state_vector(emotion_label, profile_embeddings, device="cpu",
                                  noise_std=0.05):
    """Convert an empathetic_dialogues emotion label to a synthetic SNN state vector.

    Used during training to create (state_vector, conversation) pairs from the dataset.

    Args:
        emotion_label: str, e.g. "sad", "anxious", "joyful"
        profile_embeddings: (5, 256) tensor from NED model
        device: torch device
        noise_std: noise to add for diversity

    Returns:
        (267,) state vector
    """
    profile_id = EMOTION_TO_PROFILE.get(emotion_label.lower().strip(), 2)  # default: healthy
    valence = PROFILE_VALENCE[profile_id]
    nt_levels = PROFILE_NT_LEVELS[profile_id]
    emotion_probs = PROFILE_EMOTION_PROBS[profile_id]

    emotional_state = profile_embeddings[profile_id].clone().to(device)
    emotional_state += torch.randn_like(emotional_state) * noise_std
    
    valence += (torch.randn(1).item() * 0.1)
    valence = max(-1.0, min(1.0, valence))
    habituation_factor = 0.8 + torch.randn(1).item() * 0.2
    habituation_factor = max(-0.3, min(1.0, habituation_factor))

    nt_noisy = [max(0, min(1, v + torch.randn(1).item() * 0.05)) for v in nt_levels]
    ep_noisy = [max(0, v + torch.randn(1).item() * 0.02) for v in emotion_probs]
    ep_sum = sum(ep_noisy)
    ep_noisy = [v / ep_sum for v in ep_noisy]

    return torch.cat([
        emotional_state,
        torch.tensor(nt_noisy, device=device),
        torch.tensor(ep_noisy, device=device),
        torch.tensor([valence, habituation_factor], device=device),
    ])  # (267,)
