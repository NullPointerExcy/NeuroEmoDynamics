import json
import math
import time
from collections import deque

import torch
import torch.nn.functional as F

# Habituation constants
HABITUATION_SIMILARITY_THRESHOLD = 0.85  # cosine sim above this = "same message"
HABITUATION_DECAY_SECONDS = 86400        # 24 hours per count decay
HABITUATION_RATE = 0.3                   # factor drops by this per repetition
HABITUATION_FLOOR = -0.3                 # minimum factor (reversal cap)


PROFILE_NAMES = {
    0: "depressed",
    1: "anxious",
    2: "healthy",
    3: "impulsive",
    4: "resilient",
}
PROFILE_IDS = {v: k for k, v in PROFILE_NAMES.items()}

# Positive emotion indices in the 6-class emotion set (sadness=0, joy=1, love=2, anger=3, fear=4, surprise=5)
POSITIVE_EMOTIONS = {1, 2, 5}  # joy, love, surprise
NEGATIVE_EMOTIONS = {0, 3, 4}  # sadness, anger, fear


class EmotionalStateMemory:
    def __init__(self, model, initial_profile="depressed", momentum=0.85, device="cpu"):
        self.device = device
        self.momentum = momentum
        self.num_profiles = model.num_profiles

        # Extract all profile embeddings from model (detached copy)
        with torch.no_grad():
            self.profile_embeddings = model.profile_embedding.weight.detach().clone().to(device)

        # Initialize state from the chosen profile
        initial_id = PROFILE_IDS[initial_profile]
        self.emotional_state = self.profile_embeddings[initial_id].clone()
        self.initial_profile = initial_profile
        self.interaction_history = deque(maxlen=200)

        # Long-term message memory for habituation (persists across sessions)
        # key(str) → {embedding: list[float], count: int, last_seen: float}
        self.message_memory = {}
        self._next_memory_id = 0

        # Persistent facts extracted by LLM
        self.user_facts = []   # facts about the human user
        self.self_facts = []   # facts about NED itself (own name, preferences, statements)

    def compute_habituation(self, text_embedding):
        """Check if this message is similar to a previously seen one.

        Uses cosine similarity against stored embeddings to detect repetition.
        Applies time-based decay: count drops by 1 per 24h since last seen.

        Returns:
            factor (float): 1.0 = fresh, 0.0 = no effect, <0 = reversal
            effective_count (int): how many times this message has been seen
        """
        now = time.time()
        emb = text_embedding.detach().cpu()
        if emb.dim() > 1:
            emb = emb.mean(dim=0)
        emb = emb.unsqueeze(0)  # (1, dim)

        best_sim = -1.0
        best_key = None

        # Find most similar stored embedding
        for key, mem in self.message_memory.items():
            stored_emb = torch.tensor(mem["embedding"], dtype=torch.float32).unsqueeze(0)
            sim = F.cosine_similarity(emb, stored_emb, dim=1).item()
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim >= HABITUATION_SIMILARITY_THRESHOLD and best_key is not None:
            mem = self.message_memory[best_key]

            # Time decay: reduce count by 1 per 24h since last seen
            elapsed = now - mem["last_seen"]
            decay = int(elapsed / HABITUATION_DECAY_SECONDS)
            decayed_count = max(1, mem["count"] - decay)

            # Increment for this new occurrence
            mem["count"] = decayed_count + 1
            mem["last_seen"] = now
            # Update embedding with running average for better fuzzy matching
            old = torch.tensor(mem["embedding"], dtype=torch.float32)
            mem["embedding"] = (0.9 * old + 0.1 * emb.squeeze(0)).tolist()
        else:
            # New message — store it
            best_key = str(self._next_memory_id)
            self._next_memory_id += 1
            self.message_memory[best_key] = {
                "embedding": emb.squeeze(0).tolist(),
                "count": 1,
                "last_seen": now,
            }

        effective_count = self.message_memory[best_key]["count"]
        factor = max(1.0 - (effective_count - 1) * HABITUATION_RATE, HABITUATION_FLOOR)

        return factor, effective_count

    def compute_emotional_impact(self, text_logits, neurotransmitters, self_ref_score,
                                 text_embedding=None):
        """Compute a shift vector based on the text's own emotion (aux_logits).

        Uses the text encoder's direct classification (before fusion with
        neural dynamics) so that valence reflects what the text *says*,
        not how the current mood filters it.

        Positive emotions push toward 'healthy' embedding,
        negative emotions push toward 'depressed' embedding.
        A negativity bias (2.5x) makes it harder to recover than to relapse.
        Habituation reduces impact of repeated messages, eventually reversing them.
        """
        serotonin, dopamine, norepinephrine = neurotransmitters

        # Average across batch — text_logits are pre-fusion, so they
        # reflect the actual emotional content of the input text.
        probs = F.softmax(text_logits.detach().mean(dim=0), dim=0)  # (6,)
        self_ref = self_ref_score.detach().mean().item()

        # Compute positive vs negative emotion balance
        pos_score = sum(probs[i].item() for i in POSITIVE_EMOTIONS)
        neg_score = sum(probs[i].item() for i in NEGATIVE_EMOTIONS)
        valence = pos_score - neg_score  # Range: [-1, 1]

        # Habituation: repeated messages lose impact, then reverse
        habituation_factor = 1.0
        habituation_count = 0
        if text_embedding is not None:
            habituation_factor, habituation_count = self.compute_habituation(text_embedding)
            if habituation_factor < 0:
                # Reversal: positive compliment becomes annoying, insult becomes numb
                valence = -valence
            # Scale down by absolute factor
            # (factor 1.0 = full, 0.5 = half, 0.0 = none, -0.3 = reversed at 30%)

        # Determine target profile based on (possibly reversed) valence
        # Positive valence → move toward healthy, negative → move toward depressed
        healthy_emb = self.profile_embeddings[PROFILE_IDS["healthy"]]
        depressed_emb = self.profile_embeddings[PROFILE_IDS["depressed"]]
        resilient_emb = self.profile_embeddings[PROFILE_IDS["resilient"]]

        if valence > 0:
            # Blend between healthy and resilient for positive shift
            target = (0.7 * healthy_emb + 0.3 * resilient_emb)
        else:
            target = depressed_emb

        # Scale impact by absolute valence and self-reference
        # Self-referencing statements have 2x impact
        impact_strength = abs(valence) * (1.0 + self_ref)

        # Apply habituation scaling
        impact_strength *= abs(habituation_factor)

        # Negativity bias: negative inputs hit 2.5x harder.
        # It's easy to fall into depression, hard to climb out.
        if valence < 0:
            impact_strength *= 2.5

        # Compute direction from current state toward target
        shift_vector = self.emotional_state + impact_strength * (target - self.emotional_state)

        return shift_vector, valence, self_ref, habituation_factor, habituation_count

    def update(self, shift_vector, text="", valence=0.0):
        """Apply exponential moving average update to emotional state."""
        self.emotional_state = (
            self.momentum * self.emotional_state
            + (1 - self.momentum) * shift_vector
        )

        self.interaction_history.append({
            "text": text,
            "valence": valence,
            "timestamp": time.time(),
        })

    def get_profile_vec(self, batch_size=1):
        """Return current emotional state expanded to batch size."""
        return self.emotional_state.unsqueeze(0).expand(batch_size, -1)

    def get_mood_summary(self):
        """Compute cosine similarity to each profile embedding."""
        state = self.emotional_state.unsqueeze(0)  # (1, 256)
        similarities = F.cosine_similarity(state, self.profile_embeddings, dim=1)  # (5,)
        # Normalize to [0, 1] range
        sims = similarities.cpu().detach()
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        return {PROFILE_NAMES[i]: sims[i].item() for i in range(self.num_profiles)}

    def get_dominant_mood(self):
        """Return the profile name most similar to current state."""
        summary = self.get_mood_summary()
        return max(summary, key=summary.get)

    def get_valence_trend(self, last_n=10):
        """Return average valence of last N interactions."""
        recent = list(self.interaction_history)[-last_n:]
        if not recent:
            return 0.0
        return sum(h["valence"] for h in recent) / len(recent)

    def reset(self, profile="depressed"):
        """Reset mood to a base profile. Message memory is preserved."""
        pid = PROFILE_IDS[profile]
        self.emotional_state = self.profile_embeddings[pid].clone()
        self.initial_profile = profile
        self.interaction_history.clear()

    def clear_memory(self):
        """Wipe all long-term message memory (habituation resets)."""
        self.message_memory.clear()
        self._next_memory_id = 0

    def save(self, path, conversation_history=None):
        """Persist state, history, message memory, conversation, and user facts to JSON."""
        data = {
            "emotional_state": self.emotional_state.cpu().tolist(),
            "initial_profile": self.initial_profile,
            "momentum": self.momentum,
            "history": [
                {"text": h["text"], "valence": h["valence"], "timestamp": h["timestamp"]}
                for h in self.interaction_history
            ],
            "message_memory": self.message_memory,
            "next_memory_id": self._next_memory_id,
            "user_facts": self.user_facts,
            "self_facts": self.self_facts,
        }
        if conversation_history is not None:
            data["conversation_history"] = conversation_history
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Load state, history, message memory, and conversation from JSON.

        Returns:
            list or None: conversation_history if present in the file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.emotional_state = torch.tensor(data["emotional_state"], dtype=torch.float32).to(self.device)
        self.initial_profile = data.get("initial_profile", "depressed")
        self.momentum = data.get("momentum", self.momentum)
        self.interaction_history = deque(maxlen=200)
        for h in data.get("history", []):
            self.interaction_history.append(h)
        self.message_memory = data.get("message_memory", {})
        self._next_memory_id = data.get("next_memory_id", len(self.message_memory))
        self.user_facts = data.get("user_facts", [])
        self.self_facts = data.get("self_facts", [])
        return data.get("conversation_history")
