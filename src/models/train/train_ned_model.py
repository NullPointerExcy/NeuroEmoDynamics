import torch
import torch.nn.functional as F
from collections import Counter
from datasets import load_dataset
from torch import optim
from torch.utils.data import DataLoader
from data.emotion_dataset import EmotionDataset
from data.synthetic_data import generate_synthetic_data
from models.neuro_emotional_dynamics import NeuroEmoDynamics
from utils.helper_functions import build_vocab
from sklearn.metrics import roc_auc_score
from safetensors.torch import save_file


# Expected neurotransmitter levels per profile
EXPECTED_NT_LEVELS = torch.tensor([
    [0.3, 0.4, 0.6],  # depressed
    [0.6, 0.5, 0.8],  # anxious
    [0.8, 0.7, 0.5],  # healthy
    [0.5, 0.6, 0.7],  # impulsive
    [0.7, 0.5, 0.6]   # resilient
])

# Emotion bias per profile
EMOTION_BIAS = torch.tensor([
    [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],  # depressed
    [0.1, 0.3, 0.1, 0.4, 0.0, 0.1],       # anxious
    [0.1, 0.3, 0.3, 0.1, 0.1, 0.1],       # healthy
    [0.0, 0.2, 0.1, 0.1, 0.6, 0.0],       # impulsive
    [0.1, 0.2, 0.1, 0.1, 0.1, 0.4]        # resilient
])


def hybrid_loss(logits, targets, neurotransmitters, self_ref_scores,
                class_weights=None, ls=0.05, alpha=0.8, beta=0.1, gamma=0.1,
                foc_gamma=2.0, foc_lambda=0.1,
                profile_ids=None, profile_weights=None):
    """Hybrid loss supporting both discrete profile_ids and continuous profile_weights.

    Args:
        profile_ids: (B,) discrete profile indices - used when profile_weights is None
        profile_weights: (B, 5) soft profile mixture weights - used for interpolated profiles
    """
    device = logits.device

    # Cross-entropy with class weights & label smoothing
    ce = F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=ls)

    # Neuromodulatory consistency
    serotonin, dopamine, norepinephrine = neurotransmitters

    if profile_weights is not None:
        # Continuous: profile_weights @ expected_levels
        profile_factor = profile_weights.to(device)
    else:
        # Discrete: one-hot encoding
        profile_factor = F.one_hot(profile_ids, num_classes=5).float()

    expected_levels = EXPECTED_NT_LEVELS.to(device)
    neuromod_loss = F.mse_loss(
        torch.stack([serotonin.mean(1), dopamine.mean(1), norepinephrine.mean(1)], dim=1),
        profile_factor @ expected_levels
    )

    # Coherence
    eps = 1e-8
    prob = F.softmax(logits, dim=1) + eps
    emotion_bias = EMOTION_BIAS.to(device)

    if profile_weights is not None:
        # Continuous: weighted combination of emotion biases
        target_bias = profile_weights.to(device) @ emotion_bias
    else:
        target_bias = emotion_bias[profile_ids]

    kl = F.kl_div(prob.log(), target_bias, reduction='batchmean')
    coherence_loss = (kl * (1 - self_ref_scores.squeeze())).mean()

    pt = prob.gather(1, targets.view(-1, 1)).squeeze(1)
    focal = ((1.0 - pt) ** foc_gamma).mean()

    return alpha * ce + beta * neuromod_loss + gamma * coherence_loss + foc_lambda * focal


def generate_profile_signals(profile: str, batch_size: int, device: str):
    profile_map = {
        'depressed': 0,
        'anxious': 1,
        'healthy': 2,
        'impulsive': 3,
        'resilient': 4
    }
    return torch.full((batch_size,), profile_map[profile], device=device)


def get_profile_by_bias(batch_labels, bias_prob: float, device):
    emotion_to_preferred_profile = {
        0: 'depressed',  # sadness
        1: 'healthy',    # joy
        2: 'healthy',    # love
        3: 'impulsive',  # anger
        4: 'anxious',    # fear
        5: 'healthy'     # surprise (or resilient)
    }

    profile_to_idx = {
        'depressed': 0,
        'anxious': 1,
        'healthy': 2,
        'impulsive': 3,
        'resilient': 4
    }

    profile_ids = []
    for l in batch_labels:
        if torch.rand(1).item() < bias_prob:
            preferred = emotion_to_preferred_profile[l.item()]
            profile_ids.append(profile_to_idx[preferred])
        else:
            profile_ids.append(torch.randint(0, 5, (1,)).item())
    return torch.tensor(profile_ids, device=device)


def interpolate_profiles(model, profile_ids, interp_prob=0.3, device="cpu"):
    """With probability interp_prob, create interpolated profile vectors.

    This teaches the model to handle continuous emotional states (not just
    the 5 discrete profiles), which is essential for the ESM at inference time.

    Returns:
        profile_vec: (B, 256) continuous profile vectors
        profile_weights: (B, 5) soft mixture weights (for loss computation)
        used_interpolation: bool
    """
    B = profile_ids.size(0)

    if torch.rand(1).item() > interp_prob:
        # Use discrete embeddings (normal training)
        return None, None, False

    with torch.no_grad():
        embeddings = model.profile_embedding.weight  # (5, 256)

    # Create soft mixture weights centered on the assigned profile
    # Start with one-hot, then add noise and normalize
    weights = F.one_hot(profile_ids, num_classes=5).float()  # (B, 5)

    # Add Dirichlet-like noise: keeps primary profile dominant but adds blending
    noise = torch.rand(B, 5, device=device) * 0.4  # up to 0.4 weight on other profiles
    weights = weights * 0.7 + noise * 0.3  # 70% primary, 30% noise blend
    weights = weights / weights.sum(dim=1, keepdim=True)  # normalize to sum=1

    # Compute interpolated profile vectors
    profile_vec = weights @ embeddings  # (B, 5) @ (5, 256) = (B, 256)

    return profile_vec.to(device), weights.to(device), True


def train_model(num_epochs=10, batch_size=16, timesteps=50, lr=1e-3,
                lambda_aux=0.5, bias_prob=0.8, interp_prob=0.3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load dataset (using Hugging Face's datasets)
    dataset = load_dataset("dair-ai/emotion", "split")
    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    # Class weights
    counts = Counter(labels)
    total = sum(counts.values())
    inv_freq = [total / counts[i] for i in range(6)]
    class_weights = torch.tensor(inv_freq, device=device, dtype=torch.float)

    # Map emotions to psychological profiles
    idx_to_profile = {
        0: 'depressed',
        1: 'anxious',
        2: 'healthy',
        3: 'impulsive',
        4: 'resilient'
    }

    # Prepare datasets and vocabulary
    vocab = build_vocab(texts, min_freq=2, max_size=30000)
    train_dataset = EmotionDataset(split="train", vocab=vocab, max_len=32)
    val_dataset = EmotionDataset(split="validation", vocab=vocab, max_len=32)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = NeuroEmoDynamics(
        vocab,
        num_classes=6,
        num_profiles=5,
        batch_size=batch_size
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    clip_max_norm = 1.0

    for epoch in range(num_epochs):
        model.train()
        _total_loss = 0.0
        _interp_batches = 0

        for batch in train_dataloader:
            text_in = batch["text"].to(device)
            batch_labels = batch["label"].to(device)

            profile_ids = get_profile_by_bias(batch_labels, bias_prob, device)

            # Profile interpolation: with interp_prob chance, use blended profiles
            profile_vec, profile_weights, used_interp = interpolate_profiles(
                model, profile_ids, interp_prob=interp_prob, device=device
            )
            if used_interp:
                _interp_batches += 1

            sensory_inputs = []
            reward_signals = []
            for pid in profile_ids:
                profile = idx_to_profile[int(pid)]
                si, rs = generate_synthetic_data(
                    profile=profile,
                    timesteps=timesteps,
                    batch_size=1,
                    input_size=512,
                    reward_size=1024,
                    device=device
                )
                sensory_inputs.append(si)
                reward_signals.append(rs)
            sensory_input = torch.cat(sensory_inputs, dim=1)
            reward_signal = torch.cat(reward_signals, dim=0)

            optimizer.zero_grad(set_to_none=True)

            # ===== Forward + Loss autocast =====
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                spks, volts, logits, aux_logits, serotonin, dopamine, norepinephrine, self_ref_score = model(
                    sensory_input, reward_signal, text_in, profile_ids,
                    profile_vec=profile_vec
                )

                primary_loss = hybrid_loss(
                    logits=logits,
                    targets=batch_labels,
                    neurotransmitters=(serotonin, dopamine, norepinephrine),
                    self_ref_scores=self_ref_score,
                    class_weights=class_weights, ls=0.05,
                    profile_ids=profile_ids,
                    profile_weights=profile_weights,
                )
                aux_loss = F.cross_entropy(aux_logits, batch_labels, label_smoothing=0.05)
                total_loss = primary_loss + lambda_aux * aux_loss

            # ===== Backward (scaled) =====
            scaler.scale(total_loss).backward()

            # ===== Unscale + Clip + Step =====
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

            scaler.step(optimizer)
            scaler.update()

            _total_loss += total_loss.item()

        scheduler.step()
        avg_train_loss = _total_loss / len(train_dataloader)
        interp_pct = _interp_batches / len(train_dataloader) * 100

        # ================ Validation ================
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_dataloader:
                text_in = batch["text"].to(device)
                batch_labels = batch["label"].to(device)

                profile_ids = get_profile_by_bias(batch_labels, bias_prob, device)

                sensory_inputs = []
                reward_signals = []
                for pid in profile_ids:
                    profile = idx_to_profile[int(pid)]
                    si, rs = generate_synthetic_data(
                        profile=profile,
                        timesteps=timesteps,
                        batch_size=1,
                        input_size=512,
                        reward_size=1024,
                        device=device
                    )
                    sensory_inputs.append(si)
                    reward_signals.append(rs)
                sensory_input = torch.cat(sensory_inputs, dim=1)
                reward_signal = torch.cat(reward_signals, dim=0)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    spks, volts, logits, aux_logits, serotonin, dopamine, norepinephrine, self_ref_score = model(
                        sensory_input, reward_signal, text_in, profile_ids
                    )

                    primary_loss = hybrid_loss(
                        logits=logits,
                        targets=batch_labels,
                        neurotransmitters=(serotonin, dopamine, norepinephrine),
                        self_ref_scores=self_ref_score,
                        class_weights=class_weights, ls=0.05,
                        profile_ids=profile_ids,
                    )
                    aux_loss = F.cross_entropy(aux_logits, batch_labels, label_smoothing=0.05)
                    loss = primary_loss + lambda_aux * aux_loss

                val_losses.append(loss.item())

                probs = F.softmax(logits, dim=1)
                all_preds.append(probs.cpu())
                all_targets.append(batch_labels.cpu())

        avg_val_loss = sum(val_losses) / len(val_losses)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        try:
            val_auc = roc_auc_score(all_targets, all_preds, multi_class="ovr")
        except Exception as e:
            val_auc = float('nan')
            print(f"Error computing AUC: {e}")

        print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} "
              f"| Val AUC {val_auc:.4f} | Interp {interp_pct:.0f}%")

        if (epoch + 1) % 10 == 0:
            print(f"Saving model at epoch [{epoch + 1}]...")
            save_file(model.state_dict(), f"../../checkpoints/neuro_emo_dynamics_v10_{epoch + 1}.safetensors")
            with open(f"../../checkpoints/neuro_emo_dynamics_v10_{epoch + 1}.json", "w") as f:
                import json
                json.dump({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_auc": val_auc
                }, f)
            print("Model saved!")

    save_file(model.state_dict(), "../../checkpoints/neuro_emo_dynamics_v10q1.safetensors")


if __name__ == "__main__":
    # 4 Epochs, 16 timesteps, lambda_aux=0.5 (Increase epoch if needed, but overfitting may occur)
    # If you want to train for more epochs, consider using a lower learning rate or add dropout.
    # interp_prob=0.3 means 30% of batches use interpolated profile vectors
    train_model(num_epochs=4, lambda_aux=0.5, timesteps=16, interp_prob=0.3)
