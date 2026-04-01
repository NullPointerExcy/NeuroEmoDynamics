"""
Fine-tune Llama3-8B-Instruct with LoRA to understand raw SNN state vectors.

The state vector (267 dims) is projected into Llama3's embedding space as
virtual prefix tokens. The model learns to condition its responses on the
emotional state encoded in these vectors.

Dataset: facebook/empathetic_dialogues

Usage:
    cd src
    python -m models.train.train_llama_lora
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.state_projector import (
    StateVectorProjector, emotion_label_to_state_vector, EMOTION_TO_PROFILE,
)

LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
NED_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "neuro_emo_dynamics_v10q1.safetensors"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "llama_lora")
NUM_EPOCHS = 3
BATCH_SIZE = 2          # small -> More VRAM usage
GRAD_ACCUM_STEPS = 8    # effective batch = 16
LR = 2e-4
MAX_SEQ_LEN = 256
NUM_VIRTUAL_TOKENS = 4
WARMUP_RATIO = 0.05

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmpatheticDialoguesDataset(Dataset):
    """Wraps facebook/empathetic_dialogues for state-vector-conditioned training.

    Each sample:
        - state_vector: (267,) synthetic SNN state based on emotion label
        - input_ids: tokenized "[user_message]<sep>[listener_response]"
        - labels: -100 for user tokens, token ids for response tokens
    """

    def __init__(self, split, tokenizer, profile_embeddings, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.profile_embeddings = profile_embeddings.detach().cpu()
        self.max_len = max_len

        raw = load_dataset("facebook/empathetic_dialogues", split=split, trust_remote_code=True)

        self.samples = []
        convs = {}
        for row in raw:
            cid = row["conv_id"]
            if cid not in convs:
                convs[cid] = {"emotion": row["context"], "utterances": []}
            convs[cid]["utterances"].append(row["utterance"])

        for cid, conv in convs.items():
            emotion = conv["emotion"].strip().lower()
            utts = conv["utterances"]
            for i in range(0, len(utts) - 1, 2):
                user_msg = utts[i].strip().replace("_comma_", ",")
                response = utts[i + 1].strip().replace("_comma_", ",")
                if user_msg and response:
                    self.samples.append({
                        "emotion": emotion,
                        "user": user_msg,
                        "response": response,
                    })

        print(f"[{split}] Loaded {len(self.samples)} conversation pairs "
              f"from {len(convs)} conversations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        state_vector = emotion_label_to_state_vector(
            sample["emotion"], self.profile_embeddings, device="cpu"
        )

        # Tokenize: <user_msg>\n<response><eos>
        user_text = sample["user"]
        response_text = sample["response"]

        user_tokens = self.tokenizer.encode(user_text, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        eos = [self.tokenizer.eos_token_id]

        max_text_len = self.max_len - 1  # for EOS

        total = len(user_tokens) + len(response_tokens) + 1
        if total > max_text_len:
            max_user = max_text_len - len(response_tokens) - 1
            if max_user < 10:
                half = max_text_len // 2
                user_tokens = user_tokens[:half]
                response_tokens = response_tokens[:half - 1]
            else:
                user_tokens = user_tokens[:max_user]

        input_ids = user_tokens + response_tokens + eos
        labels = [-100] * len(user_tokens) + response_tokens + eos

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        pad_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "state_vector": state_vector.float(),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class StateConditionedLlama(nn.Module):
    """Wraps a LoRA-adapted Llama model with a state vector projector.

    During forward:
    1. Project state_vector → virtual token embeddings
    2. Get text token embeddings from Llama's embed_tokens
    3. Concatenate [virtual_tokens | text_tokens]
    4. Run through Llama with the combined embeddings
    """

    def __init__(self, llama_model, projector):
        super().__init__()
        self.llama = llama_model
        self.projector = projector

    def forward(self, state_vector, input_ids, attention_mask, labels=None):
        B = input_ids.shape[0]
        device = input_ids.device

        virtual_embeds = self.projector(state_vector.to(device))  # (B, num_vt, hidden)
        num_vt = virtual_embeds.shape[1]
        text_embeds = self.llama.get_input_embeddings()(input_ids)  # (B, seq_len, hidden)

        inputs_embeds = torch.cat([virtual_embeds, text_embeds], dim=1)

        vt_mask = torch.ones(B, num_vt, dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([vt_mask, attention_mask], dim=1)

        if labels is not None:
            vt_labels = torch.full((B, num_vt), -100, dtype=labels.dtype, device=device)
            full_labels = torch.cat([vt_labels, labels], dim=1)
        else:
            full_labels = None

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

        return outputs



def load_ned_profile_embeddings():
    """Load profile embeddings directly from the NED checkpoint file."""
    state_dict = load_file(NED_CHECKPOINT)
    profile_embeddings = state_dict["profile_embedding.weight"].detach().clone()
    print(f"Loaded NED profile embeddings: {profile_embeddings.shape}")
    return profile_embeddings


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Llama3 LoRA Fine-Tuning for SNN State Vector Understanding")
    print("=" * 60)

    print("\n[1/5] Loading NED profile embeddings...")
    profile_embeddings = load_ned_profile_embeddings()

    print(f"\n[2/5] Loading {LLAMA_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    print("\n[3/5] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    hidden_size = model.config.hidden_size  # 4096 for Llama3-8B
    projector = StateVectorProjector(
        hidden_size=hidden_size,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
    ).to(DEVICE).to(torch.bfloat16)

    wrapped_model = StateConditionedLlama(model, projector)

    print("\n[4/5] Loading empathetic_dialogues dataset...")
    train_ds = EmpatheticDialoguesDataset("train", tokenizer, profile_embeddings)
    val_ds = EmpatheticDialoguesDataset("validation", tokenizer, profile_embeddings)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    optimizer_params = [
        {"params": projector.parameters(), "lr": LR},
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": LR},
    ]
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)

    total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[5/5] Training for {NUM_EPOCHS} epochs...")
    print(f"  Samples: {len(train_ds)} train, {len(val_ds)} val")
    print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")
    print()

    model.gradient_checkpointing_enable()
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        wrapped_model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            outputs = wrapped_model(
                state_vector=batch["state_vector"].to(DEVICE),
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(projector.parameters()) + list(model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                avg = total_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f} | LR: {lr_now:.2e}")

        train_avg = total_loss / len(train_loader)

        wrapped_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = wrapped_model(
                    state_vector=batch["state_vector"].to(DEVICE),
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
                val_loss += outputs.loss.item()
        val_avg = val_loss / len(val_loader)

        print(f"\n  Epoch {epoch+1}: Train Loss = {train_avg:.4f}, Val Loss = {val_avg:.4f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            print(f"  New best! Saving checkpoint...")
            model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
            torch.save(projector.state_dict(), os.path.join(OUTPUT_DIR, "projector.pt"))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))

        print()

    config = {
        "llama_model_id": LLAMA_MODEL_ID,
        "num_virtual_tokens": NUM_VIRTUAL_TOKENS,
        "hidden_size": hidden_size,
        "state_dim": StateVectorProjector.STATE_DIM,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_len": MAX_SEQ_LEN,
        "best_val_loss": best_val_loss,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
