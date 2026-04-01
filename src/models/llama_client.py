"""Direct Llama3 client with LoRA adapter and SNN state vector projection.

Replaces OllamaClient — no Ollama needed. Llama3 reads the raw SNN state
vector through learned virtual prefix tokens instead of a text system prompt.
"""

import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from models.state_projector import StateVectorProjector


class LlamaClient:
    """Local Llama3 client that understands raw SNN state vectors."""

    def __init__(self, checkpoint_dir, device="cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        self._loaded = False

        # Load config
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as f:
            self.config = json.load(f)

        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.tokenizer = None
        self.projector = None

    def load(self):
        if self._loaded:
            return

        print("Loading Llama3 + LoRA adapter...")
        cfg = self.config

        tok_path = os.path.join(self.checkpoint_dir, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["llama_model_id"],
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        adapter_path = os.path.join(self.checkpoint_dir, "lora_adapter")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        self.projector = StateVectorProjector(
            hidden_size=cfg["hidden_size"],
            num_virtual_tokens=cfg["num_virtual_tokens"],
        ).to(self.device).to(self.torch_dtype)

        proj_path = os.path.join(self.checkpoint_dir, "projector.pt")
        self.projector.load_state_dict(torch.load(proj_path, map_location=self.device))
        self.projector.eval()

        self._loaded = True
        print("Llama3 + LoRA loaded successfully.")

    def is_available(self):
        try:
            config_path = os.path.join(self.checkpoint_dir, "config.json")
            return os.path.exists(config_path)
        except Exception:
            return False

    def chat(self, user_message, state_vector, history=None,
             persona="a friend", user_facts=None, self_facts=None,
             max_new_tokens=200, temperature=0.7, top_p=0.9):
        """Generate a response conditioned on the SNN state vector.

        Args:
            user_message: str, the user's latest message
            state_vector: (267,) tensor from build_state_vector()
            history: list of {"role": ..., "content": ...} dicts
            persona: str, who the assistant should be
            user_facts: list[str], known facts about the user
            self_facts: list[str], known facts about the assistant
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold

        Returns:
            str: the assistant's response text
        """
        self.load()
        context_parts = [f"You are {persona}."]
        if self_facts:
            context_parts.append("About yourself: " + "; ".join(self_facts))
        if user_facts:
            context_parts.append("About the user: " + "; ".join(user_facts))
        context_parts.append("Respond concisely (2-4 sentences). "
                             "Respond in the same language the user writes in.")
        context_text = " ".join(context_parts)

        messages = []
        if history:
            for msg in history[-10:]:
                messages.append(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}")
        messages.append(f"Context: {context_text}")
        messages.append(f"User: {user_message}")
        messages.append("Assistant:")

        prompt = "\n".join(messages)

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_seq_len", 256) - self.config["num_virtual_tokens"],
        ).to(self.device)

        with torch.no_grad():
            sv = state_vector.unsqueeze(0).to(self.device).to(self.torch_dtype)
            virtual_embeds = self.projector(sv)  # (1, num_vt, hidden)

            text_embeds = self.model.get_input_embeddings()(tokens["input_ids"])
            inputs_embeds = torch.cat([virtual_embeds, text_embeds], dim=1)
            vt_mask = torch.ones(1, virtual_embeds.shape[1],
                                 dtype=tokens["attention_mask"].dtype,
                                 device=self.device)
            attention_mask = torch.cat([vt_mask, tokens["attention_mask"]], dim=1)

            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith("Assistant:"):
            response = response[len("Assistant:"):].strip()

        return response

    def extract_facts(self, conversation_history, existing_user_facts=None,
                      existing_self_facts=None):
        """Extract facts about user and assistant from conversation history.

        Uses a simple prompt-based approach (same as OllamaClient but local).
        """
        self.load()

        if not conversation_history:
            return existing_user_facts or [], existing_self_facts or []

        recent = conversation_history[-20:]
        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )

        existing_section = ""
        if existing_user_facts or existing_self_facts:
            parts = []
            if existing_user_facts:
                parts.append("Previously known User facts:\n"
                             + "\n".join(f"- {f}" for f in existing_user_facts))
            if existing_self_facts:
                parts.append("Previously known Assistant facts:\n"
                             + "\n".join(f"- {f}" for f in existing_self_facts))
            existing_section = "\n\n".join(parts) + "\n\n"

        prompt = f"""{existing_section}Conversation:
                    {transcript}
                    Extract facts into JSON with two arrays: "user_facts" and "self_facts".
                    Return ONLY the JSON object."""

        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **tokens,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(outputs[0][tokens["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

        try:
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    user_f = [str(f) for f in result.get("user_facts", []) if f]
                    self_f = [str(f) for f in result.get("self_facts", []) if f]
                    return (user_f or existing_user_facts or [],
                            self_f or existing_self_facts or [])
        except Exception:
            pass

        return existing_user_facts or [], existing_self_facts or []
