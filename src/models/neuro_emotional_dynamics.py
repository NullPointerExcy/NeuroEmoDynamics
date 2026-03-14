import torch
import torch.nn as nn
import torch.nn.functional as F
from qlif_layers.qlif_layer import QLIFLayer
from models.text_encoder import TextEncoder


class NeuroEmoDynamics(nn.Module):
    def __init__(self, vocab, embed_dim=100, text_hidden_dim=128,
                 modulation_dim=1024, num_classes=6, num_profiles=5, batch_size=16):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===== Profile modulation =====
        self.num_profiles = num_profiles
        self.profile_embedding = nn.Embedding(num_profiles, 256)
        self.neuromod_gates = nn.ModuleDict({
            'serotonin':      nn.Linear(256, 512),          # -> PFC gain
            'norepinephrine': nn.Linear(256, 256),          # -> Amygdala gain
            'dopamine':       nn.Linear(256, modulation_dim)  # -> Striatum prob_slope
        })

        # ===== LIF areas =====
        # Prefrontal: deterministic, Serotonin as Gain (external_modulation)
        self.prefrontal = QLIFLayer(
            num_neurons=512, V_th=1.2, tau=30.0, stochastic=False,
            neuromod_transform=lambda x: torch.sigmoid(3 * x),
            neuromod_mode="gain", neuromod_strength=1.0,
            learnable_threshold=True, learnable_eta=True, learnable_tau=True
        )

        # Amygdala: stochastic (we give it NE as gain)
        self.amygdala = QLIFLayer(
            num_neurons=256, noise_std=0.3, use_adaptive_threshold=False, stochastic=True,
            learnable_threshold=True, learnable_eta=True, learnable_tau=True,
        )

        # Hippocampus: adaptive Threshold (without modulation)
        self.hippocampus = QLIFLayer(
            num_neurons=256, V_th=1.1, tau=25.0, stochastic=False, use_adaptive_threshold=True,
            neuromod_mode="off",
            learnable_threshold=True, learnable_eta=True, learnable_tau=True
        )

        # Thalamus: Serotonin-Gain (take S[:256] as Mod-Signal)
        self.thalamus = QLIFLayer(
            num_neurons=256, V_th=1.0, tau=20.0, stochastic=False,
            neuromod_transform=lambda x: torch.sigmoid(3 * x),
            neuromod_mode="gain", neuromod_strength=0.5,
            learnable_threshold=True, learnable_eta=True, learnable_tau=True
        )

        # Striatum: DSP + Dopamine as prob_slope-Modulation
        self.striatum = QLIFLayer(
            num_neurons=1024, base_alpha=4.0, recovery_rate=0.15,
            stochastic=True, allow_dynamic_spike_probability=True,
            neuromod_transform=lambda x: torch.sigmoid(3 * x),
            neuromod_mode="prob_slope", neuromod_strength=1.0,
            learnable_threshold=True, learnable_eta=True, learnable_tau=True
        )

        # ===== Connectivity =====
        self.pfc_amyg = nn.Linear(512, 256)
        self.pfc_hipp = nn.Linear(512, 256)
        self.pfc_thal = nn.Linear(512, 256)

        self.cross_amyg_hipp = nn.Linear(512, 256)
        self.cross_hipp_thal = nn.Linear(512, 256)
        self.cross_thal_amyg = nn.Linear(512, 256)

        self.integrator = nn.Linear(1792, 1024)  # 7 * 256

        self.feedback = nn.Linear(1024, 512)

        # ===== Text pipeline =====
        self.scale_net = nn.Sequential(
            nn.Linear(modulation_dim + 1, 1),
            nn.Sigmoid()
        )
        self.self_ref_classifier = nn.Sequential(
            nn.Linear(modulation_dim, 1),
            nn.Sigmoid()
        )
        self.text_override_layer = nn.Linear(modulation_dim, 1)

        self.text_encoder = TextEncoder(len(vocab), embed_dim, text_hidden_dim, modulation_dim)
        self.text_gate = nn.Linear(modulation_dim, modulation_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(modulation_dim + 1024, modulation_dim),
            nn.ReLU(),
            nn.Linear(modulation_dim, modulation_dim)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2048, 1),  # 1024 + 1024
            nn.Sigmoid()
        )
        self.text_norm = nn.LayerNorm(modulation_dim)
        self.integ_norm = nn.LayerNorm(1024)

        self.final_classifier = nn.Linear(modulation_dim, num_classes)
        self.id_to_word = {i: token for token, i in vocab.items()}

        self.register_buffer("prev_feedback", torch.zeros(batch_size, 512))
        self.to(self.device)

    def forward(self, sensory_input, reward_signal, text_input, profile_ids=None, profile_vec=None):
        device = self.device

        # ===== Neuromodulation profiles =====
        if profile_vec is not None:
            profile_vec = profile_vec.to(device)
        else:
            profile_vec = self.profile_embedding(profile_ids.to(device))
        serotonin      = torch.sigmoid(self.neuromod_gates['serotonin'](profile_vec))        # (B,512)
        norepinephrine = 1.0 + torch.sigmoid(self.neuromod_gates['norepinephrine'](profile_vec))  # (B,256)
        dopamine       = torch.sigmoid(self.neuromod_gates['dopamine'](profile_vec))         # (B,1024)

        # ===== Text =====
        text_encoding = self.text_encoder(text_input)
        aux_logits = self.final_classifier(text_encoding)
        aux_probs = F.softmax(aux_logits, dim=1)
        positive_scores = (aux_probs[:, 1] + aux_probs[:, 5]).unsqueeze(1)

        gate = torch.sigmoid(self.text_gate(text_encoding))
        self_ref_flag = self.get_self_reference_flag(text_input, self.id_to_word).to(device)
        augmented_text_encoding = torch.cat([text_encoding, self_ref_flag], dim=-1)
        sample_scale = self.scale_net(augmented_text_encoding)
        self_ref_score = self.self_ref_classifier(text_encoding)
        final_scale = sample_scale * self_ref_score
        text_mod = text_encoding * (gate + dopamine * final_scale)

        # ===== Ensure shapes: sensory_input -> (T,B,512) =====
        if sensory_input.dim() == 2:
            sensory_seq = sensory_input.unsqueeze(0)  # (1,B,512)
        else:
            sensory_seq = sensory_input  # (T,B,512)
        T, B, _ = sensory_seq.shape

        # Broadcasting Serotonin over time
        sero_T = serotonin.unsqueeze(0).expand(T, -1, -1)  # (T,B,512)

        # ===== PFC LIF with Gain-Modulation (Serotonin) =====
        pfc_spikes, _ = self.prefrontal(sensory_seq, external_modulation=sero_T)
        pfc_last = pfc_spikes[-1].float() * serotonin + self.prev_feedback  # (B,512)

        # ===== Time-Distributed PFC Output to Subsystems =====
        pfc_seq_float = pfc_spikes.float()  # (T,B,512)
        x = pfc_seq_float.reshape(T * B, 512)
        amyg_seq = self.pfc_amyg(x).reshape(T, B, 256)  # (T,B,256)
        hipp_seq = self.pfc_hipp(x).reshape(T, B, 256)
        thal_seq = self.pfc_thal(x).reshape(T, B, 256)

        # Amygdala-Gain (NE) directly at input
        amyg_seq = amyg_seq * norepinephrine.unsqueeze(0)  # (T,B,256)

        # Thalamus optional per Serotonin
        sero_256 = serotonin[:, :256] if serotonin.size(1) >= 256 else serotonin
        sero_256_T = sero_256.unsqueeze(0).expand(T, -1, -1)

        # ===== Subsystem LIFs =====
        amyg_spk, _ = self.amygdala(amyg_seq)                          # (T,B,256)
        hipp_spk, _ = self.hippocampus(hipp_seq)                       # (T,B,256)
        thal_spk, _ = self.thalamus(thal_seq, external_modulation=sero_256_T)  # (T,B,256)

        # Vectors
        amyg_vec = amyg_spk[-1].float()
        hipp_vec = hipp_spk[-1].float()
        thal_vec = thal_spk[-1].float()

        # ===== Cross-Connections =====
        cross_ah = self.cross_amyg_hipp(torch.cat([amyg_vec, hipp_vec], dim=-1))  # (B,256)
        cross_ht = self.cross_hipp_thal(torch.cat([hipp_vec, thal_vec], dim=-1))
        cross_ta = self.cross_thal_amyg(torch.cat([thal_vec, amyg_vec], dim=-1))

        # ===== Integration + Feedback =====
        integ_input = torch.cat([
            amyg_vec, hipp_vec, thal_vec,
            cross_ah, cross_ht, cross_ta,
            self.prev_feedback[:, :256]
        ], dim=-1)  # (B, 7*256) = (B,1792)

        integ_out = self.integrator(integ_input)  # (B,1024)

        new_feedback = self.feedback(integ_out).detach().clamp_(-10.0, 10.0)
        self.prev_feedback = new_feedback

        text_mod = self.text_norm(text_mod)
        integ_out = self.integ_norm(integ_out)

        override_score = torch.sigmoid(self.text_override_layer(text_mod))
        override_boost = (positive_scores * self_ref_score) * 1.5
        override_score = torch.clamp(override_score + override_boost, 0, 1.5)

        w = self.fusion_gate(torch.cat([text_mod, integ_out], dim=-1))
        w = torch.max(w, override_score)
        fused_rep = w * text_mod + (1 - w) * integ_out

        logits = self.final_classifier(fused_rep)

        # ===== Striatum: Reward as Gain, Dopamine as prob_slope-External =====
        T_str = 10
        str_in = integ_out.unsqueeze(0).expand(T_str, -1, -1)  # (T,B,1024)

        # reward_signal: (B,) or (B,1024) -> broadcast to (T,B,1|1024)
        if reward_signal.dim() == 1:
            reward_T = reward_signal.view(1, -1, 1).expand(T_str, -1, 1)
        elif reward_signal.dim() == 2:
            reward_T = reward_signal.unsqueeze(0).expand(T_str, -1, -1)
        else:
            reward_T = reward_signal

        str_in = str_in * reward_T

        dop_T = dopamine.unsqueeze(0).expand(T_str, -1, -1)  # (T,B,1024) -> prob_slope
        spks, volts = self.striatum(str_in, external_modulation=dop_T)

        return spks, volts, logits, aux_logits, serotonin, dopamine, norepinephrine, self_ref_score

    @staticmethod
    def decode_tokens(token_tensor, id_to_word):
        tokens = [id_to_word[int(token)] for token in token_tensor]
        return " ".join(tokens)

    def get_self_reference_flag(self, token_tensor, id_to_word):
        first_person_pronouns = {
            "i", "me", "my", "mine", "myself",
            "we", "us", "our", "ours", "ourselves",
        }
        flags = []
        for sample in token_tensor:
            text = self.decode_tokens(sample, id_to_word)
            tokens = text.lower().split()
            flag = 1.0 if any(token in first_person_pronouns for token in tokens) else 0.0
            flags.append(flag)
        return torch.tensor(flags, dtype=torch.float32).unsqueeze(1).to(self.device)

    @staticmethod
    def compute_depression_score(voltages):
        return voltages.var(dim=0, unbiased=False).mean()
