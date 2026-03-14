import os
import sys
import time

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLineEdit, QComboBox, QMessageBox, QTextEdit, QGroupBox,
    QProgressBar, QSlider, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from ui.style import set_dark_mode

import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from models.neuro_emotional_dynamics import NeuroEmoDynamics
from models.emotional_state_memory import EmotionalStateMemory, PROFILE_NAMES, PROFILE_IDS
from datasets import load_dataset
from data.synthetic_data import generate_synthetic_data
from utils.helper_functions import build_vocab
from utils.perspective_transform import transform_perspective, detect_second_person
from models.ollama_client import OllamaClient
from safetensors.torch import load_file

# Auto-save path (relative to this file → src/checkpoints/states/)
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOSAVE_PATH = os.path.join(_UI_DIR, "..", "checkpoints", "states", "autosave.json")


def tokenize(text):
    return text.lower().split()


def encode_text(text, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]


def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) < max_len:
        return seq + [pad_value] * (max_len - len(seq))
    else:
        return seq[:max_len]


vocab = None

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
timesteps = 10
input_size = 512
reward_size = 1024
max_text_len = 32
num_of_classes = 6

label_to_emotion = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}

profile_to_idx = {
    'depressed': 0,
    'anxious': 1,
    'healthy': 2,
    'impulsive': 3,
    'resilient': 4
}

MOOD_COLORS = {
    "depressed": "#8B0000",
    "anxious": "#FF8C00",
    "healthy": "#228B22",
    "impulsive": "#FF4500",
    "resilient": "#4169E1",
}


def load_vocab():
    global vocab
    try:
        ds = load_dataset("dair-ai/emotion", split="train")
        texts = ds["text"]
        vocab = build_vocab(texts, min_freq=2, max_size=30000)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset and build vocab: {e}")


# ================= Ollama Worker (background thread) =================
class OllamaWorker(QThread):
    """Run Ollama chat in a background thread so the UI doesn't freeze."""
    finished = pyqtSignal(str)   # response text
    error = pyqtSignal(str)      # error message

    def __init__(self, client, user_message, system_prompt, history):
        super().__init__()
        self.client = client
        self.user_message = user_message
        self.system_prompt = system_prompt
        self.history = history

    def run(self):
        try:
            response = self.client.chat(self.user_message, self.system_prompt, self.history)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))


class FactExtractorWorker(QThread):
    """Extract user + self facts from conversation in the background."""
    finished = pyqtSignal(list, list)  # (user_facts, self_facts)

    def __init__(self, client, conversation_history, existing_user_facts, existing_self_facts):
        super().__init__()
        self.client = client
        self.conversation_history = conversation_history
        self.existing_user_facts = existing_user_facts
        self.existing_self_facts = existing_self_facts

    def run(self):
        try:
            user_facts, self_facts = self.client.extract_facts(
                self.conversation_history,
                self.existing_user_facts,
                self.existing_self_facts,
            )
            self.finished.emit(user_facts, self_facts)
        except Exception:
            self.finished.emit(self.existing_user_facts or [], self.existing_self_facts or [])


# ================= Main Window =================
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Neuro Emotional Dynamics UI")
        self.resize(1000, 750)

        self.model = None
        self.esm = None
        self.figure_canvas = None
        self.ollama_client = None
        self.conversation_history = []  # [{"role": "user"/"assistant", "content": "..."}]
        self._ollama_worker = None  # keep reference to prevent GC
        self._fact_worker = None
        self._message_count_since_extraction = 0

        main_layout = QVBoxLayout()

        # ===== Top bar: Model loading + Profile selection =====
        top_layout = QHBoxLayout()

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model_file)
        top_layout.addWidget(self.load_model_btn)

        profile_label = QLabel("Base Profile:")
        top_layout.addWidget(profile_label)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(list(profile_to_idx.keys()))
        self.profile_combo.currentTextChanged.connect(self.on_profile_changed)
        top_layout.addWidget(self.profile_combo)

        # Momentum slider
        momentum_label = QLabel("Momentum:")
        top_layout.addWidget(momentum_label)

        self.momentum_slider = QSlider(Qt.Horizontal)
        self.momentum_slider.setMinimum(50)
        self.momentum_slider.setMaximum(99)
        self.momentum_slider.setValue(85)
        self.momentum_slider.setFixedWidth(120)
        self.momentum_slider.valueChanged.connect(self.on_momentum_changed)
        top_layout.addWidget(self.momentum_slider)

        self.momentum_label = QLabel("0.95")
        self.momentum_label.setFixedWidth(35)
        top_layout.addWidget(self.momentum_label)

        self.reset_btn = QPushButton("Reset State")
        self.reset_btn.clicked.connect(self.reset_emotional_state)
        top_layout.addWidget(self.reset_btn)

        self.save_state_btn = QPushButton("Save State")
        self.save_state_btn.clicked.connect(self.save_state)
        top_layout.addWidget(self.save_state_btn)

        self.load_state_btn = QPushButton("Load State")
        self.load_state_btn.clicked.connect(self.load_state)
        top_layout.addWidget(self.load_state_btn)

        self.clear_memory_btn = QPushButton("Clear Memory")
        self.clear_memory_btn.setToolTip("Wipe long-term message memory (habituation resets)")
        self.clear_memory_btn.clicked.connect(self.clear_message_memory)
        top_layout.addWidget(self.clear_memory_btn)

        main_layout.addLayout(top_layout)

        # ===== Ollama Chat Settings =====
        chat_settings_layout = QHBoxLayout()

        self.chat_enabled_cb = QCheckBox("Chat")
        self.chat_enabled_cb.setChecked(True)
        self.chat_enabled_cb.setToolTip("Enable/disable LLM chat responses")
        chat_settings_layout.addWidget(self.chat_enabled_cb)

        chat_settings_layout.addWidget(QLabel("Model:"))
        self.ollama_model_input = QLineEdit("llama3")
        self.ollama_model_input.setFixedWidth(120)
        self.ollama_model_input.setToolTip("Ollama model name (e.g. llama3, mistral, gemma2)")
        self.ollama_model_input.editingFinished.connect(self.on_ollama_model_changed)
        chat_settings_layout.addWidget(self.ollama_model_input)

        chat_settings_layout.addWidget(QLabel("Persona:"))
        self.persona_input = QLineEdit("a friend")
        self.persona_input.setToolTip("Who should the chatbot be? (e.g. 'a therapist', 'a sarcastic teenager')")
        chat_settings_layout.addWidget(self.persona_input)

        self.ollama_status_label = QLabel("")
        self.ollama_status_label.setFixedWidth(20)
        chat_settings_layout.addWidget(self.ollama_status_label)

        main_layout.addLayout(chat_settings_layout)

        # ===== Mood Indicator Panel =====
        mood_group = QGroupBox("Emotional State")
        mood_layout = QVBoxLayout()

        self.dominant_mood_label = QLabel("Dominant Mood: --")
        self.dominant_mood_label.setAlignment(Qt.AlignCenter)
        self.dominant_mood_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        mood_layout.addWidget(self.dominant_mood_label)

        self.valence_label = QLabel("Valence Trend: --")
        self.valence_label.setAlignment(Qt.AlignCenter)
        mood_layout.addWidget(self.valence_label)

        # Profile similarity bars
        self.mood_bars = {}
        bars_layout = QHBoxLayout()
        for profile_name in profile_to_idx.keys():
            bar_container = QVBoxLayout()
            bar_label = QLabel(profile_name.capitalize())
            bar_label.setAlignment(Qt.AlignCenter)
            bar_label.setStyleSheet("font-size: 10px;")
            bar = QProgressBar()
            bar.setMinimum(0)
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%v%")
            color = MOOD_COLORS.get(profile_name, "#7a7a7a")
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            bar_container.addWidget(bar_label)
            bar_container.addWidget(bar)
            bars_layout.addLayout(bar_container)
            self.mood_bars[profile_name] = bar
        mood_layout.addLayout(bars_layout)
        mood_group.setLayout(mood_layout)
        main_layout.addWidget(mood_group)

        # ===== Middle: Splitter with Chat + Plot =====
        splitter = QSplitter(Qt.Horizontal)

        # Chat history
        chat_group = QGroupBox("Interaction History")
        chat_layout = QVBoxLayout()
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        chat_layout.addWidget(self.chat_history)
        chat_group.setLayout(chat_layout)
        splitter.addWidget(chat_group)

        # Plot widget
        plot_group = QGroupBox("Membrane Voltage")
        self.plot_layout = QVBoxLayout()
        plot_group.setLayout(self.plot_layout)
        splitter.addWidget(plot_group)

        splitter.setSizes([500, 500])
        main_layout.addWidget(splitter)

        # ===== Bottom: Text input =====
        input_layout = QHBoxLayout()

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type something... (e.g. 'You look great today')")
        self.text_input.returnPressed.connect(self.run_processing)
        input_layout.addWidget(self.text_input)

        self.run_processing_btn = QPushButton("Send")
        self.run_processing_btn.clicked.connect(self.run_processing)
        input_layout.addWidget(self.run_processing_btn)

        main_layout.addLayout(input_layout)

        # ===== Result label =====
        self.result_label = QLabel("Predicted emotions: ")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

        try:
            load_vocab()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_profile_changed(self, profile_name):
        if self.esm is not None:
            self.esm.reset(profile_name)
            self.update_mood_display()
            self.chat_history.append(
                f'<i style="color: #888;">-- Emotional state reset to: {profile_name} --</i>'
            )

    def on_momentum_changed(self, value):
        momentum = value / 100.0
        self.momentum_label.setText(f"{momentum:.2f}")
        if self.esm is not None:
            self.esm.momentum = momentum

    def reset_emotional_state(self):
        if self.esm is not None:
            profile = self.profile_combo.currentText()
            self.esm.reset(profile)
            self.update_mood_display()
            self.chat_history.append(
                f'<i style="color: #888;">-- Emotional state reset to: {profile} --</i>'
            )

    def save_state(self):
        if self.esm is None:
            QMessageBox.warning(self, "Warning", "No emotional state to save!")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Emotional State", "", "JSON files (*.json)"
        )
        if path:
            self.esm.save(path, conversation_history=self.conversation_history)
            QMessageBox.information(self, "Saved", "Emotional state saved.")

    def load_state(self):
        if self.esm is None:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Emotional State", "", "JSON files (*.json)"
        )
        if path:
            conv_history = self.esm.load(path)
            if conv_history:
                self.conversation_history = conv_history
            self.update_mood_display()
            self.chat_history.append(
                '<i style="color: #888;">-- Emotional state loaded from file --</i>'
            )

    def clear_message_memory(self):
        if self.esm is None:
            QMessageBox.warning(self, "Warning", "No emotional state loaded!")
            return
        self.esm.clear_memory()
        self.chat_history.append(
            '<i style="color: #888;">-- Long-term message memory cleared --</i>'
        )

    def on_ollama_model_changed(self):
        model_name = self.ollama_model_input.text().strip()
        if model_name:
            self.ollama_client = OllamaClient(model=model_name)
            self._update_ollama_status()

    def _update_ollama_status(self):
        if self.ollama_client and self.ollama_client.is_available():
            self.ollama_status_label.setText("\u2705")
            self.ollama_status_label.setToolTip(
                f"Connected to Ollama ({self.ollama_client.model})"
            )
        else:
            self.ollama_status_label.setText("\u274c")
            self.ollama_status_label.setToolTip(
                "Ollama not available. Run: ollama serve"
            )

    def _init_ollama(self):
        """Lazily initialize the Ollama client on first use."""
        if self.ollama_client is None:
            model_name = self.ollama_model_input.text().strip() or "llama3"
            self.ollama_client = OllamaClient(model=model_name)
            self._update_ollama_status()

    def _on_ollama_response(self, response):
        """Handle successful Ollama response (called from worker thread)."""
        self.conversation_history.append({
            "role": "assistant", "content": response, "timestamp": time.time()
        })
        # Keep conversation history bounded
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]

        self.chat_history.append(
            f'<b style="color: #4a9fc4;">NED:</b> {response}'
        )
        self.chat_history.append("")  # spacer
        self._set_input_enabled(True)

        # Extract facts every 5 exchanges (background, non-blocking)
        self._message_count_since_extraction += 1
        if self._message_count_since_extraction >= 5 and self.ollama_client:
            self._message_count_since_extraction = 0
            existing_user = self.esm.user_facts if self.esm else []
            existing_self = self.esm.self_facts if self.esm else []
            self._fact_worker = FactExtractorWorker(
                self.ollama_client, list(self.conversation_history),
                existing_user, existing_self,
            )
            self._fact_worker.finished.connect(self._on_facts_extracted)
            self._fact_worker.start()

    def _on_facts_extracted(self, user_facts, self_facts):
        """Store extracted facts in ESM (called from background thread)."""
        if self.esm is not None:
            if user_facts:
                self.esm.user_facts = user_facts
            if self_facts:
                self.esm.self_facts = self_facts

    def _on_ollama_error(self, error_msg):
        """Handle Ollama error (called from worker thread)."""
        self.chat_history.append(
            f'<i style="color: #aa4444;">  [Chat unavailable: {error_msg}]</i>'
        )
        self.chat_history.append("")  # spacer
        self._set_input_enabled(True)

    def _set_input_enabled(self, enabled):
        self.text_input.setEnabled(enabled)
        self.run_processing_btn.setEnabled(enabled)
        if enabled:
            self.text_input.setFocus()

    def update_mood_display(self):
        if self.esm is None:
            return

        summary = self.esm.get_mood_summary()
        for profile_name, bar in self.mood_bars.items():
            pct = int(summary.get(profile_name, 0) * 100)
            bar.setValue(pct)

        dominant = self.esm.get_dominant_mood()
        color = MOOD_COLORS.get(dominant, "#ffffff")
        self.dominant_mood_label.setText(f"Dominant Mood: {dominant.capitalize()}")
        self.dominant_mood_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {color};"
        )

        trend = self.esm.get_valence_trend()
        trend_text = f"Valence Trend (last 10): {trend:+.3f}"
        if trend > 0.1:
            trend_text += "  (improving)"
        elif trend < -0.1:
            trend_text += "  (declining)"
        else:
            trend_text += "  (stable)"
        self.valence_label.setText(trend_text)

    def load_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "checkpoints",
            "SafeTensor files (*.safetensors);;All Files (*)"
        )
        if not file_path:
            return

        try:
            model_instance = NeuroEmoDynamics(vocab, num_classes=num_of_classes, batch_size=batch_size).to(device)

            ckpt = load_file(file_path)

            emb_key = "text_encoder.embedding.weight"
            if emb_key in ckpt:
                w_ckpt = ckpt[emb_key]
                w = model_instance.text_encoder.embedding.weight
                if w_ckpt.shape != w.shape:
                    with torch.no_grad():
                        n = min(w.shape[0], w_ckpt.shape[0])
                        d = min(w.shape[1], w_ckpt.shape[1])
                        w[:n, :d].copy_(w_ckpt[:n, :d])
                    del ckpt[emb_key]

            # QLIF runtime buffers (V, spikes, ahp, refrac, etc.) are batch-size-dependent
            # and get resized on first forward pass anyway — skip them during loading
            qlif_runtime_buffers = {
                ".V", ".spikes", ".ahp", ".refrac", ".adaptation_current",
                ".neuromodulator", ".spike_values", ".synaptic_efficiency",
                ".dynamic_spike_probability.adaptation", ".rate_ema",
            }

            model_sd = model_instance.state_dict()
            filtered = {}
            skipped_buffers = []
            for k, v in ckpt.items():
                if k not in model_sd:
                    continue
                if model_sd[k].shape != v.shape:
                    # Check if it's a QLIF runtime buffer (shape mismatch is expected)
                    if any(k.endswith(buf) for buf in qlif_runtime_buffers):
                        skipped_buffers.append(k)
                        continue
                filtered[k] = v

            missing_after_filter = [k for k in model_sd.keys()
                                    if k not in filtered and k not in skipped_buffers]
            unexpected = [k for k in ckpt.keys() if k not in model_sd]

            model_instance.load_state_dict(filtered, strict=False)

            if hasattr(model_instance, "prev_feedback"):
                model_instance.prev_feedback = torch.zeros(batch_size, 512, device=device)

            model_instance.eval()
            self.model = model_instance

            # Initialize Emotional State Memory
            initial_profile = self.profile_combo.currentText()
            momentum = self.momentum_slider.value() / 100.0
            self.esm = EmotionalStateMemory(
                model=self.model,
                initial_profile=initial_profile,
                momentum=momentum,
                device=device,
            )
            self.update_mood_display()

            # Auto-load saved state if available
            if os.path.isfile(AUTOSAVE_PATH):
                try:
                    conv_history = self.esm.load(AUTOSAVE_PATH)
                    if conv_history:
                        self.conversation_history = conv_history
                    self.update_mood_display()
                    self.chat_history.append(
                        '<i style="color: #888;">-- Autosave restored --</i>'
                    )
                except Exception as e:
                    self.chat_history.append(
                        f'<i style="color: #aa4444;">-- Failed to load autosave: {e} --</i>'
                    )

            msg = []
            if unexpected:
                msg.append(f"Unexpected: {len(unexpected)}")
            if missing_after_filter:
                msg.append(f"Missing: {len(missing_after_filter)}")
            info = " ( " + ", ".join(msg) + " )" if msg else ""
            QMessageBox.information(self, "Success", f"Model loaded successfully{info}!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def run_processing(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return

        input_text = self.text_input.text().strip()
        if not input_text:
            QMessageBox.warning(self, "Warning", "Please enter some text!")
            return

        selected_profile = self.profile_combo.currentText()

        try:
            # Perspective transformation: "Du siehst gut aus" → "ich sehe gut aus"
            is_second_person = detect_second_person(input_text)
            transformed_text, was_transformed = transform_perspective(input_text)
            model_text = transformed_text if was_transformed else input_text

            # Log to chat
            self.chat_history.append(f'<b style="color: #cc8427;">You:</b> {input_text}')
            if was_transformed:
                self.chat_history.append(
                    f'<i style="color: #6a9fb5;">  [Internalized: "{transformed_text}"]</i>'
                )

            # Generate synthetic data based on ESM's current mood (not the fixed dropdown).
            # This breaks the vicious cycle: as mood slowly improves, sensory context follows.
            sensory_profile = self.esm.get_dominant_mood() if self.esm else selected_profile
            sensory_input, reward_signal = generate_synthetic_data(
                sensory_profile, timesteps, batch_size, input_size, reward_size, device=device
            )

            # Use ESM continuous profile vector instead of fixed profile_id
            profile_vec = self.esm.get_profile_vec(batch_size).to(device) if self.esm else None
            profile_ids = torch.full((batch_size,), profile_to_idx[selected_profile],
                                     dtype=torch.long, device=device)

            # Encode the (possibly transformed) text
            encoded = encode_text(model_text, vocab)
            encoded = pad_sequence(encoded, max_text_len, pad_value=vocab["<pad>"])
            text_input_tensor = torch.tensor([encoded] * batch_size, dtype=torch.long, device=device)

            with torch.no_grad():
                # Get text embedding for habituation matching (same call the model makes internally)
                text_embedding = self.model.text_encoder(text_input_tensor).mean(dim=0)  # (1024,)

                spikes, voltages, logits, aux_logits, serotonin, dopamine, norepinephrine, self_ref_score = self.model(
                    sensory_input, reward_signal, text_input_tensor, profile_ids, profile_vec=profile_vec
                )

            # Emotion prediction
            predicted_classes = logits.argmax(dim=1).cpu().numpy()
            predicted_emotions = [label_to_emotion.get(idx, "unknown") for idx in predicted_classes]
            predicted_emotions = list(set(predicted_emotions))

            # Compute probabilities for display
            probs = torch.nn.functional.softmax(logits.mean(dim=0), dim=0)
            prob_str = ", ".join(
                f"{label_to_emotion[i]}: {probs[i].item():.1%}" for i in range(6)
            )

            self.result_label.setStyleSheet("QLabel { color: #cc8427; font-weight: bold; }")
            self.result_label.setText(f"Predicted: {', '.join(predicted_emotions)}  |  {prob_str}")

            # Update Emotional State Memory using aux_logits (text-only emotion).
            # aux_logits reflect what the text *says*, not how the depressed
            # neural dynamics filter it — so positive text → positive valence.
            if self.esm is not None:
                shift_vector, valence, self_ref, hab_factor, hab_count = self.esm.compute_emotional_impact(
                    aux_logits, (serotonin, dopamine, norepinephrine), self_ref_score,
                    text_embedding=text_embedding
                )
                self.esm.update(shift_vector, text=input_text, valence=valence)
                self.update_mood_display()

                # Log emotional response with habituation info
                mood = self.esm.get_dominant_mood()
                color = MOOD_COLORS.get(mood, "#ffffff")
                hab_label = (
                    f"fresh" if hab_count <= 1 else
                    f"x{hab_count} ({hab_factor:+.1f})"
                )
                self.chat_history.append(
                    f'<span style="color: {color};">  Emotion: {", ".join(predicted_emotions)} '
                    f'| Valence: {valence:+.3f} | Self-ref: {self_ref:.2f} '
                    f'| Habit: {hab_label} '
                    f'| Mood: {mood}</span>'
                )

            # ===== Ollama Chat Response =====
            self.conversation_history.append({
                "role": "user", "content": input_text, "timestamp": time.time()
            })
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

            chat_on = self.chat_enabled_cb.isChecked()
            if chat_on and self.esm is not None:
                self._init_ollama()
                mood_summary = self.esm.get_mood_summary()
                system_prompt = self.ollama_client.build_system_prompt(
                    mood_summary=mood_summary,
                    dominant_mood=mood,
                    valence=valence,
                    emotion=", ".join(predicted_emotions),
                    habituation_factor=hab_factor,
                    habituation_count=hab_count,
                    valence_trend=self.esm.get_valence_trend(),
                    persona=self.persona_input.text().strip() or "a friend",
                    user_facts=self.esm.user_facts,
                    self_facts=self.esm.self_facts,
                    conversation_history=self.conversation_history,
                )
                self._set_input_enabled(False)
                self._ollama_worker = OllamaWorker(
                    self.ollama_client, input_text, system_prompt,
                    self.conversation_history[:-1]  # exclude current user msg (added by client)
                )
                self._ollama_worker.finished.connect(self._on_ollama_response)
                self._ollama_worker.error.connect(self._on_ollama_error)
                self._ollama_worker.start()
            else:
                self.chat_history.append("")  # spacer

            # Plot membrane voltage
            avg_voltage = voltages[:, 0, :].mean(dim=1).cpu().numpy()
            time_steps_arr = np.arange(len(avg_voltage))
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(time_steps_arr, avg_voltage, label="Average Voltage", color="#cc8427")
            ax.set_title("Average Membrane Voltage Over Time")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Voltage")
            ax.legend()
            fig.tight_layout()
            self.update_plot(fig)

            # Clear input for next message
            self.text_input.clear()
            self.text_input.setFocus()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during processing: {e}")

    def closeEvent(self, event):
        """Auto-save emotional state + conversation on window close."""
        if self.esm is not None:
            try:
                # Final fact extraction (synchronous, blocking — OK on close)
                if self.ollama_client and self.conversation_history:
                    try:
                        user_facts, self_facts = self.ollama_client.extract_facts(
                            self.conversation_history,
                            self.esm.user_facts,
                            self.esm.self_facts,
                        )
                        if user_facts:
                            self.esm.user_facts = user_facts
                        if self_facts:
                            self.esm.self_facts = self_facts
                    except Exception:
                        pass
                os.makedirs(os.path.dirname(AUTOSAVE_PATH), exist_ok=True)
                self.esm.save(AUTOSAVE_PATH, conversation_history=self.conversation_history)
            except Exception:
                pass  # don't block closing on save failure
        super().closeEvent(event)

    def update_plot(self, fig):
        if self.figure_canvas is not None:
            self.plot_layout.removeWidget(self.figure_canvas)
            self.figure_canvas.setParent(None)

        self.figure_canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.figure_canvas)
        self.figure_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()

    set_dark_mode(app)

    window.show()
    sys.exit(app.exec_())
