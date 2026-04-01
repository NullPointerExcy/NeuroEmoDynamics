# NeuroEmoDynamics: A Neurocognitive Profiling Quantum Spiking Neural Network (QSNN)

> This project is created purely out of interest and curiosity.

> **Hint**: 4 Epochs is enough to get a good result, but you can train it for longer if you want (maybe add some dropouts etc.).

NeuroEmoDynamics is a biologically inspired quantum spiking neural network (QSNN) designed to simulate complex cognitive and
emotional states. It combines custom [QLIF-Neurons](https://github.com/DanjelPiDev/QLIF-Neurons) with neuromodulatory simulation,
text-based emotion analysis, continuous emotional state tracking, and LLM-powered conversational AI -- all wrapped in an
interactive PyQt5 desktop application.

## Features

### Biologically Inspired Brain Architecture

The network models 7 brain regions using QLIF neuron layers:

| Region | Neurons | Role | Neuromodulation |
|--------|---------|------|-----------------|
| **Prefrontal Cortex (PFC)** | 512 | Sensory input processing, executive control | Serotonin (gain) |
| **Amygdala** | 256 | Emotional response, fear processing | Norepinephrine (gain) |
| **Hippocampus** | 256 | Memory formation, context | Adaptive threshold |
| **Thalamus** | 256 | Sensory relay, filtering | Serotonin (gain) |
| **Striatum** | 1024 | Decision-making, reward integration | Dopamine (prob_slope) |
| **Raphe, VTA, Locus Coeruleus** | -- | Neuromodulator source regions | -- |

- **Cross-Connectivity** integrates information across amygdala, hippocampus, and thalamus.
- **Feedback loops** from the striatum back to PFC simulate reward prediction and learning (clamped to [-10, 10]).

### QLIF-Neuron Model

Uses an enhanced Quantum Leaky Integrate-and-Fire (QLIF) model with:
- Neuromodulation mechanisms (serotonin, dopamine, norepinephrine) that regulate different regions.
- Adaptive thresholds and noise integration.
- **Quantum mode**: Stochastic spiking with dynamic spike probability based on neuron adaptation state, adding biological realism and uncertainty.
- See [QLIF-Neurons](https://github.com/DanjelPiDev/QLIF-Neurons) for more information.

### Psychological Profiles

Five discrete profiles, each with a unique neuromodulatory signature:

| Profile | Serotonin | Dopamine | Norepinephrine | Characteristics |
|---------|-----------|----------|----------------|-----------------|
| **Depressed** | 0.3 | 0.4 | 0.6 | Low motivation, sadness bias |
| **Anxious** | 0.6 | 0.5 | 0.8 | High arousal, fear/anger bias |
| **Healthy** | 0.8 | 0.7 | 0.5 | Balanced, joy/love bias |
| **Impulsive** | 0.5 | 0.6 | 0.7 | High reactivity, anger/fear bias |
| **Resilient** | 0.7 | 0.5 | 0.6 | Stable, recovery-oriented |

During training, 30% of batches use **profile interpolation** (e.g., 70% healthy + 30% anxious) to teach the model continuous emotional blends.

### Emotional State Memory (ESM)

A continuous 256-dimensional emotional state vector that evolves over conversation:

- **Momentum-based smoothing** (default 0.85): Smooth transitions instead of discrete jumps.
- **Habituation**: Repeated messages lose impact (30% per repetition). After 3+ repetitions, effects can reverse (e.g., a compliment becomes perceived as sarcastic).
- **Negativity bias**: Negative inputs hit 2.5x harder than positive ones (realistic depression modeling).
- **Self-reference boost**: First-person statements ("I am strong") get 2x emotional impact.
- **Valence tracking**: Continuous -1 to +1 scale with trend detection (improving/declining/stable).
- **Persistence**: State, conversation history, and extracted user/self facts are saved/loaded across sessions.

### Text Processing & Emotional Override

- **Text Encoder**: Embedding + Bidirectional LSTM producing a 1024-dim representation.
- **Gating mechanism**: Text can override neural dynamics (e.g., *"I feel strong"* can shift a depressed profile toward joy).
- **Self-reference detection**: Identifies first-person pronouns to amplify self-directed statements.
- **Perspective transform**: Converts second-person to first-person (e.g., "You look great" -> "I look great") to simulate internalization.

### Conversational AI (Two Backends)

#### Ollama (Text-Prompt Based)
- Sends a rich system prompt describing current mood, neurotransmitter levels, and valence trend to a local Ollama instance.
- The LLM generates responses conditioned on the emotional state description.
- Supports configurable model names and persona.
- **Fact extraction**: Periodically extracts user/self facts from conversation to maintain persistent context.

#### Llama-LoRA (State-Vector Based)
- Runs Llama3-8B-Instruct locally with a fine-tuned LoRA adapter.
- A **StateVectorProjector** maps the raw 267-dim SNN state vector directly into 4 virtual prefix tokens for Llama3.
- The LLM processes `[virtual_tokens] [context] [user_message]`, conditioning its response on the numerical emotional state.
- Advantage: Direct state encoding instead of lossy text approximation.

### Interactive Desktop UI (PyQt5)

- **Profile selector** to switch between the 5 psychological profiles.
- **Momentum slider** (0.50 -- 0.99) to adjust emotional state smoothing.
- **Emotional state display**: Dominant mood with color coding, valence trend, and profile similarity bars.
- **Chat interface** with annotated interaction history (detected emotions, valence, habituation, mood shifts).
- **Membrane voltage plot**: Real-time visualization of average striatum voltage.
- **State management**: Save, load, and reset emotional state. Auto-save/restore on exit/launch.
- **Chat backend toggle**: Switch between Llama-LoRA and Ollama at runtime.

### 3D Brain Visualization

- Interactive Plotly visualization with 11 anatomically positioned brain regions.
- Connection lines showing weight magnitude.
- Neuromodulator pathways (serotonin, dopamine, norepinephrine) highlighted.
- Run `model_neuron_plot.py` with a trained model, then open `interactive_viz.html`.

<div align="center">
    <img src="images/NAV.png" alt="Interactive Visualization" width="1000"/>
</div>

## Project Structure

```
src/
├── models/
│   ├── neuro_emotional_dynamics.py    # Core QSNN architecture (7 brain regions)
│   ├── emotional_state_memory.py      # Continuous state tracking + habituation
│   ├── text_encoder.py                # Bidirectional LSTM text encoding
│   ├── ollama_client.py               # Ollama LLM integration
│   ├── llama_client.py                # Local Llama3 + LoRA integration
│   ├── state_projector.py             # State vector -> Llama embeddings
│   ├── model_neuron_plot.py           # 3D brain visualization (Plotly)
│   └── train/
│       ├── train_ned_model.py         # Main NED training script
│       └── train_llama_lora.py        # Llama3 LoRA fine-tuning
├── ui/
│   ├── ned_ui.py                      # PyQt5 interactive GUI
│   └── style.py                       # Dark mode styling
├── data/
│   ├── emotion_dataset.py             # dair-ai/emotion dataset wrapper
│   ├── sst2_dataset.py                # SST-2 dataset wrapper
│   └── synthetic_data.py              # Profile-based synthetic data generator
├── utils/
│   ├── helper_functions.py            # Tokenization, vocab building
│   └── perspective_transform.py       # 2nd person -> 1st person conversion
└── checkpoints/
    ├── neuro_emo_dynamics_v10q1.safetensors   # Trained model
    ├── llama_lora/                             # LoRA adapter + projector
    └── states/                                 # Saved emotional states
```

## Installation

```bash
# Clone the repository
git clone https://github.com/DanjelPiDev/NeuroEmoDynamics.git
cd NeuroEmoDynamics

# Install dependencies
pip install -r requirements.txt
```

**Additional requirements:**
- For Ollama backend: [Ollama](https://ollama.com/) must be installed and running (`ollama serve`).
- For Llama-LoRA backend: A pre-trained LoRA adapter in `src/checkpoints/llama_lora/` (requires ~24 GB VRAM for training).

## Usage

### Training the Model

```bash
cd src
python -m models.train.train_ned_model
```

Training uses the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset (6 classes: sadness, joy, love, anger, fear, surprise) with mixed precision, profile-biased sampling, and a multi-component loss (cross-entropy + neuromod consistency + coherence + focal + auxiliary).

### Training the Llama LoRA Adapter (Optional)

```bash
cd src
python -m models.train.train_llama_lora
```

Fine-tunes Llama3-8B-Instruct on [facebook/empathetic_dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) with state vector conditioning via a learned projector.

### Running the UI

```bash
cd src
python ui/ned_ui.py
```

## How It Works

### 1. Profile-Based Neuromodulation

Each psychological profile is assigned a neuromodulatory signature. These modulate serotonin, dopamine, and norepinephrine levels, influencing emotional response, cognitive flexibility, and attention regulation.

### 2. Text Processing & Emotional Override

The text encoder extracts linguistic features. Gating layers determine if textual information can override the emotional profile:
- A **depressed** profile reading *"I feel strong"* may shift towards **joy**.
- An **anxious** profile reading *"Everything is fine"* may reduce fear responses.

### 3. Spiking Network & Decision Making

PFC processes sensory input and modulates emotional states. Feedback from the amygdala, hippocampus, and thalamus refines responses. The striatum integrates all signals to produce final emotion classification (6 classes).

### 4. Emotional State Evolution

The ESM tracks how the emotional state changes over conversation. Instead of discrete category switches, the 256-dim state vector smoothly evolves via exponential moving average. Habituation prevents repeated messages from having full effect, and negativity bias makes recovery from negative states harder -- modeling realistic psychological dynamics.

### 5. Conversational Response

Based on the current emotional state, one of two backends generates a response:
- **Ollama**: Receives a text description of the current mood, neurotransmitter levels, and conversation history.
- **Llama-LoRA**: Receives the raw state vector as virtual prefix tokens, conditioning the response directly on the numerical emotional state.

## Roadmap

- [x] Improve text-based emotion influence
- [x] Optimize the LIF neuron feedback mechanisms
- [x] Interactive PyQt5 UI with real-time emotional state tracking
- [x] Emotional State Memory with habituation and negativity bias
- [x] LLM chat integration (Ollama + Llama-LoRA backends)
- [x] State vector projection for direct LLM conditioning
- [x] Session persistence (auto-save/restore)
- [ ] Experiment with reinforcement learning for adaptive emotion modulation

## References

This project uses the dataset **[facebook/empathetic_dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues)** from Hugging Face.

```bibtex
@inproceedings{rashkin-etal-2019-towards,
    title = "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset",
    author = "Rashkin, Hannah  and
      Smith, Eric Michael  and
      Li, Margaret  and
      Boureau, Y-Lan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1534",
    doi = "10.18653/v1/P19-1534",
    pages = "5370--5381",
}
```

## License

[AGPL-3.0 License](./LICENSE)
