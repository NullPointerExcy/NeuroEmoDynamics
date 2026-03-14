import json
import time
import requests


def _format_time_ago(timestamp):
    """Format a Unix timestamp as a human-readable 'time ago' string."""
    if not timestamp:
        return ""
    elapsed = time.time() - timestamp
    if elapsed < 60:
        return "just now"
    elif elapsed < 3600:
        mins = int(elapsed / 60)
        return f"{mins}m ago"
    elif elapsed < 86400:
        hours = int(elapsed / 3600)
        return f"{hours}h ago"
    else:
        days = int(elapsed / 86400)
        return f"{days}d ago"


class OllamaClient:
    """Sync client for Ollama's chat API with emotional context injection."""

    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._available = None  # cached availability check

    def is_available(self):
        """Check if Ollama is running and the model is accessible."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            # Match with or without tag (e.g., "llama3" matches "llama3:latest")
            return any(
                m == self.model or m.startswith(f"{self.model}:")
                for m in models
            )
        except (requests.ConnectionError, requests.Timeout):
            return False

    def build_system_prompt(self, mood_summary, dominant_mood, valence,
                            emotion, habituation_factor, habituation_count,
                            valence_trend, persona, user_facts=None,
                            self_facts=None, conversation_history=None):
        """Build a system prompt that injects the current emotional state."""

        # Format mood breakdown
        mood_parts = [f"{name}: {pct:.0%}" for name, pct in mood_summary.items()]
        mood_breakdown = ", ".join(mood_parts)

        # Habituation description
        if habituation_count <= 1:
            hab_desc = "This is a fresh message you haven't heard before."
        elif habituation_factor > 0.5:
            hab_desc = f"You've heard something like this {habituation_count} times. It still has some impact."
        elif habituation_factor > 0:
            hab_desc = (f"You've heard this {habituation_count} times now. "
                        "It's starting to feel hollow and repetitive.")
        elif habituation_factor == 0:
            hab_desc = (f"You've heard this {habituation_count} times. "
                        "It means nothing to you anymore — empty words.")
        else:
            hab_desc = (f"You've heard this {habituation_count} times. "
                        "It's become annoying and feels insincere or manipulative.")

        # Valence trend description
        if valence_trend > 0.1:
            trend_desc = "Your mood has been slowly improving recently."
        elif valence_trend < -0.1:
            trend_desc = "Your mood has been declining recently."
        else:
            trend_desc = "Your mood has been relatively stable."

        # Long-term memory: facts from previous sessions
        memory_parts = []
        if self_facts:
            self_section = "\n".join(f"- {fact}" for fact in self_facts)
            memory_parts.append(f"What you know about yourself (your own identity, things you've said/decided):\n{self_section}")
        if user_facts:
            user_section = "\n".join(f"- {fact}" for fact in user_facts)
            memory_parts.append(f"What you remember about the user:\n{user_section}")
        if memory_parts:
            memory_block = "\n\n".join(memory_parts)
            memory_block = f"\n{memory_block}\nUse this knowledge naturally — don't announce that you \"remember\" unless it fits.\n"
        else:
            memory_block = ""

        # Time context: when was the last conversation?
        time_context = ""
        if conversation_history:
            # Find last assistant message timestamp to determine session gap
            last_ts = None
            for msg in reversed(conversation_history[:-1]):  # exclude current user msg
                if msg.get("timestamp"):
                    last_ts = msg["timestamp"]
                    break
            if last_ts:
                gap = time.time() - last_ts
                if gap > 86400:
                    days = int(gap / 86400)
                    time_context = f"\nYou last spoke with the user {days} day{'s' if days > 1 else ''} ago. Acknowledge the time gap naturally if appropriate.\n"
                elif gap > 3600:
                    hours = int(gap / 3600)
                    time_context = f"\nYou last spoke with the user {hours} hour{'s' if hours > 1 else ''} ago.\n"
                # Under 1 hour: no special mention needed (same session)

        return f"""You are {persona}.
{memory_block}{time_context}
Your inner emotional state right now:
- Dominant mood: {dominant_mood}
- Mood breakdown: {mood_breakdown}
- {trend_desc}

What you just perceived from the user's message:
- Detected emotion: {emotion} (valence: {valence:+.3f})
- {hab_desc}

Guidelines:
- Let your mood naturally color your responses (tone, energy, word choice, length)
- If your mood is depressed: shorter responses, less enthusiasm, might deflect compliments, low energy
- If anxious: overthinking, asking clarifying questions, hedging, nervous energy
- If healthy: warm, engaged, balanced, genuinely interested
- If impulsive: quick reactions, strong opinions, excitable, jumping between topics
- If resilient: steady, optimistic but realistic, grounded, supportive
- Do NOT explicitly state your mood or emotional state unless the user directly asks how you feel
- Respond in the same language the user writes in
- Keep responses concise (2-4 sentences unless the conversation naturally calls for more)
- You are having a natural conversation — respond to what the user says, don't just reflect emotions"""

    @staticmethod
    def _strip_timestamps(history):
        """Remove timestamp keys from history dicts (Ollama expects only role+content)."""
        if not history:
            return history
        return [{"role": m["role"], "content": m["content"]} for m in history]

    def chat(self, user_message, system_prompt, history=None):
        """Send a message to Ollama and return the response text.

        Args:
            user_message: The user's latest message
            system_prompt: System prompt with emotional context
            history: List of {"role": ..., "content": ..., "timestamp": ...} dicts

        Returns:
            str: The assistant's response text

        Raises:
            ConnectionError: If Ollama is unreachable
            RuntimeError: If the API returns an error
        """
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(self._strip_timestamps(history))

        messages.append({"role": "user", "content": user_message})

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=60,
            )
        except requests.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running? (ollama serve)"
            )
        except requests.Timeout:
            raise ConnectionError("Ollama request timed out (60s).")

        if r.status_code != 200:
            raise RuntimeError(f"Ollama API error {r.status_code}: {r.text[:200]}")

        data = r.json()
        return data.get("message", {}).get("content", "").strip()

    def extract_facts(self, conversation_history, existing_user_facts=None,
                      existing_self_facts=None):
        """Ask the LLM to extract facts about both the user and the assistant.

        Returns:
            tuple: (user_facts: list[str], self_facts: list[str])
        """
        if not conversation_history:
            return existing_user_facts or [], existing_self_facts or []

        # Build a compact transcript of recent messages
        recent = conversation_history[-20:]
        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )

        existing_section = ""
        if existing_user_facts or existing_self_facts:
            parts = []
            if existing_user_facts:
                parts.append("Previously known facts about the User:\n"
                             + "\n".join(f"- {f}" for f in existing_user_facts))
            if existing_self_facts:
                parts.append("Previously known facts about the Assistant:\n"
                             + "\n".join(f"- {f}" for f in existing_self_facts))
            existing_section = "\n\n".join(parts) + "\n\n"

        prompt = f"""{existing_section}Here is a recent conversation between User and Assistant:

{transcript}

Extract important facts from this conversation into TWO categories:

1. **user_facts**: Facts about the User (human). Include:
   - Their name, age, job, location, relationships
   - Preferences, interests, hobbies
   - Important events or situations they mentioned

2. **self_facts**: Facts about the Assistant (you/NED). Include:
   - Name the assistant chose or was given
   - Personality traits, preferences, opinions the assistant expressed
   - Promises or commitments the assistant made
   - Things the assistant said about itself

Keep previously known facts if still valid. Remove duplicates.
Return ONLY a JSON object with two arrays, nothing else. Example:
{{"user_facts": ["Name is Alex", "Works as a teacher"], "self_facts": ["Calls itself Sarah", "Said it likes rainy days"]}}
If no facts found for a category, use an empty array."""

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=30,
            )
            if r.status_code != 200:
                return existing_user_facts or [], existing_self_facts or []

            text = r.json().get("message", {}).get("content", "").strip()

            # Parse JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            # Find the JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    user_f = [str(f) for f in result.get("user_facts", []) if f]
                    self_f = [str(f) for f in result.get("self_facts", []) if f]
                    return (user_f or existing_user_facts or [],
                            self_f or existing_self_facts or [])
            return existing_user_facts or [], existing_self_facts or []

        except Exception:
            return existing_user_facts or [], existing_self_facts or []
