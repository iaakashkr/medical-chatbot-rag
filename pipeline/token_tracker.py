# token_tracker.py
class TokenTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def log_step(self, step_name, prompt_tokens, completion_tokens):
        step_usage = {
            "step": step_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.steps.append(step_usage)

        # Update totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += step_usage["total_tokens"]

    def get_summary(self):
        return {
            "steps": self.steps,
            "overall": {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            }
        }


# Create a single instance to import across pipeline
token_tracker = TokenTracker()
