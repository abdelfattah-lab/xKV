"""
Generator that properly handles xKV cache creation.

This generator ensures that the custom xKV cache is created via
_prepare_cache_for_generation before the forward pass, which is
necessary for KV compression methods to work correctly.

Supports both single-turn and multi-turn inference modes.
"""
import gc
import torch
from transformers import GenerationConfig


class Generator:
    """
    A unified generator class that properly initializes the xKV cache
    via _prepare_cache_for_generation.

    Supports:
    - Single-turn: generate_from_prompt(prompt_ids, max_new_tokens)
    - Multi-turn: prefill(context_ids) + generate(query_ids, max_new_tokens)
    """

    def __init__(self, model, tokenizer):
        model.eval()
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None

    def clear(self):
        """Clear the KV cache and free GPU memory."""
        self.past_kv = None
        gc.collect()
        torch.cuda.empty_cache()

    def _prepare_cache(self):
        """
        Prepare the cache using _prepare_cache_for_generation.
        This ensures xKV custom cache is created if the model is patched.
        """
        model_inputs = {}
        self.model._prepare_cache_for_generation(
            GenerationConfig(), model_inputs, None, None, None, None
        )
        return model_inputs["past_key_values"]

    def prefill(self, input_ids, attention_mask=None):
        """
        Prefill the context and cache KV values.

        Args:
            input_ids: Tokenized context (batch_size=1, seq_len)
            attention_mask: Optional attention mask

        Returns:
            The model output logits (can be ignored for prefill-only)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        # Create cache via _prepare_cache_for_generation (triggers xKV patch)
        if self.past_kv is None:
            self.past_kv = self._prepare_cache()

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                past_key_values=self.past_kv,
            )

        self.past_kv = out.past_key_values
        return out.logits

    def generate(self, input_ids, max_new_tokens=128, attention_mask=None):
        """
        Generate tokens using the cached KV values.

        Args:
            input_ids: Tokenized query (batch_size=1, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            attention_mask: Optional attention mask

        Returns:
            Generated token ids (including input_ids)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        # Initialize cache if not already done
        if self.past_kv is None:
            self.past_kv = self._prepare_cache()

        eos_token_id = self.tokenizer.eos_token_id
        generated_ids = input_ids.clone()

        for i in range(max_new_tokens):
            with torch.no_grad():
                if i == 0:
                    # First token: process full query
                    out = self.model(
                        input_ids=input_ids,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=self.past_kv,
                    )
                else:
                    # Subsequent tokens: only process last token
                    out = self.model(
                        input_ids=next_token,
                        use_cache=True,
                        return_dict=True,
                        past_key_values=self.past_kv,
                    )

            self.past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == eos_token_id:
                break

        return generated_ids

    def generate_from_prompt(self, input_ids, max_new_tokens=128):
        """
        Single-turn generation: prefill prompt and generate continuation.

        This is the high-level API for standard single-turn inference.
        It clears any existing cache, prefills the prompt, and generates.

        Args:
            input_ids: Tokenized prompt (batch_size=1, seq_len)
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated token ids (including input prompt)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        # Clear any existing cache
        self.clear()

        # Initialize fresh cache
        self.past_kv = self._prepare_cache()

        eos_token_id = self.tokenizer.eos_token_id
        generated_ids = input_ids.clone()

        # Prefill the prompt
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                past_key_values=self.past_kv,
            )
        self.past_kv = out.past_key_values
        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == eos_token_id:
            return generated_ids

        # Continue generating
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=self.past_kv,
                )

            self.past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == eos_token_id:
                break

        return generated_ids

    def prefill_and_generate(self, context_ids, query_ids, max_new_tokens=128):
        """
        Convenience method: prefill context, then generate with query.

        Args:
            context_ids: Tokenized context
            query_ids: Tokenized query
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, generated_ids)
        """
        # Clear any existing cache
        self.clear()

        # Prefill context
        self.prefill(context_ids)

        # Generate with query
        output_ids = self.generate(query_ids, max_new_tokens=max_new_tokens)

        # Decode only the generated part (excluding query)
        generated_ids = output_ids[0, query_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, generated_ids


# Backward compatibility alias
MultiTurnGenerator = Generator
