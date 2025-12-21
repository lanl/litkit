"""LLM-specific exceptions and error helpers.

Provides typed exceptions for LLM operations, replacing brittle string-matching
with structured error handling. All exceptions inherit from LLMError for
consistent catching patterns.
"""


class LLMError(Exception):
    """Base exception for all LLM operations.
    
    Catch this to handle any LLM-related error generically.
    """
    pass


class AuthError(LLMError):
    """Authentication or API key failure.
    
    Raised when:
    - API key is missing or invalid
    - Token is expired or revoked
    - Endpoint returns 401/403
    """
    pass


class ModelNotFoundError(LLMError):
    """Model not loaded or not recognized by endpoint.
    
    Raised when:
    - Local server has no model loaded (LM Studio, Ollama, etc.)
    - Model name doesn't match any available model
    - Endpoint returns 404 with model-related message
    """
    pass


class EndpointNotSupportedError(LLMError):
    """Endpoint doesn't support required API.
    
    Raised when:
    - Responses API called on endpoint that only supports Chat
    - Local endpoint doesn't implement required features
    - o-series model requested but endpoint lacks Responses API
    """
    pass


class ContextOverflowError(LLMError):
    """Input context exceeds model's context window.
    
    Raised when:
    - Prompt + context exceeds max_tokens
    - Server returns context length error
    - HTTP 413 Payload Too Large
    
    This error is recoverable via chunk trimming and max_out reduction.
    """
    pass


class TransportError(LLMError):
    """Network or connection failure.
    
    Raised when:
    - Connection timeout
    - DNS resolution failure
    - Server unreachable
    - TLS/SSL errors
    """
    pass


def is_overflow_error(exc: Exception) -> bool:
    """Check if exception indicates context/token overflow.
    
    Centralizes string-matching logic for overflow detection. This is used
    to determine if an error is recoverable via chunk trimming.
    
    Args:
        exc: Any exception (typically from OpenAI SDK)
    
    Returns:
        True if the error indicates context/token overflow
    
    Examples of matched error messages:
        - "context length exceeded"
        - "maximum context length"
        - "token limit exceeded"
        - "prompt too long"
        - HTTP 413 responses
    """
    msg = (str(exc) or "").lower()
    return any(s in msg for s in (
        "context length",
        "maximum context length",
        "exceeds context window",
        "token limit",
        "too many tokens",
        "reduce the length of the messages",
        "max tokens",
        "prompt too long",
        "input too long",
        "payload too large",
        "413",  # HTTP 413 Payload Too Large
    ))


def is_auth_error(exc: Exception) -> bool:
    """Check if exception indicates authentication failure.
    
    Args:
        exc: Any exception
    
    Returns:
        True if the error indicates auth/API key issues
    """
    msg = (str(exc) or "").lower()
    return any(s in msg for s in (
        "unauthorized",
        "invalid api key",
        "401",
        "403",
        "authentication",
        "permission denied",
    ))


def is_model_not_found(exc: Exception) -> bool:
    """Check if exception indicates model not found/loaded.
    
    Args:
        exc: Any exception
    
    Returns:
        True if the error indicates missing model
    """
    msg = (str(exc) or "").lower()
    return any(s in msg for s in (
        "no models loaded",
        "model_not_found",
        "model not found",
        "404",
    )) and "model" in msg


def is_endpoint_unsupported(exc: Exception) -> bool:
    """Check if exception indicates unsupported endpoint/API.
    
    Args:
        exc: Any exception
    
    Returns:
        True if the error indicates endpoint doesn't support the requested API
    """
    msg = (str(exc) or "").lower()
    return any(s in msg for s in (
        "404",
        "not found",
        "405",
        "method not allowed",
        "responses.create",
        "unknown parameter",
        "unexpected argument",
        "unrecognized field",
        "invalid request body",
        "schema validation",
        "unsupported field",
        "does not support reasoning",
        "unsupported parameter",
    ))


def clarify_error(model: str, base_url: str, exc: Exception) -> str:
    """Generate operator-grade error message from raw exception.
    
    Transforms cryptic API errors into actionable messages with:
    - What went wrong
    - Likely cause
    - Concrete fix suggestions
    
    Args:
        model: Model name that was requested
        base_url: Endpoint URL that was called
        exc: The exception that occurred
    
    Returns:
        Formatted error message suitable for CLI output
    """
    msg = (str(exc) or "").strip()
    
    # Model not loaded (common for local servers)
    if is_model_not_found(exc):
        return (
            "[llm] ERROR: The endpoint is up but has no model loaded "
            f"(or does not recognize {model!r}).\n"
            f"  • Endpoint: {base_url}\n"
            "  • Fix (LM Studio): open LM Studio, load a chat model, "
            "and enable the local server (or run: `lms load <model_name>`). "
            "Then rerun your command.\n"
            "  • Alt: use OpenAI — e.g., "
            "`--llm-model o3-mini --openai-api-key $OPENAI_API_KEY`.\n"
            "  • Alt: retrieval only — add `--no-llm`.\n"
            f"  • Provider message: {msg}\n"
        )
    
    # Authentication failure
    if is_auth_error(exc):
        return (
            "[llm] ERROR: Authentication failed for the LLM endpoint.\n"
            f"  • Endpoint: {base_url}\n"
            "  • Fix: pass a valid `--openai-api-key` (OpenAI), or for "
            "local servers set a dummy token or none, depending on the "
            "server's requirements.\n"
            f"  • Provider message: {msg}\n"
        )
    
    # Endpoint doesn't support required API
    if is_endpoint_unsupported(exc):
        return (
            "[llm] ERROR: The endpoint doesn't support the requested API.\n"
            f"  • Endpoint: {base_url}\n"
            f"  • Model: {model}\n"
            "  • Fix: Use an OpenAI endpoint for o-series (Responses API), "
            "switch to a chat-compatible local model "
            "(e.g., --llm-model gpt-oss:20b), or run retrieval-only "
            "with --no-llm.\n"
            f"  • Provider message: {msg}\n"
        )
    
    # Context overflow (shouldn't normally reach here - handled by retry)
    if is_overflow_error(exc):
        return (
            "[llm] ERROR: Context length exceeded after retry attempts.\n"
            f"  • Endpoint: {base_url}\n"
            f"  • Model: {model}\n"
            "  • Fix: Reduce --top-chunks or context size.\n"
            f"  • Provider message: {msg}\n"
        )
    
    # Generic fallback
    return (
        "[llm] ERROR: LLM call failed.\n"
        f"  • Endpoint: {base_url}\n"
        f"  • Model: {model}\n"
        "  • Try: load a local model, switch to an OpenAI model with "
        "a valid API key, or run with `--no-llm`.\n"
        f"  • Provider message: {msg}\n"
    )
