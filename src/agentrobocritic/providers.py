"""Provider-native requests for the benchmark evaluation."""
from dataclasses import dataclass
from urllib.parse import quote

from .data import MUSE_SYSTEM, PROTOCOL
from .scoring import parse


@dataclass(frozen=True)
class Profile:
    api: str
    model: str
    base_url: str
    key_env: str | None
    tokens: int = 8192


MUSE = "cx-cmu/AgentRoboCritic-Muse-Glimmer-30B"
PROFILES = {
    # Standard Alibaba Cloud Model Studio (DashScope) endpoint. To use another endpoint,
    # pass it with --base-url together with --key-env QWEN_API_KEY.
    "qwen": Profile("qwen", "qwen4-max", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY"),
    "glm": Profile("glm", "glm-5.3-flash", "https://api.z.ai/api/paas/v4", "GLM_API_KEY"),
    "kimi": Profile("kimi", "kimi-k3.1-pro", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "deepseek": Profile("deepseek", "deepseek-flash", "https://api.deepseek.com", "DEEPSEEK_API_KEY", 65536),
    "anthropic": Profile("anthropic", "claude-fable-5-2", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "openai": Profile("responses", "gpt-6.1-sol", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": Profile("gemini", "gemini-4-carbon", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
    "chat": Profile("chat", "", "http://127.0.0.1:8000/v1", None),
    "muse": Profile("muse", MUSE, "http://127.0.0.1:8000/v1", None, 1024),
    "muse-base": Profile("muse-base", "meta-models/Muse-Glimmer-30B", "http://127.0.0.1:8000/v1", None, 4096),
    "fireworks": Profile("fireworks", "", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", 1024),
}
MUSE_APIS = {"muse", "muse-base", "fireworks"}
URI = "data:image/png;base64,"


def request(profile, system, parts):
    api, model, tokens = profile.api, profile.model, profile.tokens
    schema = PROTOCOL["schema"]
    fmt = {"name": "agentrobocritic_action", "strict": True, "schema": schema}
    if api == "responses":
        content = [{"type": "input_text", "text": v} if k == "text" else
                   {"type": "input_image", "image_url": URI + v, "detail": "high"} for k, v in parts]
        return "/responses", {"model": model, "instructions": system, "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "high"}, "max_output_tokens": tokens, "store": False,
            "text": {"format": {"type": "json_schema", **fmt}}}
    if api == "anthropic":
        content = [{"type": "text", "text": v} if k == "text" else {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": v}} for k, v in parts]
        return "/messages", {"model": model, "system": system, "messages": [{"role": "user", "content": content}],
            "max_tokens": tokens, "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": schema}}}
    if api == "gemini":
        content = [{"text": v} if k == "text" else {"inlineData": {"mimeType": "image/png", "data": v}} for k, v in parts]
        return "/models/" + quote(model, safe="") + ":generateContent", {
            "systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": content}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}, "maxOutputTokens": tokens,
                                 "responseMimeType": "application/json", "responseJsonSchema": schema}}

    content = []
    for kind, value in parts:
        if kind == "text":
            content.append({"type": "text", "text": value})
        else:
            image = {"url": URI + value}
            if api == "deepseek":
                image["detail"] = "high"
            content.append({"type": "image_url", "image_url": image})
    body = {"model": model, "stream": False, "messages": [
        {"role": "system", "content": MUSE_SYSTEM if api in MUSE_APIS else system},
        {"role": "user", "content": content}]}
    if api in MUSE_APIS:
        body.update(temperature=0, seed=42, max_tokens=tokens)
        if api == "fireworks":
            body["context_length_exceeded_behavior"] = "error"
    elif api in ("deepseek", "glm"):
        body.update(reasoning_effort="high", thinking={"type": "enabled"}, max_tokens=tokens,
                    response_format={"type": "json_object"})
    elif api in ("kimi", "qwen"):
        body.update(reasoning_effort="high", max_completion_tokens=tokens,
                    response_format={"type": "json_schema", "json_schema": fmt})
    else:
        body.update(max_tokens=tokens, temperature=0)
    return "/chat/completions", body


def headers(api, key):
    if api == "anthropic":
        return {"anthropic-version": "2023-06-01", "x-api-key": key}
    if api == "gemini":
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"} if key else {}


def read_answer(api, result):
    if api == "responses":
        stop = result.get("status")
        blocks = [b for item in result.get("output", []) if item["type"] == "message"
                  for b in item["content"]]
        texts = [b["text"] if b["type"] == "output_text" else None for b in blocks]
        complete = stop == "completed"
    elif api == "anthropic":
        stop = result.get("stop_reason")
        texts = [b["text"] for b in result.get("content", []) if b["type"] == "text"]
        complete = stop == "end_turn"
    elif api == "gemini":
        choices = result.get("candidates") or [{}]
        stop = choices[0].get("finishReason")
        texts = ["".join(p["text"] for p in choices[0].get("content", {}).get("parts", [])
                         if "text" in p and not p.get("thought"))]
        complete = len(choices) == 1 and stop == "STOP"
    else:
        choices = result.get("choices") or [{}]
        stop = choices[0].get("finish_reason")
        texts = [choices[0].get("message", {}).get("content")]
        complete = len(choices) == 1 and (stop == "stop" or api in MUSE_APIS)
    text = texts[0] if len(texts) == 1 and isinstance(texts[0], str) else None
    if complete:
        target, error = parse(text, binary=api not in MUSE_APIS)
    else:
        target, error = None, f"finish_{stop}"
    return {"prediction": target, "error": error, "finish_reason": stop, "text": text,
            "usage": result.get("usage", result.get("usageMetadata")),
            "returned_model": result.get("model", result.get("modelVersion"))}
