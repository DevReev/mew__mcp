import asyncio
import logging.config
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

import anyio
import httpx
import markdownify
import psutil
import readabilipy
import structlog
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from transformers import pipeline, set_seed
import functools
import inspect

# ───────────────────────────────
# 0. Environment & Configuration
# ───────────────────────────────
load_dotenv()


class ServerSettings(BaseSettings):
    auth_token: str
    my_number: str
    host: str = "0.0.0.0"
    port: int = 8086
    log_level: str = "INFO"
    model_cache_dir: str = "./model_cache"
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    rate_limit: int = 30  # requests
    rate_window: int = 60  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = ServerSettings()

# Ensure model cache directory exists
os.makedirs(settings.model_cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = settings.model_cache_dir
os.environ["HF_HOME"] = settings.model_cache_dir

# ───────────────────────────────
# 1. Structured Logging
# ───────────────────────────────
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "level": settings.log_level,
            }
        },
        "loggers": {"": {"handlers": ["console"], "level": settings.log_level}},
    }
)

logger = structlog.get_logger()

# ───────────────────────────────
# 2. Authentication Provider
# ───────────────────────────────
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self._token = token

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        if token == self._token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None


# ───────────────────────────────
# 3. Utility Classes
# ───────────────────────────────
class Fetch:
    USER_AGENT = "Puch/2.0 (Enhanced)"

    @classmethod
    async def fetch_url(cls, url: str, force_raw: bool = False) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            try:
                resp = await client.get(url, follow_redirects=True, headers={"User-Agent": cls.USER_AGENT})
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

        if resp.status_code >= 400:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {resp.status_code}"))

        content_type = resp.headers.get("content-type", "")
        is_html = "text/html" in content_type

        if is_html and not force_raw:
            return cls._extract_markdown(resp.text), ""
        return resp.text, f"Raw content (type: {content_type}) follows:\n"

    @staticmethod
    def _extract_markdown(html: str) -> str:
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "Could not simplify HTML"
        return markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)


# ───────────────────────────────
# 4. Rate Limiter
# ───────────────────────────────
class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self._requests: defaultdict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self._requests[client_id] = [
            t for t in self._requests[client_id] if now - t < self.window
        ]
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        self._requests[client_id].append(now)
        return True


rate_limiter = RateLimiter(settings.rate_limit, settings.rate_window)

# ───────────────────────────────
# 5. Error Handling
# ───────────────────────────────
class ErrorCategory(str, Enum):
    MODEL = "model_error"
    NETWORK = "network_error"
    VALIDATION = "validation_error"
    SYSTEM = "system_error"


class ErrorHandler:
    async def handle(self, err: Exception, context: str) -> str:
        error_id = f"err_{int(time.time())}"
        logger.error(
            "UnhandledError",
            error_id=error_id,
            context=context,
            err=str(err),
            traceback=structlog.processors.format_exc_info(err),
        )
        return f"❌ **Error {error_id}**: Something went wrong while processing your request."


error_handler = ErrorHandler()

# ───────────────────────────────
# 6. Retry Decorator
# ───────────────────────────────
def with_retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))
        # overwrite wrapper’s signature to match fn’s signature
        wrapper.__signature__ = sig
        return wrapper
    return decorator

# ───────────────────────────────
# 7. Model Manager (Caching)
# ───────────────────────────────
class ModelManager:
    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        self._models: dict[str, pipeline] = {}
        set_seed(42)  # deterministic output

    async def get(self, model_name: str) -> pipeline:
        if model_name in self._models:
            return self._models[model_name]

        def _load():
            logger.info("LoadingModel", model=model_name, cache_dir=self._cache_dir)
            return pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                cache_dir=self._cache_dir,
                device_map="auto",
                torch_dtype="auto",
                pad_token_id=50256,
            )

        model = await anyio.to_thread.run_sync(_load)
        self._models[model_name] = model
        return model


model_manager = ModelManager(settings.model_cache_dir)

# ───────────────────────────────
# 8. FastMCP Server Setup
# ───────────────────────────────
mcp = FastMCP(
    "Shah Rukh Khan Pickup Lines & Date Locations MCP Server (Enhanced)",
    auth=SimpleBearerAuthProvider(settings.auth_token),
)

start_time = time.time()

# ───────────────────────────────
# 9. Helper Functions
# ───────────────────────────────
def _clean_output(text: str, prompt: str) -> str:
    return (
        text.replace(prompt, "")
        .replace("Pickup line:", "")
        .replace("SRK says:", "")
        .strip()
    )


def _srk_fallback(user_info: str) -> str:
    lines = [
        f"Kuch kuch hota hai, {user_info}, every time I see your smile.",
        f"If love had a face, {user_info}, it would look just like you.",
        f"Main hoon na, {user_info}—to hold you when the world lets go.",
    ]
    return random.choice(lines)


# ───────────────────────────────
# 10. Tools
# ───────────────────────────────
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: Optional[str] = None


@mcp.tool
async def validate() -> str:
    return settings.my_number


PickupLineDescription = RichToolDescription(
    description="Generate romantic pickup lines in Shah Rukh Khan style.",
    use_when="User wants a charming SRK pickup line.",
    side_effects="Returns stylised romantic text.",
)

# @with_retry()
@mcp.tool(description=PickupLineDescription.model_dump_json()) 
async def generate_srk_pickup_line(
    user_info: Annotated[str, Field(description="Information about the user (name, interests)")] = "friend",
    target_info: Annotated[str | None, Field(description="Person to impress")] = None,
) -> str:
    """
    Generate an SRK-style pickup line using cached local models,
    falling back to curated lines when necessary.
    """
    try:
        context = f"User: {user_info}"
        if target_info:
            context += f" | Target: {target_info}"
        prompt = (
            "Generate a romantic, witty pickup line in Shah Rukh Khan's voice.\n"
            f"Context: {context}\n"
            "- Use Bollywood flair and Hindi phrases.\n"
            "Pickup line:"
        )

        model = await model_manager.get("microsoft/DialoGPT-small")
        generated = await anyio.to_thread.run_sync(
            lambda: model(prompt, max_length=len(prompt.split()) + 40, temperature=0.8, top_p=0.9)[0][
                "generated_text"
            ]
        )
        logger.info("GeneratedPickupLine", prompt=prompt, generated=generated)
        line = _clean_output(generated, prompt)
        if len(line) < 8:
            raise ValueError("Empty generation")
        return f"💕 **SRK Pickup Line**\n\n*\"{line}\"*"
    except Exception as e:
        logger.warning("FallbackPickupLine", err=str(e))
        return f"💕 **SRK Pickup Line**\n\n*\"{_srk_fallback(user_info)}\"*"
# generate_srk_pickup_line = with_retry()(generate_srk_pickup_line)

DateLocationDescription = RichToolDescription(
    description="Find romantic date venues in a given city.",
    use_when="User needs date locations.",
    side_effects="Performs web searches.",
)


@mcp.tool(description=DateLocationDescription.model_dump_json())
  
async def find_date_locations(
    city: Annotated[str, Field(description="City to search")] = "Mumbai",
    date_type: Annotated[str, Field(description="Type of date")] = "romantic dinner",
    budget: Annotated[str, Field(description="Budget level")] = "moderate",
) -> str:
    """Search DuckDuckGo for date spots and return formatted list."""
    try:
        query = f"best {date_type} {city} {budget}"
        url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": Fetch.USER_AGENT})
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = [
            (
                r.find("a", class_="result__a").get_text(strip=True),
                r.find("a", class_="result__snippet").get_text(strip=True),
                r.find("a", class_="result__a")["href"],
            )
            for r in soup.find_all("div", class_="result__body")[:8]
            if r.find("a", class_="result__a") and r.find("a", class_="result__snippet")
        ]
        if not results:
            return "❌ No specific locations found—try adjusting your query."

        formatted = f"💝 **Date Spots in {city.title()}**\n\n"
        for i, (name, desc, link) in enumerate(results, 1):
            formatted += f"**{i}. {name}**\n{desc}\n🔗 {link}\n\n"
        return formatted
    except Exception as e:
        return await error_handler.handle(e, "find_date_locations")

# find_date_locations = with_retry()(find_date_locations)    


FlirtyReplyDescription = RichToolDescription(
    description="Generate SRK-style flirty reply to a message.",
    use_when="User wants SRK reply.",
    side_effects="Returns stylised reply.",
)


@mcp.tool(description=FlirtyReplyDescription.model_dump_json())

async def generate_srk_flirty_reply(
    message: Annotated[str, Field(description="Message to reply to")],
    your_name: Annotated[str | None, Field(description="Your name")] = None,
) -> str:
    """Generate an SRK-style flirty response to `message`."""
    try:
        prompt = (
            "You are Shah Rukh Khan. Craft a flirty, witty reply in Hindi-English.\n"
            f"Incoming message: \"{message}\"\n"
            "Reply:"
        )
        model = await model_manager.get("microsoft/DialoGPT-small")
        generated = await anyio.to_thread.run_sync(
            lambda: model(prompt, max_length=len(prompt.split()) + 30, temperature=0.8, top_p=0.9)[0][
                "generated_text"
            ]
        )
        reply = _clean_output(generated, prompt)
        if len(reply) < 5:
            raise ValueError("Empty generation")
        return f"💕 **SRK Reply**\n\n*\"{reply}\"*"
    except Exception as e:
        logger.warning("FallbackFlirtyReply", err=str(e))
        fallback = f"Kuch kuch hota hai jab tum baat karti ho—couldn't resist replying, {your_name or 'jaan'}!"
        return f"💕 **SRK Reply**\n\n*\"{fallback}\"*"
# generate_srk_flirty_reply = with_retry()(generate_srk_flirty_reply)

class OutfitInspoDescription(BaseModel):
    description: str = "Suggest SRK-inspired outfit ideas for dates and special occasions."
    use_when: str = "User wants wardrobe inspiration in a Bollywood style."
    side_effects: Optional[str] = None

OutfitInspoMetadata = OutfitInspoDescription()

@mcp.tool(description=OutfitInspoMetadata.model_dump_json())

async def outfit_inspiration(
    occasion: Annotated[str, Field(description="Type of occasion (e.g., romantic dinner, casual day out)")] = "romantic dinner",
    budget: Annotated[str, Field(description="Budget level (low, moderate, high)")] = "moderate",
    color_palette: Annotated[Optional[str], Field(description="Preferred colors (e.g., pastels, jewel tones)")] = None,
) -> str:
    """
    Provide outfit inspiration inspired by Shah Rukh Khan's style.
    """
    try:
        # Example static suggestions; replace with API or database lookup as needed.
        inspo = {
            "romantic dinner": {
                "low": [
                    "White linen shirt + dark jeans + brown loafers",
                    "Soft pink kurta + denim jacket + minimal accessories"
                ],
                "moderate": [
                    "Fitted black blazer + charcoal trousers + white tee",
                    "Burgundy silk shirt + black chinos + loafers"
                ],
                "high": [
                    "Tailored velvet tuxedo + silk lapel + patent leather shoes",
                    "Custom Nehru jacket + silk trousers + statement watch"
                ]
            },
            "casual day out": {
                "low": ["Graphic tee + ripped jeans + sneakers", "Striped polo + khaki shorts + canvas shoes"],
                "moderate": ["Denim jacket + white shirt + chinos", "Linen shirt + slim-fit joggers + slip-ons"],
                "high": ["Designer bomber jacket + tapered jeans + leather sneakers", "Cashmere sweater + tailored joggers + loafers"]
            }
        }

        options = inspo.get(occasion.lower(), inspo["romantic dinner"])
        choices = options.get(budget.lower(), options["moderate"])
        if color_palette:
            # Append color cue
            choices = [f"{c}, in {color_palette} hues" for c in choices]

        formatted = f"👔 **Outfit Inspiration** for a {occasion.title()} (Budget: {budget.title()})\n\n"
        for i, suggestion in enumerate(choices, 1):
            formatted += f"**{i}.** {suggestion}\n\n"
        return formatted

    except Exception as e:
        return await error_handler.handle(e, "outfit_inspiration")
# outfit_inspiration = with_retry()(outfit_inspiration)
# ───────────────────────────────
# 11. Health Check Tool
# ───────────────────────────────
@mcp.tool
async def health_check() -> str:
    uptime = time.time() - start_time
    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
    cpu_pct = psutil.cpu_percent()
    cached_models = len(model_manager._models)
    return (
        f"🟢 **Server Healthy**\n\n"
        f"Uptime: {uptime:.1f}s\nMemory: {mem_mb:.1f} MB\nCPU: {cpu_pct}%\n"
        f"Cached models: {cached_models}"
    )


# ───────────────────────────────
# 12. Global Request Interceptor
# ───────────────────────────────
# @mcp.middleware
# async def apply_rate_limit(request, call_next):
#     client_id = request.headers.get("X-Client-ID", "anonymous")
#     if not rate_limiter.is_allowed(client_id):
#         return "❌ Rate limit exceeded. Please try again later."
#     return await call_next(request)


# ───────────────────────────────
# 13. Run Server
# ───────────────────────────────
async def main():
    logger.info("ServerStarting", host=settings.host, port=settings.port)
    await mcp.run_async("streamable-http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    asyncio.run(main())
