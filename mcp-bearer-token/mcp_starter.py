import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
import markdownify
import httpx
import readabilipy
import json
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text
            content_type = response.headers.get("content-type", "")
            is_page_html = "text/html" in content_type

            if is_page_html and not force_raw:
                return cls.extract_content_from_html(page_raw), ""

            return (
                page_raw,
                f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
            )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "Page failed to be simplified from HTML"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

# --- MCP Server Setup ---
mcp = FastMCP(
    "Shah Rukh Khan Pickup Lines & Date Locations MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool 1: Generate Shah Rukh Khan-style Pick Up Line ---
PickupLineDescription = RichToolDescription(
    description="Generate  pickup lines using user information and context.",
    use_when="Use this when user wants a charming pickup line or wants to start a conversation with a girl, especially when they ask something like 'What do I say to her?'",
    side_effects="Generates a personalized pickup line based on user context and SRK's romantic dialogue style.",
)

@mcp.tool(description=PickupLineDescription.model_dump_json())
# @mcp.tool(description=PickupLineDescription.model_dump_json())
async def generate_srk_pickup_line(
    user_info: Annotated[str, Field(description="Information about the user or context (e.g., name, interests, setting, mood)")],
    target_info: Annotated[str | None, Field(description="Information about the person they want to impress")] = None,
) -> str:
    """
    Generate a Shah Rukh Khan-style pickup line using local Hugging Face models.
    """
    
    # Create the prompt for generating SRK-style pickup lines
    context = f"User context: {user_info}"
    if target_info:
        context += f"\nTarget person: {target_info}"
    
    prompt = f"""Generate a charming, romantic pickup line in Shah Rukh Khan's style.

Context: {context}

The pickup line should be:
- Charming and romantic like SRK's famous dialogues
- Clever and witty
- Appropriate and respectful
- Include a touch of Bollywood flair

Generate only the pickup line:"""

    try:
        # Import transformers for local model usage
        from transformers import pipeline, set_seed
        import torch
        
        # Set random seed for reproducible results
        set_seed(42)
        
        # Try different models in order of preference (smaller models first for better performance)
        models_to_try = [
            "microsoft/DialoGPT-small",  # Lightweight conversational model
            "gpt2",                      # Classic text generation
            "distilgpt2",               # Even lighter version of GPT-2
        ]
        
        for model_name in models_to_try:
            try:
                # Create text generation pipeline with local model
                generator = pipeline(
                    'text-generation',
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                    pad_token_id=50256,  # Set pad token to avoid warnings
                )
                
                # Generate text with the model
                result = generator(
                    prompt,
                    max_length=len(prompt.split()) + 50,  # Limit output length
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=generator.tokenizer.eos_token_id,
                )
                
                # Extract generated text and clean it up
                generated_text = result[0]['generated_text']
                pickup_line = generated_text.replace(prompt, "").strip()
                logger.info("GeneratedPickupLine", prompt=prompt, generated=pickup_line)
                # If we got a good result, return it
                if pickup_line and len(pickup_line) > 10:
                    # Add user context to make it more personalized
                    if user_info and user_info.lower() not in pickup_line.lower():
                        pickup_line = pickup_line.replace("you", user_info if len(user_info.split()) <= 2 else "you")
                    logger.info("FinalPickupLine", pickup_line=pickup_line)
                    return f"💕 **Shah Rukh Khan-style Pickup Line** 💕\n\n*\"{pickup_line}\"*\n\n✨ *Generated locally with SRK's signature charm!*\n\n🤖 *Model used: {model_name}*"
                
            except Exception as model_error:
                # Try next model if this one fails
                continue
        
        # If all models fail, raise an exception to use fallback
        print("All local models failed")
        raise Exception("All local models failed")
        
    except Exception as e:
        # Enhanced fallback with more SRK-style lines
        fallback_lines = [
            f"Just like in my movies, {user_info}, you've made my heart skip a beat. Kuch kuch hota hai when I see you!",
            f"If I were to write a love story, {user_info}, you would be both the beginning and the happy ending.",
            f"They say love is friendship plus something more, {user_info}. Want to find out what that 'something more' is?",
            f"In all my films, I've said 'I love you' in a hundred ways, but seeing you, {user_info}, I'm speechless for the first time.",
            f"You know {user_info}, I've spread my arms wide in many movies, but I'd do it for real if you'd run into them.",
            f"Like Raj from DDLJ, {user_info}, I'd travel the world just to win your heart.",
            f"Main hoon na, {user_info}? Just like SRK, I promise to always be there for you.",
            f"Kabhi Khushi Kabhie Gham... but with you {user_info}, it's always khushi!",
            f"Baazigar ho tum, {user_info}, because you've won my heart without even trying.",
            f"Kal Ho Naa Ho, {user_info}, but today I know I want to spend it with you."
        ]
        import random
        selected_line = random.choice(fallback_lines)
        print(f"Using fallback line: {selected_line}")
        return f"💕 **Shah Rukh Khan-style Pickup Line** 💕\n\n*\"{selected_line}\"*\n\n✨ *Delivered with SRK's signature charm and romance!*\n\n_(Generated using curated SRK-style fallback lines)_"

# --- Tool 2: Find Date Locations ---
DateLocationDescription = RichToolDescription(
    description="Find romantic date locations and venues based on user preferences, city, and date type.",
    use_when="Use this when user wants to find places for a romantic date or outing.",
    side_effects="Returns a list of suitable date locations with details like address, type, and atmosphere.",
)

@mcp.tool(description=DateLocationDescription.model_dump_json())
async def find_date_locations(
    city: Annotated[str, Field(description="City or location where to find date spots")],
    date_type: Annotated[str, Field(description="Type of date (e.g., romantic dinner, coffee, outdoor, adventure, cultural)")] = "romantic",
    budget: Annotated[str, Field(description="Budget preference (budget-friendly, moderate, upscale)")] = "moderate",
) -> str:
    """
    Find suitable date locations based on preferences using location APIs and web search.
    """
    
    try:
        # Search for date locations using web search
        search_query = f"best {date_type} date spots {city} {budget} restaurants cafes"
        
        async with httpx.AsyncClient() as client:
            # Using DuckDuckGo search for date locations
            ddg_url = f"https://html.duckduckgo.com/html/?q={search_query.replace(' ', '+')}"
            
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            
            if resp.status_code != 200:
                return f"❌ Failed to search for date locations in {city}"
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            
            locations = []
            for i, result in enumerate(soup.find_all("div", class_="result__body")):
                if i >= 8:  # Limit to 8 results
                    break
                    
                title_elem = result.find("a", class_="result__a")
                snippet_elem = result.find("a", class_="result__snippet")
                
                if title_elem and snippet_elem:
                    title = title_elem.get_text().strip()
                    snippet = snippet_elem.get_text().strip()
                    url = title_elem.get("href", "")
                    
                    locations.append({
                        "name": title,
                        "description": snippet,
                        "url": url
                    })
            
            # Format the response
            response = f"💝 **Date Locations in {city}** 💝\n\n"
            response += f"**Date Type:** {date_type.title()}\n"
            response += f"**Budget:** {budget.title()}\n\n"
            
            if not locations:
                response += "❌ No specific locations found. Here are some general suggestions:\n\n"
                response += f"**For {date_type} dates in {city}:**\n"
                response += "• Local restaurants with good ambiance\n"
                response += "• Coffee shops or cafes\n"
                response += "• Parks or scenic spots\n"
                response += "• Museums or cultural centers\n"
                response += "• Entertainment venues\n"
                return response
            
            for i, location in enumerate(locations, 1):
                response += f"**{i}. {location['name']}**\n"
                response += f"   {location['description'][:200]}{'...' if len(location['description']) > 200 else ''}\n"
                if location['url'] and 'http' in location['url']:
                    response += f"   🔗 [More Info]({location['url']})\n"
                response += "\n"
            
            response += "💡 **Pro Tips:**\n"
            response += "• Check opening hours and make reservations if needed\n"
            response += "• Consider the weather for outdoor locations\n"
            response += "• Read recent reviews before visiting\n"
            response += "• Plan backup options in case of unexpected changes\n"
            
            return response
            
    except Exception as e:
        return f"❌ **Error finding date locations:** {str(e)}\n\n💡 **General suggestions for {date_type} dates in {city}:**\n• Romantic restaurants\n• Cozy cafes\n• Scenic parks\n• Local attractions\n• Entertainment venues"

# --- Tool 3: Generate SRK-Style Flirty Reply to Message ---
FlirtyReplyDescription = RichToolDescription(
    description="Generate flirty and charming replies to received messages in Shah Rukh Khan's romantic style.",
    use_when="Use this when user wants to respond to a message with SRK's signature charm, romance, and Bollywood flair.",
    side_effects="Generates playful, romantic responses inspired by Shah Rukh Khan's iconic dialogues and romantic persona.",
)

@mcp.tool(description=FlirtyReplyDescription.model_dump_json())
async def generate_srk_flirty_reply(
    message: Annotated[str, Field(description="The message you received that you want to reply to in SRK style")],
    your_name: Annotated[str | None, Field(description="Your name (optional, for personalization)")] = None,
    context: Annotated[str | None, Field(description="Additional context about your relationship or conversation history")] = None,
) -> str:
    """
    Generate flirty, charming replies in Shah Rukh Khan's romantic style using local Hugging Face models.
    """
    
    # Create context for SRK-style flirty response
    context_info = f"Message to reply to: {message}"
    if your_name:
        context_info += f"\nYour name: {your_name}"
    if context:
        context_info += f"\nContext: {context}"
    
    prompt = f"""You are Shah Rukh Khan, the King of Bollywood romance. Generate a flirty, charming reply to this message in your signature romantic style.

{context_info}

The reply should be:
- Flirty and playfully romantic like SRK's famous dialogues
- Charming and witty with Bollywood flair
- Engaging and conversation-continuing
- Appropriate but with romantic undertones
- Sound like something SRK would say in his romantic movies
- Include a touch of Hindi/Urdu if appropriate

Generate only the SRK-style flirty reply:"""

    try:
        # Import transformers for local model usage
        from transformers import pipeline, set_seed
        import torch
        
        # Set random seed for reproducible results
        set_seed(42)
        
        # Try different models in order of preference
        models_to_try = [
            "microsoft/DialoGPT-small",  # Good for conversational responses
            "gpt2",                      # Classic text generation
            "distilgpt2",               # Lightweight option
        ]
        
        for model_name in models_to_try:
            try:
                # Create text generation pipeline with local model
                generator = pipeline(
                    'text-generation',
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                    pad_token_id=50256,  # Set pad token to avoid warnings
                )
                
                # Generate text with the model
                result = generator(
                    prompt,
                    max_length=len(prompt.split()) + 40,  # Limit output length for concise replies
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=generator.tokenizer.eos_token_id,
                )
                
                # Extract generated text and clean it up
                generated_text = result[0]['generated_text']
                srk_reply = generated_text.replace(prompt, "").strip()
                
                # If we got a good result, return it
                if srk_reply and len(srk_reply) > 8:
                    return f"💕 **SRK-Style Flirty Reply** 💕\n\n**Original Message:** \"{message}\"\n\n**Your Shah Rukh Khan Response:**\n*\"{srk_reply}\"*\n\n✨ *Delivered with SRK's signature charm and Bollywood romance!*\n\n🤖 *Generated using: {model_name}*"
                
            except Exception as model_error:
                # Try next model if this one fails
                continue
        
        # If all models fail, raise an exception to use fallback
        raise Exception("All local models failed")
        
    except Exception as e:
        # Generate contextual SRK-style flirty fallback based on message content
        srk_reply = generate_srk_flirty_fallback(message, your_name, context)
        
        return f"💕 **SRK-Style Flirty Reply** 💕\n\n**Original Message:** \"{message}\"\n\n**Your Shah Rukh Khan Response:**\n*\"{srk_reply}\"*\n\n✨ *Delivered with SRK's signature charm and Bollywood romance!*\n\n_(Generated using curated SRK-style responses)_"

def generate_srk_flirty_fallback(message: str, your_name: str = None, context: str = None) -> str:
    """Generate contextual SRK-style flirty fallback replies based on message analysis."""
    
    message_lower = message.lower()
    name_to_use = your_name if your_name else "beautiful"
    
    # Analyze message type and generate appropriate SRK-style flirty responses
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
        # SRK-style flirty greetings
        responses = [
            f"Namaste {name_to_use}! Just like in my movies, you've made my heart do a little dance 💃",
            f"Hello gorgeous! Seeing your message is like the best scene in any of my films 🎬",
            f"Well hello there! You know, I've spread my arms wide in many movies, but I'd do it for real just for you 😉"
        ]
    
    elif any(word in message_lower for word in ['how are you', 'what\'s up', 'how\'s it going']):
        # SRK-style check-ins
        responses = [
            f"Much better now that I'm talking to you, {name_to_use}! Kuch kuch hota hai when I see your messages 💕",
            f"I'm wonderful, but like in DDLJ, everything's perfect when you're around 🚂",
            f"Great! But you know what would make it better? If we were having this conversation over coffee, Bollywood style ☕"
        ]
    
    elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
        # SRK-style thanks responses
        responses = [
            f"Anything for you, {name_to_use}! Like I always say in my films - main hoon na! 🤗",
            f"My pleasure, gorgeous! You know I'd move mountains for you, just like any good Bollywood hero 🏔️",
            f"Don't mention it, beautiful! Making you happy is my favorite role to play 🎭"
        ]
    
    elif any(word in message_lower for word in ['miss', 'thinking', 'wish']):
        # SRK-style emotional/longing responses
        responses = [
            f"That's so sweet, {name_to_use}! Like in Kal Ho Naa Ho, every moment with you is precious 💎",
            f"Aww, you're making me feel like the luckiest hero in Bollywood! The feeling is definitely mutual 🎬❤️",
            f"You always know how to make my heart skip like a perfect movie scene, {name_to_use} 💕"
        ]
    
    elif any(word in message_lower for word in ['want to', 'let\'s', 'would you like', 'free', 'available']):
        # SRK-style invitation responses
        responses = [
            f"With you, {name_to_use}? I'm always ready for the adventure! Like Raj in DDLJ, I'd follow you anywhere 🚂",
            f"That sounds amazing! You know how to plan the perfect romantic scene, just like a Bollywood director 🎬",
            f"Count me in, gorgeous! Spending time with you is better than any movie script I've ever read 📝"
        ]
    
    elif any(word in message_lower for word in ['beautiful', 'gorgeous', 'pretty', 'hot', 'cute', 'amazing']):
        # SRK-style compliment responses
        responses = [
            f"Stop it, {name_to_use}! You're making me blush more than in my romantic scenes! But please, keep going... 😉",
            f"You're such a charmer! If love had a definition, it would be you making me feel this special 💕",
            f"Coming from someone as incredible as you, that means everything! You're my real-life co-star ⭐"
        ]
    
    elif any(word in message_lower for word in ['sorry', 'apologize', 'my bad']):
        # SRK-style apology responses
        responses = [
            f"Don't you worry about it, {name_to_use}! Like in my movies, true love means never having to say sorry 💕",
            f"How could I stay upset with someone so adorable? Consider it forgotten, beautiful 😘",
            f"You're too precious when you apologize! All is forgiven, just like in every Bollywood love story 🎬"
        ]
    
    elif '?' in message:
        # SRK-style question responses
        responses = [
            f"Great question, {name_to_use}! I love how your brilliant mind works - it's one of your many charms 🧠💕",
            f"Hmm, let me think about that while I admire how thoughtful you are, gorgeous 🤔",
            f"You always ask the most interesting questions! It's like you're writing the perfect script for our story 📖"
        ]
    
    else:
        # General SRK-style flirty responses
        responses = [
            f"You always know just what to say to make me smile, {name_to_use}! It's like you have the perfect dialogue for every scene 😊",
            f"Every message from you is like getting the best role in Bollywood! You seriously just made my whole day 🎬",
            f"You're absolutely irresistible, you know that? Like the perfect Bollywood heroine 😉",
            f"Is it just me, or are you getting more charming by the day? You're like a real-life movie magic ✨",
            f"If I were to write a love story, {name_to_use}, every chapter would be about moments like these 📚💕",
            f"Kuch kuch hota hai when I talk to you - and it's always something wonderful! 💫"
        ]
    
    # Pick a random response
    import random
    selected_response = random.choice(responses)
    
    return selected_response

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    port = int(os.environ.get("PORT", 8086))
    await mcp.run_async("streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    asyncio.run(main())
