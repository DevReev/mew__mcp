import asyncio
from fastmcp import Client

async def main():
    url = "http://localhost:8086/mcp/"
    auth_token = "bobbyistoocute2007"  # if needed
    async with Client(url, auth=auth_token) as client:
        tools = await client.list_tools()
        print("Available tools:", tools)
        result = await client.call_tool("greet", {"name": "FastMCP User"})
        print("Greet result:", result.data)

asyncio.run(main())
