import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.research_agent import run_research
from agent.models import ResearchResult
from typing import Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPConnectionManager:
    """Manages connections to MCP servers"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.sessions: Dict[str, ClientSession] = {}
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        logger.warning(f"Config file {self.config_path} not found, using defaults")
        return {
            "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": "gemini-1.5-flash",
            "max_papers": 10,
            "servers": []
        }
    
    async def connect_to_servers(self) -> Dict[str, ClientSession]:
        """Connect to all enabled MCP servers"""
        servers = self.config.get("servers", [])
        
        for server_config in servers:
            if not server_config.get("enabled", True):
                continue
            
            name = server_config["name"]
            command = server_config["command"]
            args = server_config["args"]
            
            try:
                logger.info(f"Connecting to {name} server...")
                
                server_params = StdioServerParameters(
                    command=command,
                    args=args
                )
                
                read, write = await stdio_client(server_params).__aenter__()
                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()
                
                self.sessions[name] = session
                logger.info(f"✓ Connected to {name} server")
                
            except Exception as e:
                logger.error(f"✗ Failed to connect to {name} server: {e}")
        
        return self.sessions
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for name, session in self.sessions.items():
            try:
                await session.__aexit__(None, None, None)
                logger.info(f"Disconnected from {name} server")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
    
    def get_sessions(self) -> Dict[str, ClientSession]:
        """Get all connected sessions"""
        return self.sessions

async def main():
    """Main application entry point"""
    print("🔬 Smart Research Assistant")
    print("=" * 60)
    print()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not set")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        print()
    
    manager = MCPConnectionManager()
    
    try:
        sessions = await manager.connect_to_servers()
        
        if not sessions:
            print("❌ No MCP servers available. Exiting.")
            return
        
        print()
        print("Available servers:")
        for name in sessions.keys():
            print(f"  ✓ {name}")
        print()
        
        query = "transformer architectures in natural language processing"
        
        print(f"Researching: {query}")
        print("-" * 60)
        print()
        
        result = await run_research(
            query=query,
            mcp_sessions=sessions,
            user_id="demo_user",
            max_papers=5,
            include_web_search=True
        )
        
        print("=" * 60)
        print("RESEARCH RESULTS")
        print("=" * 60)
        print()
        
        print(f"Query: {result.query}")
        print()
        
        print("Summary:")
        print(result.summary)
        print()
        
        print(f"Found {len(result.papers)} Papers:")
        print("-" * 60)
        for i, paper in enumerate(result.papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors)}")
            if paper.year:
                print(f"   Year: {paper.year}")
            if paper.citations:
                print(f"   Citations: {paper.citations}")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:200]}...")
        
        print()
        print("Key Findings:")
        print("-" * 60)
        for i, finding in enumerate(result.key_findings, 1):
            print(f"{i}. {finding}")
        
        print()
        print("Recommended Reading Order:")
        print("-" * 60)
        for i, idx in enumerate(result.recommended_reading_order, 1):
            if idx < len(result.papers):
                paper = result.papers[idx]
                print(f"{i}. {paper.title}")
        
        if result.web_resources:
            print()
            print("Additional Web Resources:")
            print("-" * 60)
            for i, resource in enumerate(result.web_resources, 1):
                print(f"{i}. {resource.title}")
                print(f"   URL: {resource.url}")
                print(f"   {resource.snippet[:150]}...")
        
        print()
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    
    finally:
        print("\nCleaning up...")
        await manager.disconnect_all()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
