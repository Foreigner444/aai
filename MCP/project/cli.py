import asyncio
import os
import sys
from main import MCPConnectionManager
from agent.research_agent import run_research
import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchCLI:
    """Interactive CLI for the research assistant"""
    
    def __init__(self):
        self.manager = MCPConnectionManager()
        self.sessions = None
        self.running = True
        self.history = []
    
    async def start(self):
        """Start the interactive CLI"""
        print("🔬 Smart Research Assistant - Interactive CLI")
        print("=" * 60)
        print()
        
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️  Warning: GEMINI_API_KEY not set")
            print("The agent may not work without it.")
            print("Set it with: export GEMINI_API_KEY='your-key-here'")
            print()
        
        self.sessions = await self.manager.connect_to_servers()
        
        if not self.sessions:
            print("❌ No MCP servers available. Exiting.")
            return
        
        print("Connected to servers:")
        for name in self.sessions.keys():
            print(f"  ✓ {name}")
        print()
        
        print("Commands:")
        print("  research <query>  - Research a topic")
        print("  notes             - List saved notes")
        print("  history           - Show query history")
        print("  help              - Show this help")
        print("  exit              - Exit the application")
        print()
        
        while self.running:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                await self.handle_command(user_input)
                
            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error handling command: {e}", exc_info=True)
                print(f"❌ Error: {e}")
        
        await self.cleanup()
    
    async def handle_command(self, command: str):
        """Handle user commands"""
        if command == "exit" or command == "quit":
            self.running = False
            print("Goodbye!")
        
        elif command == "help":
            self.show_help()
        
        elif command.startswith("research "):
            query = command[9:].strip()
            if query:
                await self.do_research(query)
            else:
                print("❌ Please provide a research query")
        
        elif command == "notes":
            await self.list_notes()
        
        elif command == "history":
            self.show_history()
        
        else:
            print(f"❌ Unknown command: '{command}'")
            print("Type 'help' for available commands")
    
    async def do_research(self, query: str):
        """Perform research on a topic"""
        print(f"\n🔍 Researching: {query}")
        print("-" * 60)
        print("Please wait...\n")
        
        try:
            result = await run_research(
                query=query,
                mcp_sessions=self.sessions,
                user_id="cli_user",
                max_papers=5,
                include_web_search=True
            )
            
            self.history.append(query)
            self.display_results(result)
            
        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            print(f"❌ Research failed: {e}")
    
    def display_results(self, result):
        """Display research results"""
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print()
        
        print("📊 Summary:")
        print(result.summary)
        print()
        
        print(f"📄 Found {len(result.papers)} papers:")
        print("-" * 60)
        for i, paper in enumerate(result.papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   By: {', '.join(paper.authors)}")
            if paper.year:
                print(f"   Year: {paper.year}")
            if paper.citations:
                print(f"   Citations: {paper.citations}")
            print(f"   URL: {paper.url}")
        
        print()
        print("💡 Key Findings:")
        print("-" * 60)
        for i, finding in enumerate(result.key_findings, 1):
            print(f"{i}. {finding}")
        
        print()
        print("📚 Recommended Reading Order:")
        print("-" * 60)
        for i, idx in enumerate(result.recommended_reading_order, 1):
            if idx < len(result.papers):
                paper = result.papers[idx]
                print(f"{i}. {paper.title}")
        
        print()
        print("=" * 60)
        print()
    
    async def list_notes(self):
        """List all saved research notes"""
        notes_session = self.sessions.get('notes')
        if not notes_session:
            print("❌ Notes server not available")
            return
        
        try:
            result = await notes_session.call_tool("list_all_notes", {})
            print()
            print(result.content[0].text)
            print()
        except Exception as e:
            logger.error(f"Failed to list notes: {e}", exc_info=True)
            print(f"❌ Failed to list notes: {e}")
    
    def show_history(self):
        """Show query history"""
        if not self.history:
            print("\nNo research history yet.\n")
            return
        
        print()
        print("Query History:")
        print("-" * 60)
        for i, query in enumerate(self.history, 1):
            print(f"{i}. {query}")
        print()
    
    def show_help(self):
        """Show help message"""
        print()
        print("Available Commands:")
        print("-" * 60)
        print("  research <query>  - Research a topic using academic papers and web search")
        print("  notes             - List all saved research notes")
        print("  history           - Show your research query history")
        print("  help              - Show this help message")
        print("  exit              - Exit the application")
        print()
        print("Examples:")
        print("  >>> research machine learning optimization")
        print("  >>> research BERT architecture")
        print("  >>> notes")
        print()
    
    async def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        await self.manager.disconnect_all()
        print("Done!")

def main():
    """CLI entry point"""
    try:
        cli = ResearchCLI()
        asyncio.run(cli.start())
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
