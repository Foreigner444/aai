import asyncio
import json
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
import mcp.server.stdio
from datetime import datetime

server = Server("papers-server")

mock_papers_db = [
    {
        "id": 1,
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        "url": "https://arxiv.org/abs/1706.03762",
        "year": 2017,
        "citations": 50000,
        "keywords": ["transformer", "attention", "neural networks", "nlp"]
    },
    {
        "id": 2,
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin et al."],
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.",
        "url": "https://arxiv.org/abs/1810.04805",
        "year": 2018,
        "citations": 40000,
        "keywords": ["bert", "transformer", "pretraining", "nlp"]
    },
    {
        "id": 3,
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Brown et al."],
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
        "url": "https://arxiv.org/abs/2005.14165",
        "year": 2020,
        "citations": 30000,
        "keywords": ["gpt", "few-shot", "language model", "scaling"]
    }
]

@server.list_resources()
async def list_resources() -> list[Resource]:
    resources = []
    for paper in mock_papers_db:
        resource = Resource(
            uri=f"paper:///{paper['id']}",
            name=f"{paper['title']} ({paper['year']})",
            description=f"By {', '.join(paper['authors'])} - {paper['citations']} citations",
            mimeType="application/json"
        )
        resources.append(resource)
    return resources

@server.read_resource()
async def read_resource(uri: str) -> str:
    if not uri.startswith("paper:///"):
        raise ValueError(f"Invalid URI: {uri}")
    
    paper_id = int(uri.replace("paper:///", ""))
    paper = next((p for p in mock_papers_db if p['id'] == paper_id), None)
    
    if not paper:
        raise ValueError(f"Paper {paper_id} not found")
    
    return json.dumps(paper, indent=2)

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_papers",
            description="Search for academic papers by keywords. Returns papers with title, authors, abstract, and URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query keywords"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_paper_details",
            description="Get detailed information about a specific paper by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "integer",
                        "description": "Paper ID"
                    }
                },
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="get_citations",
            description="Get citation count and related papers for a given paper",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "integer",
                        "description": "Paper ID"
                    }
                },
                "required": ["paper_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_papers":
        query = arguments["query"].lower()
        max_results = arguments.get("max_results", 5)
        
        results = []
        for paper in mock_papers_db:
            if (query in paper["title"].lower() or
                query in paper["abstract"].lower() or
                any(query in keyword for keyword in paper["keywords"])):
                results.append({
                    "id": paper["id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "abstract": paper["abstract"],
                    "url": paper["url"],
                    "year": paper["year"],
                    "citations": paper["citations"]
                })
            
            if len(results) >= max_results:
                break
        
        if not results:
            results = mock_papers_db[:max_results]
        
        return [TextContent(type="text", text=json.dumps(results, indent=2))]
    
    elif name == "get_paper_details":
        paper_id = arguments["paper_id"]
        paper = next((p for p in mock_papers_db if p['id'] == paper_id), None)
        
        if not paper:
            return [TextContent(type="text", text=json.dumps({"error": "Paper not found"}))]
        
        return [TextContent(type="text", text=json.dumps(paper, indent=2))]
    
    elif name == "get_citations":
        paper_id = arguments["paper_id"]
        paper = next((p for p in mock_papers_db if p['id'] == paper_id), None)
        
        if not paper:
            return [TextContent(type="text", text=json.dumps({"error": "Paper not found"}))]
        
        citation_info = {
            "paper_id": paper_id,
            "title": paper["title"],
            "citation_count": paper["citations"],
            "related_papers": [p["id"] for p in mock_papers_db if p["id"] != paper_id][:3]
        }
        
        return [TextContent(type="text", text=json.dumps(citation_info, indent=2))]
    
    return [TextContent(type="text", text=json.dumps({"error": "Unknown tool"}))]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
