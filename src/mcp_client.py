import json
from typing import Dict, List, Optional, Any


class MCPClient:
    """MCP (Model Context Protocol) client for external tool integration"""

    def __init__(self):
        self.tools: Dict[str, Dict] = {}

    def register_tool(self, name: str, description: str, parameters: Dict):
        self.tools[name] = {"description": description, "parameters": parameters}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        if tool_name not in self.tools:
            return None
        return json.dumps({"tool": tool_name, "args": arguments, "result": "mock_result"})

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
