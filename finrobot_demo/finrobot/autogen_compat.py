# Minimal compatibility shim to provide 'autogen' API used by FinRobot on top of autogen_agentchat/autogen_core >=0.7
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent  # type: ignore
from autogen_agentchat.base import Agent as BaseAgent  # type: ignore
from autogen_agentchat.teams import RoundRobinGroupChat as GroupChat  # closest equivalent
from autogen_agentchat.teams import BaseGroupChat as GroupChatManager  # placeholder manager
from autogen_agentchat.tools import AgentTool as register_function  # not identical, but placeholder to avoid import errors
from autogen_core.cache_store import CacheStore as Cache  # expose Cache-like name

# Fallback types
Agent = BaseAgent

# Provide config_list_from_json similar helper
import json
from typing import Any, Dict, List

def config_list_from_json(path: str, filter_dict: Dict[str, List[str]] | None = None) -> List[Dict[str, Any]]:
	with open(path, 'r', encoding='utf-8') as f:
		cfg = json.load(f)
	if filter_dict is None:
		return cfg
	def match(item: Dict[str, Any]) -> bool:
		for k, vals in filter_dict.items():
			if str(item.get(k)) not in vals:
				return False
		return True
	return [c for c in cfg if match(c)]