from typing import Dict, Optional

_PATH_STORE: Dict[str, str] = {}

def set_node_path(node_id: Optional[str], path: Optional[str]) -> None:
    if not node_id:
        return

    if path:
        _PATH_STORE[node_id] = path
    else:
        _PATH_STORE.pop(node_id, None)

def get_node_path(node_id: Optional[str]) -> str:
    if not node_id:
        return ""

    return _PATH_STORE.get(node_id, "")
