import sys
from pathlib import Path

def add_path(path, prepend=False):
    if isinstance(path, list):
        for p in path:
            add_path(p, prepend)
        return

    if isinstance(path, Path):
        path = path.resolve().as_posix()

    if path not in sys.path:
        if prepend:
            sys.path.insert(0, path)
        else:
            sys.path.append(path)

here = Path(__file__).parent.absolute()
comfy_dir = here.parent.parent

add_path(comfy_dir)
add_path((comfy_dir/"custom_nodes"))
