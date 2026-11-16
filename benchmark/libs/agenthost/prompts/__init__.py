from pathlib import Path


class FilePromptLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, name: str) -> str:
        path = self.root / name
        if path.suffix == "":
            for suffix in (".prompt", ".md", ".txt"):
                candidate = path.with_suffix(suffix)
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8")
            path = path.with_suffix(".prompt")
        return path.read_text(encoding="utf-8")


class PromptManager:
    def __init__(self, loader: FilePromptLoader | None = None, prompts: dict[str, str] | None = None):
        self.loader = loader
        self.prompts = prompts or {}

    async def load_prompt(self, name: str) -> str:
        return self.get(name)

    def get(self, name: str) -> str:
        if name in self.prompts:
            return self.prompts[name]
        if self.loader is None:
            raise KeyError(name)
        return self.loader.load(name)
