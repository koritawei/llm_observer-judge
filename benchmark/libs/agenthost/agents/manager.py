class AgentManager:
    def __init__(self, registry: dict | None = None):
        self.registry = registry or {}

    def register(self, name: str, agent) -> None:
        self.registry[name] = agent

    def list(self) -> list[str]:
        return list(self.registry)

    def get(self, name: str):
        return self.registry[name]

    async def call(self, name: str, request, **kwargs):
        agent = self.get(name)
        return await agent.call(request, **kwargs)
