from benchmark.libs.agenthost.app.config import AppConfigManager


def register_section_extension(section: str, name: str, schema: object) -> None:
    AppConfigManager._section_extensions[(section, name)] = schema


__all__ = ["register_section_extension"]
