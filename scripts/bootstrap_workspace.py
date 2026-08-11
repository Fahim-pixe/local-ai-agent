"""Create the runtime workspace and initialize the local SQLite system of record."""

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository


def main() -> None:
    settings = load_settings()
    ensure_workspace(settings)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    print(f"Workspace ready at {settings.workspace_root}")
    print(f"SQLite database ready at {settings.sqlite_path}")


if __name__ == "__main__":
    main()
