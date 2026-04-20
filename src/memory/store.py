from datetime import datetime
from pathlib import Path
from typing import Dict
import json


class NoteStore:
    def __init__(self, notes_dir: Path) -> None:
        self.notes_dir = notes_dir
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def save_note(self, note_id: str, title: str, body: str, metadata: Dict[str, str]) -> Path:
        filename = f"{note_id}.md"
        path = self.notes_dir / filename
        frontmatter = {
            "id": note_id,
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
        }
        payload = f"---\n{json.dumps(frontmatter, ensure_ascii=False, indent=2)}\n---\n\n{body}\n"
        path.write_text(payload, encoding="utf-8")
        return path

