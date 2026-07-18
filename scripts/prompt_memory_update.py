#!/usr/bin/env python3
"""
prompt-memory MCP SQLite store updater.

The MCP server exposes store_memory/tag_node/relate_nodes but not update.
This script updates name/content/label/properties/tags directly in the
prompt-memory SQLite DB. FTS5 triggers keep the search index in sync.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path.home() / ".cache" / "prompt-memory-mcp" / "store.db"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if env := os.environ.get("PROMPT_MEMORY_DB"):
        return Path(env)
    return DEFAULT_DB


def ensure_tag_ids(cur: sqlite3.Cursor, names: list[str]) -> list[int]:
    ids: list[int] = []
    for name in names:
        cur.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
        cur.execute("SELECT id FROM tags WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Failed to resolve tag id for {name!r}")
        ids.append(row[0])
    return ids


def get_node_id(cur: sqlite3.Cursor, uuid: str | None, node_id: int | None) -> int:
    if uuid:
        cur.execute("SELECT id FROM nodes WHERE uuid = ?", (uuid,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"Node with uuid {uuid!r} not found")
        return row[0]
    if node_id:
        cur.execute("SELECT id FROM nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"Node with id {node_id} not found")
        return row[0]
    raise SystemExit("Specify --uuid or --id")


def load_properties(cur: sqlite3.Cursor, node_id: int) -> dict[str, Any]:
    cur.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,))
    row = cur.fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row[0]) if row[0] else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Existing properties are not valid JSON: {exc}")


def current_tags(cur: sqlite3.Cursor, node_id: int) -> set[str]:
    cur.execute(
        """
        SELECT t.name FROM tags t
        JOIN node_tags nt ON nt.tag_id = t.id
        WHERE nt.node_id = ?
        """,
        (node_id,),
    )
    return {row[0] for row in cur.fetchall()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Update a prompt-memory node in the SQLite store."
    )
    parser.add_argument("--db", help="Path to prompt-memory store.db")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--uuid", help="Node uuid to update")
    group.add_argument("--id", type=int, help="Node integer id to update")

    parser.add_argument("--name", help="New node name")
    parser.add_argument("--text", help="New node content/text")
    parser.add_argument("--label", help="New node label")
    parser.add_argument(
        "--properties", help="JSON object merged into existing properties"
    )
    parser.add_argument(
        "--replace-properties",
        action="store_true",
        help="Replace properties entirely instead of merging",
    )

    parser.add_argument("--tags", help="Comma-separated tags (replaces existing)")
    parser.add_argument("--add-tags", help="Comma-separated tags to add")
    parser.add_argument("--remove-tags", help="Comma-separated tags to remove")
    parser.add_argument("--dry-run", action="store_true", help="Print, do not write")
    args = parser.parse_args(argv)

    db_path = get_db_path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    updates: dict[str, Any] = {}
    if args.name is not None:
        updates["name"] = args.name
    if args.text is not None:
        updates["content"] = args.text
    if args.label is not None:
        updates["label"] = args.label

    new_properties: dict[str, Any] | None = None
    if args.properties:
        try:
            new_properties = json.loads(args.properties)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--properties is not valid JSON: {exc}")
        if not isinstance(new_properties, dict):
            raise SystemExit("--properties must be a JSON object")

    tag_mode: str | None = None
    tag_list: list[str] = []
    if args.tags is not None:
        tag_mode = "replace"
        tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    elif args.add_tags or args.remove_tags:
        tag_mode = "delta"
        tag_list = [t.strip() for t in (args.add_tags or "").split(",") if t.strip()]
        remove_list = [t.strip() for t in (args.remove_tags or "").split(",") if t.strip()]

    if not updates and new_properties is None and tag_mode is None:
        print("Nothing to update.", file=sys.stderr)
        return 1

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()
        node_id = get_node_id(cur, args.uuid, args.id)

        cur.execute("SELECT uuid, name, label, content, properties FROM nodes WHERE id = ?", (node_id,))
        old = cur.fetchone()
        if old is None:
            raise SystemExit("Node disappeared")

        if new_properties is not None:
            if args.replace_properties:
                updates["properties"] = json.dumps(new_properties)
            else:
                merged = load_properties(cur, node_id)
                merged.update(new_properties)
                updates["properties"] = json.dumps(merged)

        if updates:
            updates["updated_at"] = now_iso()
            fields = ", ".join(f"{k} = ?" for k in updates)
            sql = f"UPDATE nodes SET {fields} WHERE id = ?"
            cur.execute(sql, (*updates.values(), node_id))

        if tag_mode == "replace":
            cur.execute("DELETE FROM node_tags WHERE node_id = ?", (node_id,))
            if tag_list:
                tag_ids = ensure_tag_ids(cur, tag_list)
                cur.executemany(
                    "INSERT OR IGNORE INTO node_tags (node_id, tag_id) VALUES (?, ?)",
                    [(node_id, tid) for tid in tag_ids],
                )
        elif tag_mode == "delta":
            current = current_tags(cur, node_id)
            current.update(tag_list)
            current.difference_update(remove_list)
            cur.execute("DELETE FROM node_tags WHERE node_id = ?", (node_id,))
            if current:
                tag_ids = ensure_tag_ids(cur, sorted(current))
                cur.executemany(
                    "INSERT OR IGNORE INTO node_tags (node_id, tag_id) VALUES (?, ?)",
                    [(node_id, tid) for tid in tag_ids],
                )

        if args.dry_run:
            print("DRY RUN — no changes written")
            print(f"Node id={node_id} uuid={old[0]}")
            print("Fields to update:", {k: v for k, v in updates.items() if k != "updated_at"})
            if tag_mode:
                print(f"Tag mode: {tag_mode}, tags: {tag_list if tag_mode == 'replace' else current}")
            conn.rollback()
            return 0

        conn.commit()

        cur.execute("SELECT uuid, name, label, content, properties, updated_at FROM nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        print(f"Updated node id={node_id} uuid={row[0]}")
        print(f"  name: {row[1]}")
        print(f"  label: {row[2]}")
        print(f"  content chars: {len(row[3])}")
        print(f"  properties: {row[4]}")
        print(f"  updated_at: {row[5]}")

        cur.execute(
            """
            SELECT t.name FROM tags t
            JOIN node_tags nt ON nt.tag_id = t.id
            WHERE nt.node_id = ? ORDER BY t.name
            """,
            (node_id,),
        )
        tags = [r[0] for r in cur.fetchall()]
        print(f"  tags: {', '.join(tags) if tags else '(none)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
