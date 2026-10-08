"""Schema migrations for databases created before a Chainlit upgrade.

Covers `ui/persistence.py`: `CREATE TABLE IF NOT EXISTS` never alters an
existing table, so columns newer Chainlit releases write (e.g. `autoCollapse`)
are added by `ensure_sqlite_schema` on startup.
"""

from __future__ import annotations

import sqlite3

import ui.persistence as persistence


def _step_columns(path) -> set:
    with sqlite3.connect(path) as con:
        return {row[1] for row in con.execute('PRAGMA table_info("steps")')}


class TestStepsMigrations:
    def test_fresh_schema_includes_autoCollapse(self, tmp_path):
        db = tmp_path / "fresh.db"
        persistence.ensure_sqlite_schema(db)
        assert "autoCollapse" in _step_columns(db)

    def test_existing_db_gets_the_missing_column(self, tmp_path):
        db = tmp_path / "old.db"
        persistence.ensure_sqlite_schema(db)
        with sqlite3.connect(db) as con:
            con.execute('ALTER TABLE steps DROP COLUMN "autoCollapse"')

        persistence.ensure_sqlite_schema(db)

        assert "autoCollapse" in _step_columns(db)

    def test_migration_is_idempotent_and_writable(self, tmp_path):
        db = tmp_path / "old.db"
        persistence.ensure_sqlite_schema(db)
        with sqlite3.connect(db) as con:
            con.execute('ALTER TABLE steps DROP COLUMN "autoCollapse"')
        persistence.ensure_sqlite_schema(db)
        persistence.ensure_sqlite_schema(db)

        with sqlite3.connect(db) as con:
            con.execute(
                'INSERT INTO steps ("id", "name", "type", "threadId", '
                '"streaming", "autoCollapse") VALUES (?, ?, ?, ?, ?, ?)',
                ("s1", "paso", "run", "t1", 0, 0),
            )
