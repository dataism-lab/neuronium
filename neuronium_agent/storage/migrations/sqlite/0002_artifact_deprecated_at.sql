-- B1 Part 2: logical deprecation of artifacts on rollback (§3.4.1)
ALTER TABLE artifacts ADD COLUMN deprecated_at TEXT NULL;
