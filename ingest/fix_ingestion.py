import re

path = "/home/siva/work/codebase/RP/synthea-neo4j/ingest/ingestion.py"

with open(path, 'r') as f:
    c = f.read()

# ── Fix 1: UNWIND → WHERE without WITH ──────────────────────────────────────
# Replace all: UNWIND $rows AS row\n<whitespace>WHERE
# With:        UNWIND $rows AS row\n<whitespace>WITH row\n<whitespace>WHERE
pattern = r'(UNWIND \$rows AS row\n)([ \t]+)(WHERE)'
replacement = r'\1\2WITH row\n\2\3'
c_new = re.sub(pattern, replacement, c)
fixed1 = len(re.findall(pattern, c))
c = c_new

# ── Fix 2: Remove Cypher // comments between WITH and WHERE ─────────────────
pattern2 = r'([ \t]+WITH [^\n]+\n)([ \t]+//[^\n]+\n)([ \t]+WHERE)'
replacement2 = r'\1\3'
c_new2 = re.sub(pattern2, replacement2, c)
fixed2 = len(re.findall(pattern2, c))
c = c_new2

print(f"Fixed {fixed1} UNWIND→WHERE issues")
print(f"Fixed {fixed2} comment-between-WITH-WHERE issues")

with open(path, 'w') as f:
    f.write(c)
print("Done. File saved.")