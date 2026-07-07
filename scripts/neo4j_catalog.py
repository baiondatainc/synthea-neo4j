"""
Neo4j Data Model → YAML Catalog Exporter
Requires: pip install neo4j pyyaml
"""
import yaml
from datetime import datetime
from neo4j import GraphDatabase


# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "rp_strong_pass_2025"
OUTPUT_FILE    = "data_catalog.yaml"
# ─────────────────────────────────────────────────────────────────────────────

"""
Neo4j Data Model → YAML Catalog Exporter
Requires: pip install neo4j pyyaml
"""

def get_schema_apoc(session):
    """Try APOC meta.schema — returns dict or None."""
    try:
        result = session.run("CALL apoc.meta.schema() YIELD value RETURN value")
        record = result.single()
        if record and isinstance(record["value"], dict):
            return record["value"]
    except Exception as e:
        print(f"⚠️  APOC not available ({e}), falling back to basic introspection.")
    return None


def get_schema_basic(session):
    """Fallback: stitch schema from built-in db.* procedures."""
    labels = [r["label"] for r in session.run("CALL db.labels() YIELD label")]
    rel_types = [r["relationshipType"] for r in session.run("CALL db.relationshipTypes() YIELD relationshipType")]

    nodes = {}
    for label in labels:
        sample = session.run(
            f"MATCH (n:`{label}`) RETURN keys(n) AS props LIMIT 1"
        ).single()
        props = {p: {"type": "unknown"} for p in (sample["props"] if sample else [])}
        count = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()["c"]

        rels = []
        for r in session.run(
            f"MATCH (n:`{label}`)-[r]->(m) RETURN DISTINCT type(r) AS t, labels(m) AS ml, count(r) AS c"
        ):
            rels.append({"type": r["t"], "direction": "out",
                         "target": r["ml"][0] if r["ml"] else "unknown", "count": r["c"]})

        nodes[label] = {"count": count, "labels": [label], "properties": props, "relationships": rels}

    relationships = {}
    for rt in rel_types:
        count = session.run(f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c").single()["c"]
        sample = session.run(
            f"MATCH ()-[r:`{rt}`]->() RETURN keys(r) AS props LIMIT 1"
        ).single()
        props = {p: {"type": "unknown"} for p in (sample["props"] if sample else [])}
        relationships[rt] = {"count": count, "properties": props}

    return {"nodes": nodes, "relationships": relationships}


def parse_apoc_schema(raw: dict) -> dict:
    """Convert APOC apoc.meta.schema() output into catalog structure.
    Handles both formats:
      - relationships as list of dicts  (newer APOC)
      - relationships as list of strings (older APOC)
    """
    nodes = {}
    relationships = {}

    for name, meta in raw.items():
        if not isinstance(meta, dict):
            continue
        entity_type = meta.get("type", "node")

        if entity_type == "node":
            # Parse properties
            properties = {}
            for prop, info in (meta.get("properties") or {}).items():
                if isinstance(info, dict):
                    properties[prop] = {
                        "type":      info.get("type", "unknown"),
                        "indexed":   info.get("indexed", False),
                        "unique":    info.get("unique", False),
                        "existence": info.get("existence", False),
                    }
                else:
                    properties[prop] = {"type": str(info)}

            # Parse relationships — handle str OR dict items
            rels = []
            for r in (meta.get("relationships") or []):
                if isinstance(r, str):
                    # Older APOC: just the relationship type name
                    rels.append({"type": r, "direction": "out", "target": "unknown", "count": 0})
                elif isinstance(r, dict):
                    target_labels = r.get("labels") or []
                    rels.append({
                        "type":      r.get("type", "unknown"),
                        "direction": r.get("direction", "out"),
                        "target":    target_labels[0] if target_labels else "unknown",
                        "count":     r.get("count", 0),
                    })

            nodes[name] = {
                "count":         meta.get("count", 0),
                "labels":        meta.get("labels", [name]),
                "properties":    properties,
                "relationships": rels,
            }

        elif entity_type == "relationship":
            properties = {}
            for prop, info in (meta.get("properties") or {}).items():
                if isinstance(info, dict):
                    properties[prop] = {"type": info.get("type", "unknown")}
                else:
                    properties[prop] = {"type": str(info)}
            relationships[name] = {
                "count":      meta.get("count", 0),
                "properties": properties,
            }

    return {"nodes": nodes, "relationships": relationships}


def build_catalog(schema: dict, db_uri: str) -> dict:
    return {
        "catalog": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": db_uri,
            "version": "1.0",
        },
        "nodes": schema["nodes"],
        "relationships": schema["relationships"],
    }


def export_to_yaml(catalog: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(catalog, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print(f"✅  Catalog written to {path}")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        print("🔍  Fetching schema from Neo4j...")
        with driver.session() as session:
            raw = get_schema_apoc(session)
            if raw is not None:
                print("   Using APOC meta.schema")
                schema = parse_apoc_schema(raw)
            else:
                print("   Using basic introspection")
                schema = get_schema_basic(session)

        catalog = build_catalog(schema, NEO4J_URI)
        export_to_yaml(catalog, OUTPUT_FILE)

        print(f"   Nodes found        : {len(schema['nodes'])}")
        print(f"   Relationship types : {len(schema['relationships'])}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()