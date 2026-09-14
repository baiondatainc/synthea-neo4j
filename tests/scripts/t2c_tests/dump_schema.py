# dump_schema.py  — run from t2c_tests/
import yaml
from harness import Graph, CFG
g = Graph(CFG)
nodes = {}
for r in g.run("CALL db.schema.nodeTypeProperties()"):
    for lbl in r["nodeLabels"]:
        nodes.setdefault(lbl, {})[r["propertyName"]] = r["propertyTypes"][0]
rels = {}
for r in g.run("CALL db.schema.relTypeProperties()"):
    t = r["relType"].strip(":`")
    rels.setdefault(t, {})
    if r["propertyName"]:
        rels[t][r["propertyName"]] = r["propertyTypes"][0]
paths = g.run("""MATCH (a)-[r]->(b) WITH labels(a)[0] AS s, type(r) AS t, labels(b)[0] AS e, count(*) AS n
                 RETURN s, t, e, n ORDER BY s, t, e""")
yaml.safe_dump({"nodes": nodes, "relationships": rels,
                "patterns": [f"({p['s']})-[:{p['t']}]->({p['e']})  # {p['n']}" for p in paths]},
               open("live_schema.yaml", "w"), sort_keys=False)
print(open("live_schema.yaml").read())