import duckdb

con = duckdb.connect()

# Register all parquet files as views
views = {
    # Navigation
    "patient_navigation_map": "../generate/rp_synthetic_output/00_navigation/patient_navigation_map.parquet",

    # Facts
    "fact_charges":     "../generate/rp_synthetic_output/01_facts/charges.parquet",
    "fact_ringcentral": "../generate/rp_synthetic_output/01_facts/ringcentral.parquet",
    "fact_rv_inbound":  "../generate/rp_synthetic_output/01_facts/rv_inbound.parquet",
    "fact_rv_outbound": "../generate/rp_synthetic_output/01_facts/rv_outbound.parquet",
    "fact_statements":  "../generate/rp_synthetic_output/01_facts/statements.parquet",
    "fact_transactions":"../generate/rp_synthetic_output/01_facts/transactions.parquet",
    "fact_visits":      "../generate/rp_synthetic_output/01_facts/visits.parquet",

    # Dims
    "dim_birdeye":   "../generate/rp_synthetic_output/02_dims/birdeye.parquet",
    "dim_insurance": "../generate/rp_synthetic_output/02_dims/insurance.parquet",
    "dim_location":  "../generate/rp_synthetic_output/02_dims/location.parquet",
    "dim_patient":   "../generate/rp_synthetic_output/02_dims/patient.parquet",

    # Supplementary
    "sup_campaign_map":  "../generate/rp_synthetic_output/03_supplementary/campaign_map.parquet",
    "sup_phone_bridge":  "../generate/rp_synthetic_output/03_supplementary/phone_bridge.parquet",
}

for view_name, path in views.items():
    con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')")

# List all views with row counts and column counts
summary = con.execute("""
    SELECT
        table_name,
        table_type
    FROM information_schema.tables
    WHERE table_type = 'VIEW'
    ORDER BY table_name
""").df()

print("=== Registered Datasets ===")
print(summary.to_string(index=False))

# Show schema for each view
# print("\n=== Column Details ===")
# for view_name in views.keys():
#     print(f"\n── {view_name} ──")
#     schema = con.execute(f"DESCRIBE {view_name}").df()
#     print(schema[["column_name", "column_type"]].to_string(index=False))
con.execute("INSTALL ui FROM core;")
con.execute("LOAD ui;")
con.execute("CALL start_ui_server();")

print("🦆 DuckDB UI running at http://localhost:4213")
print("   All your views are visible in the left panel under 'memory'")
print("   Press Ctrl+C to stop.")

# Keep the process alive so UI stays up
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down.")