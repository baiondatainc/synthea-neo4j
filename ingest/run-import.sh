#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# SGS — RP Knowledge Graph
# Neo4j Docker Import Script
# ──────────────────────────────────────────────────────────────────
# Usage:
#   bash run_import.sh                    # default paths
#   bash run_import.sh /path/to/csvs      # custom CSV directory
# ──────────────────────────────────────────────────────────────────

set -e  # exit on any error

CONTAINER="neo4j_rp"
COMPOSE_FILE="$HOME/work/codebase/RP/synthea-neo4j/dockers/docker-compose.yml"
CSV_DIR="${1:-$HOME/work/codebase/RP/synthea-neo4j/generator/neo4j_import}"

IMPORT_MOUNT="/var/lib/neo4j/import"   # path inside container
DB_NAME="neo4j"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "════════════════════════════════════════════════════════"
echo "  SGS — RP Knowledge Graph"
echo "  Neo4j Admin Import"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Verify CSV directory ────────────────────────────────────
info "Step 1/6: Checking CSV directory..."
[ -d "$CSV_DIR" ] || error "CSV directory not found: $CSV_DIR"

NODE_COUNT=$(ls "$CSV_DIR"/nodes_*.csv 2>/dev/null | wc -l)
REL_COUNT=$(ls "$CSV_DIR"/rel_*.csv 2>/dev/null | wc -l)
CSV_SIZE=$(du -sh "$CSV_DIR"/*.csv 2>/dev/null | tail -1 | cut -f1)

[ "$NODE_COUNT" -gt 0 ] || error "No node CSV files found in $CSV_DIR"
[ "$REL_COUNT"  -gt 0 ] || error "No relationship CSV files found in $CSV_DIR"
success "Found $NODE_COUNT node CSVs + $REL_COUNT relationship CSVs in $CSV_DIR"

# ── Step 2: Copy CSVs to import folder ──────────────────────────────
info "Step 2/6: Copying CSVs to Docker import folder..."
DOCKER_IMPORT_DIR="$HOME/work/codebase/RP/synthea-neo4j/dockers/import"
mkdir -p "$DOCKER_IMPORT_DIR"
cp "$CSV_DIR"/*.csv "$DOCKER_IMPORT_DIR"/
success "Copied $(ls "$DOCKER_IMPORT_DIR"/*.csv | wc -l) CSV files to $DOCKER_IMPORT_DIR"

# ── Step 3: Stop Neo4j ───────────────────────────────────────────────
info "Step 3/6: Stopping Neo4j container..."
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    docker stop "$CONTAINER"
    success "Container stopped"
else
    warn "Container $CONTAINER was not running"
fi

# ── Step 4: Clear Neo4j data volume ────────────────────────────────
info "Step 4/6: Clearing Neo4j data volume..."
echo ""
echo -e "${YELLOW}  WARNING: This will DELETE all existing graph data.${NC}"
read -p "  Type 'yes' to confirm: " CONFIRM
echo ""
[ "$CONFIRM" = "yes" ] || error "Aborted by user"

# Remove and recreate the data volume
docker volume rm neo4j_data 2>/dev/null && \
    success "Removed volume neo4j_data" || \
    warn "Volume neo4j_data did not exist (fresh install)"

docker volume create neo4j_data
success "Created fresh neo4j_data volume"

# ── Step 5: Run neo4j-admin import ──────────────────────────────────
info "Step 5/6: Running neo4j-admin import..."
echo "  This may take 20-30 minutes for 10GB production data."
echo "  Grab a coffee ☕"
echo ""

START_TIME=$(date +%s)

docker run --rm \
  --volume neo4j_data:/data \
  --volume "$DOCKER_IMPORT_DIR":/import \
  neo4j:5.26-community \
  neo4j-admin database import full "$DB_NAME" \
    --nodes=Patient=/import/nodes_patient.csv \
    --nodes=Practice=/import/nodes_practice.csv \
    --nodes=Location=/import/nodes_location.csv \
    --nodes=InsurancePlan=/import/nodes_insurance.csv \
    --nodes=Campaign=/import/nodes_campaign.csv \
    --nodes=BirdeyeReview=/import/nodes_birdeye.csv \
    --nodes=Visit=/import/nodes_visit.csv \
    --nodes=Charge=/import/nodes_charge.csv \
    --nodes=Transaction=/import/nodes_transaction.csv \
    --nodes=Statement=/import/nodes_statement.csv \
    --nodes=RCCall=/import/nodes_rccall.csv \
    --nodes=IVRInbound=/import/nodes_ivrinbound.csv \
    --nodes=DiallerCall=/import/nodes_diallercall.csv \
    --nodes=PhoneBridge=/import/nodes_phonebridge.csv \
    --nodes=DiagnosisCode=/import/nodes_diagnosiscode.csv \
    --nodes=ProcedureCode=/import/nodes_procedurecode.csv \
    --relationships=REGISTERED_AT=/import/rel_patient_practice.csv \
    --relationships=BELONGS_TO_PRACTICE=/import/rel_location_practice.csv \
    --relationships=ISSUED_BY_PRACTICE=/import/rel_insurance_practice.csv \
    --relationships=RUN_BY=/import/rel_campaign_practice.csv \
    --relationships=REVIEWS=/import/rel_birdeye_location.csv \
    --relationships=HAD_VISIT=/import/rel_patient_visit.csv \
    --relationships=PERFORMED_AT=/import/rel_visit_location.csv \
    --relationships=UNDER_PLAN=/import/rel_visit_insurance.csv \
    --relationships=HAS_CHARGE=/import/rel_patient_charge.csv \
    --relationships=PART_OF_VISIT=/import/rel_charge_visit.csv \
    --relationships=AT_LOCATION=/import/rel_charge_location.csv \
    --relationships=DIAGNOSED_WITH=/import/rel_charge_diagnosis.csv \
    --relationships=USES_PROCEDURE=/import/rel_charge_procedure.csv \
    --relationships=SETTLES=/import/rel_transaction_charge.csv \
    --relationships=HAS_TRANSACTION=/import/rel_patient_transaction.csv \
    --relationships=RECEIVED_STATEMENT=/import/rel_patient_statement.csv \
    --relationships=PART_OF_CAMPAIGN=/import/rel_rccall_campaign.csv \
    --relationships=ATTRIBUTED_TO_PHONE=/import/rel_rccall_phonebridge.csv \
    --relationships=IDENTIFIED_BY_PHONE=/import/rel_patient_phonebridge.csv \
    --relationships=CALLED_IVR=/import/rel_patient_ivr.csv \
    --relationships=CONTACTED_BY_DIALLER=/import/rel_patient_dialler.csv \
    --skip-bad-relationships=true \
    --skip-duplicate-nodes=true \
    --high-parallel-io=on

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
success "Import completed in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"

# ── Step 6: Start Neo4j and verify ──────────────────────────────────
info "Step 6/6: Starting Neo4j..."
docker compose -f "$COMPOSE_FILE" up -d

echo ""
info "Waiting for Neo4j to be ready..."
for i in $(seq 1 30); do
    if docker exec "$CONTAINER" cypher-shell \
        -u neo4j -p rp_strong_pass_2025 \
        "RETURN 1 AS ping" >/dev/null 2>&1; then
        success "Neo4j is ready!"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# Quick node count
echo ""
info "Graph summary:"
docker exec "$CONTAINER" cypher-shell \
    -u neo4j -p rp_strong_pass_2025 \
    "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC" \
    2>/dev/null || warn "Could not query graph yet — try again in 30 seconds"

echo ""
echo "════════════════════════════════════════════════════════"
echo -e "  ${GREEN}Import complete!${NC}"
echo ""
echo "  Next steps:"
echo "  1. Create indexes:"
echo "     cd ~/work/codebase/RP/synthea-neo4j && source .venv/bin/activate"
echo "     python main.py schema"
echo ""
echo "  2. Verify in browser:  http://localhost:7474"
echo "     Login: neo4j / rp_strong_pass_2025"
echo ""
echo "  3. Test QA chain:"
echo "     python main.py ask 'Which patients have the highest outstanding balance?'"
echo "════════════════════════════════════════════════════════"