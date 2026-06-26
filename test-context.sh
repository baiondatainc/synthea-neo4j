#!/bin/bash
# test-context.sh — clean version

echo "=== Clear cache ==="
curl -s -X POST http://localhost:8001/cache/clear

echo ""
echo "=== Turn 1 ==="
curl -s -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me unpaid invoices for RADM:1000042", "conversation_id": "test-001", "use_cache": false}' \
  | python3 -m json.tool

echo ""
echo "=== Redis after turn 1 ==="
redis-cli keys "chat:*"
redis-cli keys "focus:*"
redis-cli lrange "chat:test-001" 0 -1

echo ""
echo "=== Turn 2 ==="
curl -s -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "…. from Complete Care Brooks City", "conversation_id": "test-001", "use_cache": false}' \
  | python3 -m json.tool

echo ""
echo "=== Redis after turn 2 ==="
redis-cli lrange "chat:test-001" 0 -1