#!/bin/bash
# fix_titles_api.sh
# ─────────────────
# Renames all "New Chat" conversations via LibreChat's REST API.
# Requires: curl, jq
# Usage: ./fix_titles_api.sh
#
# Get your token by opening LibreChat → F12 → Application → LocalStorage
# → look for the key that contains your JWT token

LIBRECHAT_URL="http://localhost:3080"
TOKEN="YOUR_JWT_TOKEN_HERE"   # paste your Bearer token here

echo "=== Fetching all conversations ==="
PAGE=1
TOTAL_UPDATED=0

while true; do
  RESP=$(curl -s \
    -H "Authorization: Bearer $TOKEN" \
    "$LIBRECHAT_URL/api/convos?pageNumber=$PAGE&pageSize=50&isArchived=false")

  CONVOS=$(echo "$RESP" | jq -c '.conversations[]')
  COUNT=$(echo "$RESP" | jq '.conversations | length')
  PAGES=$(echo "$RESP" | jq '.pages')

  if [ -z "$CONVOS" ] || [ "$COUNT" -eq 0 ]; then
    break
  fi

  echo "Page $PAGE / $PAGES — $COUNT conversations"

  while IFS= read -r convo; do
    CONV_ID=$(echo "$convo" | jq -r '.conversationId')
    TITLE=$(echo "$convo" | jq -r '.title')

    if [ "$TITLE" != "New Chat" ] && [ -n "$TITLE" ]; then
      continue
    fi

    # Fetch the first user message
    MSGS=$(curl -s \
      -H "Authorization: Bearer $TOKEN" \
      "$LIBRECHAT_URL/api/messages/$CONV_ID")

    FIRST_MSG=$(echo "$MSGS" | jq -r '[.[] | select(.role=="user")] | sort_by(.createdAt) | first | .text // .content // ""' 2>/dev/null | head -c 60)

    if [ -z "$FIRST_MSG" ]; then
      echo "  [$CONV_ID] — no user message, skipping"
      continue
    fi

    NEW_TITLE=$(echo "$FIRST_MSG" | sed 's/^[[:space:]….]*//' | head -c 60)

    echo "  [$CONV_ID] '$TITLE' → '$NEW_TITLE'"

    curl -s -X POST \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"arg\": {\"conversationId\": \"$CONV_ID\", \"title\": \"$NEW_TITLE\"}}" \
      "$LIBRECHAT_URL/api/convos/update" > /dev/null

    TOTAL_UPDATED=$((TOTAL_UPDATED + 1))
    sleep 0.1   # be gentle with the API
  done <<< "$CONVOS"

  if [ "$PAGE" -ge "$PAGES" ]; then
    break
  fi
  PAGE=$((PAGE + 1))
done

echo ""
echo "=== Done. Updated $TOTAL_UPDATED conversations ==="