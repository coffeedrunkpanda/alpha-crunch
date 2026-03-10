curl -X POST "$ALPHA_CRUNCH_URL/generate" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -d '{
    "message": "Explain EBITDA in simple words",
    "max_new_tokens": 120,
    "temperature": 0.2
  }'
