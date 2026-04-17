curl -X POST "$URL/query" \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -d '{
    "query": "What is IBTIDA in simple words?",
    "k": 3,
    "filter": null
