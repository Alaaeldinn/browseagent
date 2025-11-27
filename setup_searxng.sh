#!/bin/bash
# Quick setup script for SearXNG using Docker

echo "Setting up SearXNG locally..."

# Pull and run SearXNG
docker run -d \
  --name searxng \
  -p 8080:8080 \
  -e SEARXNG_BASE_URL=http://localhost:8080 \
  searxng/searxng:latest

echo ""
echo "✓ SearXNG is starting on http://localhost:8080"
echo ""
echo "Wait 10-15 seconds, then update your .env file:"
echo "SEARXNG_URL=http://localhost:8080"
echo ""
echo "To stop: docker stop searxng"
echo "To restart: docker start searxng"
echo "To remove: docker rm -f searxng"
