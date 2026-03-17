import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

from backend.services.search_service import search_service

print(f"Vector Enabled: {search_service._vector_enabled}")
print(f"Collection: {search_service.collection}")

query = "fight"
results = search_service.search(query)
print(f"Search results for '{query}': {results}")

query = "general description"
results = search_service.search(query)
print(f"Search results for '{query}': {results}")
