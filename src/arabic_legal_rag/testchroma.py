import chromadb

client = chromadb.PersistentClient(path="data/index")

collections = client.list_collections()

for collection in collections:
    print(f"\nCollection: {collection.name}")
    print(f"Count: {collection.count()}")

    data = collection.get(limit=5, include=["documents", "metadatas"])

    for i, doc in enumerate(data["documents"]):
        print(f"\n--- Document {i} ---")
        print(doc)

        if data["metadatas"]:
            print("Metadata:", data["metadatas"][i])
