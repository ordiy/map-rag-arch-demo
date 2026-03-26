import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from src.rag_core import process_document_task, run_evaluation, run_chat_pipeline, collection, db_lance

# We need to wipe the DB first for a clean test
collection.delete_many({})
if "knowledge_base" in db_lance.table_names():
    db_lance.drop_table("knowledge_base")

test_docs = [
    {
        "filename": "project_artemis.txt",
        "content": "The Artemis Project is a highly confidential initiative started by the CEO in 2023. \n\n Its primary goal is to establish self-sustaining underwater habitats for scientific research. \n\n The total budget allocated for it is exactly 50 billion dollars, approved last month."
    },
    {
        "filename": "hr_policy_2026.txt",
        "content": "The new 2026 Remote Work Policy has been finalized. \n\n Employees in the engineering department are allowed to work from home 4 days a week. \n\n However, they must be present in the office every Wednesday for the all-hands sync."
    },
    {
        "filename": "product_x_launch.txt",
        "content": "Product X, our next-generation smart glasses, is scheduled for release in Q4. \n\n It features an AR display and bone-conduction audio. \n\n The target demographic for this release is primarily outdoor sports enthusiasts and cyclists."
    }
]

async def run_test():
    print("1. Ingesting Documents (Chunking & Small-to-Big mapping)...")
    for doc in test_docs:
        state = {}
        await process_document_task(doc["content"], doc["filename"], state)
        print(f"Ingested {doc['filename']}: {state}")

    print("\n2. Checking MongoDB Structure...")
    parents = list(collection.find({"type": "parent"}))
    children = list(collection.find({"type": "child"}))
    print(f"Found {len(parents)} Parent Docs and {len(children)} Child Chunks.")

    print("\n3. Testing Small-to-Big Retrieval & Evaluation...")
    # Query: "What is the budget for the Artemis Project?" 
    # Expected substring: "50 billion"
    # Chunk 3 ("The total budget allocated for it...") has the answer but lacks the name "Artemis".
    # With Small-to-Big, the whole parent doc is fetched and reranked.
    
    eval_queries = [
        {"q": "What is the budget for the Artemis Project?", "expected": "50 billion"},
        {"q": "Which day must engineering employees be in the office?", "expected": "Wednesday"},
        {"q": "Who is the target audience for Product X?", "expected": "cyclists"}
    ]
    
    for eq in eval_queries:
        print(f"\nEvaluating Query: '{eq['q']}'")
        res = run_evaluation(eq["q"], eq["expected"])
        print(f"Hit Rank: {res['hit_rank']}")
        for k, metrics in res['metrics'].items():
            print(f"  {k}: Recall={metrics['Recall']}, NDCG={metrics['NDCG']}")
            
        print("Top Retrieved Document Context (First 150 chars):")
        print(res["top_3_docs"][0][:150] + "...")

    print("\n4. Testing End-to-End Chat API...")
    chat_res = run_chat_pipeline("What is the budget for the Artemis Project?")
    print("Final Answer:")
    print(chat_res["answer"])
    print("\nExecution Logs:")
    for log in chat_res["logs"]:
        print(f" - {log}")

if __name__ == "__main__":
    asyncio.run(run_test())
