# 企业级 RAG 关联映射架构：Small-to-Big Retrieval (父子文档映射)

在高度依赖上下文碎片的 RAG 系统中，仅检索孤立的细粒度 Chunk 会导致严重的“上下文丢失”和“指代不明”（例如，Chunk 中出现“该项目”，但项目名称在上一段）。

为了解决这一问题，MAP-RAG 架构引入了 **“Small-to-Big Retrieval”（从小到大检索）** 机制。该机制通过解耦向量库和文档库的职责，利用 `Parent-Child` 映射实现精准定位与大局观判断的统一。

## 1. 知识库构建阶段 (Ingestion)

**核心原则：向量库只存“子”（细粒度，为搜得准）；文档库保存“父与族谱”（全粒度，为提供完整上下文）。**

### 1.1 文档库 (MongoDB) 的树状/链状结构
存入 MongoDB 时，摒弃扁平化结构，转而保存完整的关联族谱：

```json
// MongoDB: Parent Document (父文档 / 原文全篇)
{
    "doc_id": "doc_100",
    "type": "parent",
    "title": "2024年公司战略规划",
    "full_content": "2024年战略规划：全面投入 Agentic 架构研发... (1000字)",
    "child_chunk_ids": ["chunk_100_1", "chunk_100_2", "chunk_100_3"]
}

// MongoDB: Child Document (子文档 / 细粒度段落)
{
    "doc_id": "chunk_100_2",
    "type": "child",
    "parent_id": "doc_100",
    "prev_chunk_id": "chunk_100_1", // 链表：关联上一段
    "next_chunk_id": "chunk_100_3", // 链表：关联下一段
    "content": "特别是 MAP-RAG 系统，该系统将..."
}
```

### 1.2 向量库 (LanceDB) 的元数据桥梁
仅将极细粒度的 **Child Chunk** 进行 Embedding 处理并入库。**关键在于将 `parent_id` 作为寻根钥匙注入 Metadata：**

```python
# LanceDB 插入逻辑
Document(
    page_content="特别是 MAP-RAG 系统，该系统将...", 
    metadata={
        "chunk_id": "chunk_100_2", 
        "parent_id": "doc_100"  # <--- 映射核心桥梁
    }
)
```

---

## 2. 召回与重排阶段 (Retrieval & Reranking)

在用户发起查询（Query）时，系统将经过“精确定位 -> 上下文膨胀 -> 语义重排”三步流转。

### 步骤一：向量粗筛 (Vector Search - The Microscope)
LanceDB 充当“显微镜”，通过余弦相似度大海捞针，命中细微的子片段特征。
* **返回结果**：`[Document(content="特别是 MAP-RAG...", metadata={"parent_id": "doc_100"})]`

### 步骤二：文档库“回表”与“上下文膨胀” (Context Expansion)
拿到 `parent_id` 后，**坚决拦截碎片 Chunk 直接进入重排环节！**
系统必须携带 `parent_id` 返回 MongoDB 进行精确查询，提取关联上下文。

```python
retrieved_chunks = lancedb_retriever.invoke(query)
expanded_docs = []

for chunk in retrieved_chunks:
    parent_id = chunk.metadata["parent_id"]
    # 从 MongoDB 拉取完整的父级全景
    parent_record = mongo_collection.find_one({"doc_id": parent_id})
    
    # 策略 A：直接用完整的 Parent 覆盖碎片 Chunk
    full_context = parent_record["full_content"]
    
    # 策略 B：结构化拼装（赋予局部特征以全局坐标）
    # full_context = f"[所属源文档: {parent_record['title']}] \n {chunk.page_content}"
    
    expanded_docs.append(full_context)

# 聚合去重：多个命中的相邻 Child Chunk 可能归属于同一个 Parent
unique_expanded_docs = list(set(expanded_docs))
```

### 步骤三：交叉编码精排 (Cross-Encoder Rerank - The Referee)
经过上下文膨胀，送入 Jina Reranker 的不再是无头无尾的断句，而是包含全局语境的实体。
由于 Cross-Encoder （如基于 BERT 架构的重排器）高度依赖完整的句法结构和注意力机制矩阵，喂入带有完整关联信息的父文档或扩展段落，将使 Reranker 准确捕获“代词所指”与“真实意图”，从而实现排序精度（NDCG）的飞跃。

## 3. 架构优势总结
1. **向量显微镜 (LanceDB)**：解决长文本被高维压缩后特征模糊的问题，利用 Child Chunk 确保检索下限（高召回）。
2. **文档知识图谱 (MongoDB)**：负责存储空间映射，解决碎片化弊端，动态还原长上下文。
3. **语义裁判 (Jina Reranker)**：基于还原后的完整语境执行全局打分，确保送给 LLM 的最终 Prompt 无语义断层，进而从根本上压制生成时的幻觉。
