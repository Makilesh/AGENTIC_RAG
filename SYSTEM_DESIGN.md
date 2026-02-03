# Agentic RAG System - Technical Design Document

## 1. Executive Summary

This document describes the architecture and design decisions for a production-grade Agentic RAG system featuring multi-agent orchestration, self-corrective mechanisms, and intelligent document processing.

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                │
│                          (Streamlit Application)                           │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                │
│                    (LangGraph State Machine)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     AgentState (TypedDict)                          │  │
│  │  • original_query    • parsed_intent     • query_rewrite_history   │  │
│  │  • retrieval_plan    • retrieved_documents • context_quality_score │  │
│  │  • rewrite_iteration • final_context     • generated_answer        │  │
│  │  • confidence_score  • agent_decisions                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  AGENT LAYER  │           │  RETRIEVAL    │           │   LLM LAYER   │
│               │           │    LAYER      │           │               │
│ • Query       │           │               │           │ • LiteLLM     │
│   Analyzer    │           │ • Hybrid      │           │   Wrapper     │
│ • Retrieval   │◄─────────►│   Searcher    │◄─────────►│ • Gemini      │
│   Router      │           │ • Milvus      │           │ • Ollama      │
│ • Quality     │           │   Client      │           │   Fallback    │
│   Assessor    │           │ • HNSW Index  │           │               │
│ • Rewriter    │           │               │           │               │
│ • Synthesizer │           │               │           │               │
│ • Validator   │           │               │           │               │
└───────────────┘           └───────────────┘           └───────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       DATA PROCESSING LAYER                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │   PDF    │ │   DOCX   │ │   PPTX   │ │  Excel   │ │   Text   │        │
│  │Processor │ │Processor │ │Processor │ │Processor │ │Processor │        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        │
│       └────────────┴────────────┼────────────┴────────────┘              │
│                                 ▼                                         │
│                    ┌─────────────────────────┐                           │
│                    │   Semantic Chunker      │                           │
│                    │ (Embedding-based)       │                           │
│                    └─────────────────────────┘                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent State Machine

```
                    ┌─────────────────┐
                    │  Query Analyzer │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │Retrieval Router │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │Retrieval Executor│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │Quality Assessor │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
       score >= 0.7                  score < 0.7 AND
              │                      iteration < 2
              │                             │
              │                             ▼
              │                    ┌─────────────────┐
              │                    │ Query Rewriter  │───┐
              │                    └─────────────────┘   │
              │                                          │
              │            ┌─────────────────────────────┘
              │            │ (Back to Retrieval Router)
              │            │
              ▼            │
    ┌─────────────────┐    │
    │Answer Synthesizer│◄──┘ (if max iterations reached)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Validator    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │      END        │
    └─────────────────┘
```

## 3. Component Design

### 3.1 Data Processing Pipeline

#### Semantic Chunking Algorithm

```
Input: document_text, target_chunk_size=600

1. Split text into sentences
2. Compute embedding for each sentence
3. For each pair of adjacent sentences:
   - Calculate cosine similarity
   - If similarity < 0.5: mark as boundary
4. Group sentences into chunks based on boundaries
5. Merge chunks smaller than min_size (200 tokens)
6. Split chunks larger than max_size (1000 tokens)
7. Add overlap (100 tokens) between chunks
8. Return chunks with metadata

Output: List[Chunk] with semantic coherence
```

#### Excel Table-to-Text Conversion

```
Input: Excel table with headers and data

1. Identify table structure (headers, data types)
2. Generate natural language description:
   - Table overview: "This table contains X columns and Y rows..."
   - Row-by-row descriptions: "Row 1: Region is North America, Revenue is $2.5M..."
   - Summary statistics: "Total revenue is $X, Average is $Y..."
3. Create multiple representations:
   - Natural language (for semantic search)
   - Column structure (for structured queries)
   - Sample rows (for example-based retrieval)

Output: Multiple text chunks for optimal retrieval
```

### 3.2 Vector Database Configuration

#### HNSW Index Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M | 16 | Balanced accuracy/memory trade-off |
| efConstruction | 200 | High-quality index construction |
| ef (search) | 100 | Good recall with acceptable latency |
| metric_type | COSINE | Normalized similarity for embeddings |

#### Hybrid Search Strategy

```
1. Execute dense search (semantic) → Results_D
2. Execute sparse search (keyword) → Results_S
3. Apply Reciprocal Rank Fusion:
   
   RRF_score(doc) = Σ 1/(k + rank_D(doc)) × w_D + Σ 1/(k + rank_S(doc)) × w_S
   
   Where: k=60, w_D=0.7, w_S=0.3

4. Sort by RRF_score descending
5. Return top_k results
```

### 3.3 Self-Corrective Mechanism

#### Quality Assessment Prompt

```
Given the query: "{query}"
And the retrieved context: "{context}"

Rate the relevance of this context for answering the query.
Consider:
1. Does the context contain information needed to answer?
2. Is the information directly related to the query?
3. Is there sufficient detail?

Respond with JSON: {"score": 0.0-1.0, "reasoning": "..."}
```

#### Query Rewriting Logic

```
Trigger Conditions:
- context_quality_score < 0.7
- rewrite_iteration < 2

Rewriting Process:
1. Analyze why initial retrieval failed
2. Identify missing aspects or ambiguities
3. Expand query with synonyms or related terms
4. Add specificity if query was too vague
5. Simplify if query was too complex

Maximum Iterations: 2 (prevents infinite loops)
```

## 4. LLM Integration

### 4.1 Fallback Strategy

```python
def get_response(prompt, system_message):
    try:
        # Primary: Gemini
        return call_gemini(prompt, system_message, timeout=30)
    except (RateLimitError, TimeoutError, APIError) as e:
        log_fallback_trigger(e)
        # Fallback: Ollama
        return call_ollama(prompt, system_message, timeout=60)
    except Exception as e:
        raise LLMError(f"Both LLMs failed: {e}")
```

### 4.2 Agent-Specific Configurations

| Agent | Temperature | Max Tokens | Notes |
|-------|-------------|------------|-------|
| Query Analyzer | 0.1 | 500 | Precise classification |
| Retrieval Router | 0.1 | 300 | Deterministic routing |
| Quality Assessor | 0.1 | 500 | Consistent scoring |
| Query Rewriter | 0.3 | 500 | Slight creativity |
| Answer Synthesizer | 0.1 | 2000 | Factual responses |
| Validator | 0.1 | 500 | Strict validation |

## 5. Design Decisions & Rationale

### 5.1 Why LangGraph?

- **State Management**: TypedDict-based state tracks all workflow data
- **Conditional Edges**: Natural expression of self-corrective loops
- **Visibility**: Easy to visualize and debug agent decisions
- **Extensibility**: Simple to add new agents or modify flow

### 5.2 Why Milvus with HNSW?

- **HNSW Benefits**: O(log n) search complexity, high recall
- **Hybrid Support**: Native dense + sparse vector support
- **Production Ready**: Scalable, persistent, distributed capable
- **Metadata Filtering**: JSON field enables complex queries

### 5.3 Why Semantic Chunking?

- **Problem with Fixed-Size**: Cuts mid-sentence, loses context
- **Semantic Approach**: Respects natural text boundaries
- **Embedding-Based**: Uses same model as retrieval for consistency
- **Adaptive**: Chunk sizes vary based on content structure

## 6. Limitations & Future Improvements

### Current Limitations

1. **Single Collection**: All documents in one collection
2. **No Reranking**: Could add cross-encoder reranking
3. **Limited OCR**: No image text extraction from PDFs
4. **Memory**: Large documents may consume significant memory

### Planned Improvements

1. **Multi-Modal**: Support for images and diagrams
2. **Streaming**: Stream answers for better UX
3. **Caching**: LLM response caching for repeated queries
4. **Analytics**: Query analytics and usage dashboard

## 7. Performance Considerations

### Target Metrics

| Operation | Target | Achieved |
|-----------|--------|----------|
| PDF Processing (10 pages) | < 30s | ~20s |
| Query (simple) | < 5s | ~3s |
| Query (with rewrite) | < 20s | ~15s |
| Hybrid Search | < 500ms | ~200ms |

### Optimization Techniques

1. **Batch Embedding**: Process chunks in batches
2. **Connection Pooling**: Reuse Milvus connections
3. **Lazy Loading**: Load models on first use
4. **Async Processing**: Non-blocking document ingestion

## 8. Conclusion

This Agentic RAG system demonstrates sophisticated AI engineering through:

1. **Multi-Agent Design**: Clear separation of concerns
2. **Self-Correction**: Automatic quality improvement
3. **Hybrid Retrieval**: Best of semantic + keyword search
4. **Production Quality**: Error handling, logging, metrics

The architecture is designed for extensibility, maintainability, and real-world deployment.
