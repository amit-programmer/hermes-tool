# Graph Report - hermes-tool  (2026-11-25)

## Corpus Check
- 12 files · ~17,892 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 1 file(s) not represented in the graph (top: (none) 1)

## Summary
- 282 nodes · 490 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `faf2bc76`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- chat_upload_and_complete
- SessionState
- app.py
- ⚡ Hermes-3 8B Standalone AI Server & Somi Co-Founder Engine
- model_manager.py
- time
- session_memory.py
- SessionMemoryStore
- client_example.py
- PinnedDocRef
- delete_session_memory

## God Nodes (most connected - your core abstractions)
1. `SessionState` - 21 edges
2. `chat_completions()` - 15 edges
3. `chat_upload_and_complete()` - 15 edges
4. `extract_content_from_file()` - 12 edges
5. `PinnedDocRef` - 12 edges
6. `generate_on_cpu()` - 10 edges
7. `direct_chat_endpoint()` - 10 edges
8. `gradio_chat_fn()` - 10 edges
9. `ChatMessage` - 10 edges
10. `ChatMemory` - 10 edges

## Surprising Connections (you probably didn't know these)
- `📦 Tech Stack & Dependencies` --references--> `spaces`  [INFERRED]
  readme.md → app.py
- `SessionState` --uses--> `PinnedDocRef`  [INFERRED]
  session_memory.py → pinned_docs.py
- `SessionState` --uses--> `PinnedDocs`  [INFERRED]
  session_memory.py → pinned_docs.py
- `chat_completions()` --calls--> `identify_message()`  [EXTRACTED]
  app.py → router.py
- `direct_chat_endpoint()` --calls--> `identify_message()`  [EXTRACTED]
  app.py → router.py

## Import Cycles
- None detected.

## Communities (11 total, 2 thin omitted)

### Community 0 - "chat_upload_and_complete"
Cohesion: 0.08
Nodes (20): chat_completions(), chat_upload_and_complete(), _local_caller(), ChatCompletionRequest, construct_chatml_prompt(), direct_chat_endpoint(), DirectChatRequest, enforce_context_budget() (+12 more)

### Community 1 - "SessionState"
Cohesion: 0.09
Nodes (3): ChatMemory, ChatMessage, SessionState

### Community 2 - "app.py"
Cohesion: 0.07
Nodes (17): decode_text_bytes(), extract_content_from_file(), extract_image(), extract_pdf(), extract_powerpoint(), extract_spreadsheet(), extract_text_code(), extract_word() (+9 more)

### Community 3 - "⚡ Hermes-3 8B Standalone AI Server & Somi Co-Founder Engine"
Cohesion: 0.08
Nodes (22): spaces, 1. 4-Bit NF4 Quantization (~70% Memory Reduction), 1. Chat Completions (OpenAI Compatible), 1. Clone & Setup Virtual Environment, 2. Install Dependencies, 2. Multi-Format File Extraction, 2. Persistent Storage Caching (`/data/hf_cache`), 3. Dynamic Private Persona Engine ("Somi") (+14 more)

### Community 4 - "model_manager.py"
Cohesion: 0.10
Nodes (5): event_generator(), stream_on_cpu(), generate_text(), get_model(), ModelManager

### Community 5 - "time"
Cohesion: 0.13
Nodes (6): all_exception_handler(), get_session_memory(), health_check(), list_active_sessions(), list_models(), root_view()

### Community 6 - "session_memory.py"
Cohesion: 0.10
Nodes (3): warmup_endpoint(), estimate_tokens(), PinnedDocs

### Community 8 - "client_example.py"
Cohesion: 0.16
Nodes (9): get_client_headers(), print_separator(), resolve_gateway_url(), test_chat_completion(), test_health(), test_models(), test_streaming_chat(), test_warmup() (+1 more)

### Community 11 - "delete_session_memory"
Cohesion: 0.17
Nodes (3): delete_session_memory(), HfFolder, startup_event()

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SessionState` connect `SessionState` to `PinnedDocRef`, `app.py`, `session_memory.py`, `SessionMemoryStore`?**
  _High betweenness centrality (0.243) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `SessionState` (e.g. with `ChatMemory` and `ChatMessage`) actually correct?**
  _`SessionState` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `chat_upload_and_complete` be split into smaller, more focused modules?**
  _Cohesion score 0.07751937984496124 - nodes in this community are weakly interconnected._
- **Why does `spaces` connect `⚡ Hermes-3 8B Standalone AI Server & Somi Co-Founder Engine` to `app.py`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Should `SessionState` be split into smaller, more focused modules?**
  _Cohesion score 0.09274193548387097 - nodes in this community are weakly interconnected._
- **Should `app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06531204644412192 - nodes in this community are weakly interconnected._
- **Should `⚡ Hermes-3 8B Standalone AI Server & Somi Co-Founder Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._