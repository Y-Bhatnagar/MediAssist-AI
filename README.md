****AAYU — AI Medical Assistant****

A multi-agent, safety-conscious AI system for structured medical intake, context building, and retrieval-grounded clinical conversations.

---

**Overview**

AAYU is a non-diagnostic AI medical assistant designed to support healthcare workflows before and during clinical interactions. Its primary goal is to structure information, summarise it, and present it in a concise, structured form suitable for clinical review

The system engages patients through guided questioning, captures relevant patient-provided context, and combines it with extracted facts from medical documents. The outcome is a concise, structured summary that medical professionals can review before interacting with the patient.

---

**Motivation**

Clinical interactions often suffer from:

* Fragmented patient histories
* Long, unstructured medical reports
* Time constraints during consultations
* Repetitive intake questioning

AAYU addresses these challenges by acting as a pre-conversation structuring layer, helping ensure that clinicians start with a better context rather than raw documents or incomplete patient recollection.

---

**What AAYU Does**

Intelligent Patient Intake

* Asks context-aware, relevant questions to patients
* Maintains conversational continuity across turns
* Captures patient-provided details in a structured form

Medical Document Processing

* Extracts structured information from PDFs and scanned reports
* Uses Azure Document Intelligence for reliable document parsing

Structured Summarization

* Produces concise summaries combining:
  * patient responses
  * document-derived facts
* Designed for pre-visit review by clinicians

Doctor Mode (Grounded Exploration)

* Enables medical professionals to explore prior summaries
* Uses retrieval-grounded responses to surface relevant facts
* Supports exploratory questioning without making decisions

---

**Multi-Agent Architecture (LangGraph)**

AAYU is implemented using LangGraph to coordinate multiple specialized agents with clear separation of responsibilities.

AAYU Orchestrator Node

* Central controller for conversation flow
* Maintains shared graph state
* Routes execution between agents
* Controls transitions between Patient Mode and Doctor Mode

Summarization Agent (LLM)

* Converts conversation state into structured summaries
* Prevents context growth in long interactions

Doctor Mode (Sub-Orchestrator)

* Activated for clinician-facing workflows
* Determines when retrieval is required
* Delegates factual lookup to a retrieval agent
* Handles clinician-facing conversations that do not require retrieval

Retrieval Agent (LLM)

* Calls vector or hybrid search tools
* Evaluates retrieval sufficiency
* Rephrases queries when the retrieved content is not sufficient to answer.

---

**Post-Conversation Knowledge Pipeline**

At the end of Aayu’s session:

1. A State Writer Node extracts relevant information from the graph state
2. The extracted data is written to a structured text artifact
3. The artifact is chunked, embedded, and stored in a vector database

This enables future retrieval and longitudinal context without reprocessing original documents.

---

**Technology Stack**

* Orchestration: LangGraph
* Backend: FastAPI (async)
* Realtime Communication: WebSockets + asyncio
* Document Processing: Azure Document Intelligence
* Retrieval & Memory: Azure AI Search (vector embeddings)
* LLMs: Separate instances for conversation, summarization, and retrieval

---
**Architecture Diagram**

<img width="692" height="537" alt="image" src="https://github.com/user-attachments/assets/f6f008b9-684a-4fe8-bc82-a9056b0ff3e0" />


---
**Why This Project Matters**

AAYU demonstrates how agent-orchestrated LLM systems can be applied responsibly in high-risk domains by:

* Enforcing clear boundaries on model behavior
* Separating information gathering from reasoning
* Using retrieval as a first-class control mechanism

---

**Disclaimer**

This project is for educational and demonstration purposes only. It is not a medical device and does not provide medical advice, diagnosis, or treatment.

---

Author: Yash Bhatnagar
