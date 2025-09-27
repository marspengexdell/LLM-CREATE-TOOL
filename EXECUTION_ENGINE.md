# Execution Engine Design

## Introduction
This document details the internal logic of the backend execution engine, which is responsible for running workflows defined by the frontend visual builder. It covers the core components: execution scheduling, state management, data flow, and error handling. This engine is triggered by a `POST` request to the `/api/v1/workflow/run` endpoint.

---

## 1. Execution Order Algorithm (执行顺序算法)
To execute the nodes in the correct order, the engine uses a **Topological Sort** algorithm. This ensures that a node is only executed after all its dependencies (upstream nodes) have successfully completed.

### Input
The engine receives a JSON payload representing the workflow graph:
```json
{
  "nodes": [
    { "id": "node-1", "type": "image_input", ... },
    { "id": "node-2", "type": "model_hub", ... },
    { "id": "node-3", "type": "image_classifier", ... }
  ],
  "edges": [
    { "id": "edge-1", "fromNode": "node-1", "fromPort": "out", "toNode": "node-3", "toPort": "image" },
    { "id": "edge-2", "fromNode": "node-2", "fromPort": "out", "toNode": "node-3", "toPort": "model" }
  ]
}
```

### Algorithm Steps
The engine uses Kahn's algorithm for topological sorting:

1.  **Build Adjacency List & In-Degree Map**:
    *   Parse the `nodes` and `edges` lists.
    *   Create a directed graph representation (e.g., an adjacency list where `graph[node_id]` lists its children).
    *   Calculate the "in-degree" for each node (the number of incoming edges).

2.  **Initialize Queue**:
    *   Create a queue and add all nodes with an in-degree of 0. These are the starting nodes of the workflow (e.g., `image_input`, `model_hub`).

3.  **Process Queue**:
    *   Initialize an empty list `execution_order`.
    *   While the queue is not empty:
        *   Dequeue a node, let's call it `current_node`.
        *   Add `current_node` to the `execution_order` list.
        *   For each `neighbor_node` that `current_node` points to:
            *   Decrement the in-degree of `neighbor_node`.
            *   If the in-degree of `neighbor_node` becomes 0, add it to the queue.

4.  **Cycle Detection**:
    *   After the loop finishes, if the length of the `execution_order` list is less than the total number of nodes, it means the graph has at least one cycle. In this case, the workflow is invalid, and the engine should immediately return an error.

The final `execution_order` list dictates the sequence in which the nodes will be processed.

---

## 2. State Management (状态管理)
Each node in the workflow has a `status` field that reflects its state during the execution lifecycle.

### Node States
-   `idle`: The default state of a node before execution begins.
-   `running`: The node is actively being processed. The frontend provides immediate feedback by setting all nodes to this state upon clicking "Run".
-   `completed`: The node's execution finished successfully.
-   `error`: An error occurred during the node's execution.
-   `skipped`: The node was not executed because one of its upstream dependencies failed.

### State Communication
The current implementation uses a synchronous request-response model for state updates.

-   **Final State Update**: The `/api/v1/workflow/run` endpoint executes the entire workflow and only returns a response once the process is complete (either fully successful or terminated by an error). The response body contains the full list of nodes, each with its final `status` and updated `data` (including any outputs or error messages).

-   **(Future Extension) Real-time Updates**: For long-running workflows, a real-time update mechanism could be implemented using WebSockets or Server-Sent Events (SSE). The backend would push state changes (`pending`, `running`, `completed`, `error`) for each node to the frontend as they occur, providing a more dynamic and responsive user experience.

---

## 3. Data Passing Mechanism (数据传递机制)
Data flows between nodes according to the connections (edges) defined in the graph. The engine manages this data flow for the duration of a single workflow run.

### In-Memory Context
-   A temporary, in-memory key-value store (e.g., a dictionary or hash map), called the `execution_context`, is created for each workflow run.
-   This context stores the output data of every successfully executed node. The key is a composite of the node ID and the output port ID (e.g., `f"{node.id}:{port.id}"`), and the value is the data produced by that port.

### Execution Flow
1.  **Input Gathering**: Before executing a node from the `execution_order`, the engine identifies all its required inputs by looking at its incoming edges.
2.  **Data Retrieval**: For each input port, the engine uses the `fromNode` and `fromPort` information from the connecting edge to look up the corresponding output data in the `execution_context`.
3.  **Input Assembly**: The retrieved data is assembled into a structured format (e.g., a dictionary mapping input port IDs to their data) and passed to the node's execution function.
4.  **Output Storage**: After a node executes successfully, its return value(s) are stored in the `execution_context`. If a node has multiple output ports, it's expected to return a dictionary mapping each output port ID to its corresponding data. This data is then available for any downstream nodes.

This context is discarded once the workflow run is complete.

---

## 4. Error Handling and Logging (错误处理与日志)

Robust error handling is critical for providing clear user feedback and for debugging.

### System Behavior on Failure
1.  **Isolate Failure**: When a node's execution logic throws an exception, the engine catches it.
2.  **Update Node State**:
    *   The status of the failed node is set to `error`.
    *   A user-friendly error message is stored in the node's data payload (e.g., `node.data.error = "Detailed error message."`).
3.  **Halt Downstream Execution**: The execution of the current path is stopped. The engine iterates through all downstream nodes that depend on the failed node (and their descendants) and sets their status to `skipped`.
4.  **Terminate Workflow**: The overall workflow run is considered failed. The engine stops processing and prepares the final response. Parallel, independent branches of the workflow will not be executed once a failure is detected in any branch.
5.  **API Response**: The `/api/v1/workflow/run` endpoint returns an appropriate HTTP error status (e.g., `400 Bad Request` for user-related errors like cycles, `500 Internal Server Error` for execution failures). The response body includes the final state of all nodes, allowing the frontend to visually represent the failure.

### Logging
Comprehensive server-side logging is essential for diagnostics.

-   **Run ID**: Each call to `/api/v1/workflow/run` should generate a unique `run_id`.
-   **Tagged Logs**: All log entries related to a specific run must be tagged with this `run_id`.
-   **Log Content**: The following events should be logged:
    *   Workflow execution started and finished (with status: success/failure).
    *   Node execution started.
    *   Node execution finished successfully, including a summary of its output.
    *   Node execution failed, including the full error message and stack trace.
    *   The determined topological sort order.
