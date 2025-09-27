# UI/UX Design Specification

## 1. Introduction
This document outlines the User Interface (UI) and User Experience (UX) design for the Visual Workflow Builder. It serves as a comprehensive guide covering user motivations, interaction patterns, and visual design standards. The goal is to ensure a coherent, intuitive, and effective user experience for building, managing, and executing multi-modal AI workflows.

---

## 2. User Stories
To guide the design, we define three key user personas who represent the primary audience for this tool.

### Persona 1: Dr. Anya Sharma (AI Researcher)
- **Role**: Data Scientist / AI Researcher
- **Goal**: To rapidly prototype, validate, and iterate on new AI models and data processing pipelines without extensive setup or boilerplate code.

| As a(n)... | I want to... | So that I can... |
|---|---|---|
| AI Researcher | ...visually construct a pipeline that takes an image, classifies it, and generates a report. | ...rapidly test the viability of a multi-modal workflow. |
| Data Scientist | ...easily swap different models and datasets from a central hub within my workflow. | ...compare their performance and find the best combination for my task. |
| Researcher | ...save my experimental workflow configuration and load it again later. | ...ensure my experiments are reproducible and share them with colleagues. |

### Persona 2: Ben Carter (Solutions Architect)
- **Role**: Solutions Architect / AI Developer
- **Goal**: To build and demonstrate proof-of-concept AI solutions for clients or internal stakeholders, showcasing complex business logic.

| As a(n)... | I want to... | So that I can... |
|---|---|---|
| Solutions Architect | ...build a conditional workflow that routes data based on classification results. | ...demonstrate complex, real-world business logic to stakeholders. |
| AI Developer | ...see the real-time status of each node (running, completed, error) during execution. | ...easily identify and debug bottlenecks or failures in the workflow. |
| Architect | ...present a clean, intuitive graph of the workflow. | ...help non-technical stakeholders understand the solution's logic at a high level. |

### Persona 3: Chloe Davis (Product Manager)
- **Role**: Product Manager / Business Analyst
- **Goal**: To understand, specify, and analyze AI-driven features without getting bogged down in low-level implementation details.

| As a(n)... | I want to... | So that I can... |
|---|---|---|
| Product Manager | ...use the visual builder to map out the logic for a new AI-powered feature. | ...clearly communicate the requirements to the engineering team. |
| Business Analyst | ...experiment with different conditional logic in a workflow. | ...analyze how different scenarios impact the final business outcome. |

---

## 3. Interaction Flow
This section details the step-by-step journey a user takes to create, run, and manage a workflow.

### 3.1. Initial State
The user opens the application to a clean, empty canvas area, a toolbox populated with available nodes on the left, and a toolbar in the top-right corner.

### 3.2. Building a Workflow
1.  **Node Selection**: The user identifies a desired node (e.g., "Image Input") in the sidebar toolbox.
2.  **Add Node to Canvas**: The user clicks and drags the node from the toolbox onto the main canvas area.
3.  **Positioning**: The user releases the mouse, and the node appears on the canvas. The user can then click and drag the node's header to reposition it.

### 3.3. Configuring a Node
-   **Direct Input**: For nodes like `image_input` or `text_input`, the user interacts with controls directly within the node's body (e.g., clicking "Upload Image" or typing in a textarea).
-   **Configuration via Modal**: For nodes like `model_hub` or `data_hub`, the user clicks the "Configure" button. This action opens a modal window.
-   **Modal Interaction**: Inside the modal, the user can search, browse, and select an item (e.g., a specific model). Upon selection, the modal closes, and the node's body updates to display the selected item's ID or name.

### 3.4. Connecting Nodes
1.  **Initiate Connection**: The user moves their cursor over an output port on a source node. The port highlights.
2.  **Drag Edge**: The user clicks and drags from the output port. A dashed preview line (the "edge-preview") appears, following the cursor.
3.  **Target Connection**: The user drags the preview line to an input port on a target node. If the connection is valid, the input port highlights.
4.  **Complete Connection**: The user releases the mouse over the valid input port. The dashed line becomes a solid, curved line (an "edge"), visually connecting the two nodes.
5.  **Invalid Connection**: If the user releases the mouse over an invalid target (e.g., another output port, the same node, or empty canvas space), the preview line disappears, and no connection is made.

### 3.5. Executing the Workflow
1.  **Trigger Run**: The user clicks the "Run" button in the canvas toolbar.
2.  **Visual Feedback**:
    *   All nodes in the workflow immediately change their border color to blue (`--running-color`) and display a pulsing animation to indicate that the execution has started.
    *   The backend processes the nodes in topological order.
3.  **View Results**:
    *   As nodes complete successfully, their border turns green (`--success-color`).
    *   If a node fails, its border turns red (`--error-color`), and any downstream nodes are skipped.
    *   Output data is displayed directly in the relevant node's body (e.g., the generated text in the `generate_report` node).

### 3.6. Managing Workflows
-   **Save**: The user clicks the "Save" button. A browser prompt asks for a workflow name. Upon confirmation, the current state of nodes and edges is persisted.
-   **Load**: The user clicks the "Load" button, which opens a modal listing all saved workflows. Clicking "Load" on an item closes the modal and replaces the current canvas content with the selected workflow.
-   **Clear**: The user clicks the "Clear Canvas" button to remove all nodes and edges and start fresh.

---

## 4. Visual Specification

### 4.1. Color Palette

| Variable | HEX | Usage |
|---|---|---|
| `--bg-color` | `#121212` | Main application background. |
| `--surface-color` | `#1e1e1e` | Background for UI elements like nodes, sidebar, modals. |
| `--primary-color` | `#6a5acd` | Primary interactive elements: buttons, edges, highlights. |
| `--primary-variant-color` | `#8370e6` | Lighter shade for hover states and accents. |
| `--text-color` | `#e0e0e0` | Primary text color. |
| `--text-secondary-color` | `#a0a0a0` | Secondary text for descriptions, placeholders. |
| `--border-color` | `#333333` | Borders and separators. |
| `--running-color` | `#2196F3` | Status indicator for nodes in an "executing" state. |
| `--success-color` | `#4CAF50` | Status indicator for nodes that completed successfully. |
| `--error-color` | `#cf6679` | Status indicator for nodes that failed. |

### 4.2. Typography
-   **Font Family**: `Inter`, sans-serif.
-   **Hierarchy**:
    -   **App Title**: 1.25rem, 600 weight.
    -   **Modal Title**: 1.1rem, 600 weight.
    -   **Sidebar Title**: 1rem, 500 weight.
    -   **Body Text/Labels**: 0.9rem - 1rem, 400-500 weight.
    -   **Node Title**: 0.85rem, 600 weight.
    -   **Port Labels/Node Content**: 0.8rem, 400 weight.

### 4.3. Component Library
-   **Node**:
    -   **Structure**: A rounded rectangle (`8px` radius) with three sections: Header (icon, title), Body (content, configuration), and Ports (inputs on left, outputs on right).
    -   **State**: The node's status is indicated by its `1px` border:
        -   `idle`: `var(--border-color)`
        -   `running`: `var(--running-color)` with a pulsing `box-shadow` animation.
        -   `completed`: `var(--success-color)`
        -   `error`: `var(--error-color)`
-   **Edge**:
    -   A curved SVG path with `2px` stroke width and `var(--primary-color)`.
    -   The connection preview is a dashed line that animates to indicate direction.
-   **Port**:
    -   A `12x12px` circle that highlights with `var(--primary-color)` on hover.
-   **Buttons**:
    -   **Primary (`.run-btn`, `.select-item-btn`)**: Solid `var(--primary-color)` background with white text.
    -   **Secondary**: `var(--surface-color)` background with a `var(--border-color)`.
    -   **Danger (`.danger-btn`)**: Secondary style with red text/border on hover.
-   **Modal**:
    -   A centered container with a semi-transparent backdrop. It contains a header with a title and close icon, and a scrollable body for content.

### 4.4. Iconography
Icons provide quick visual cues for node functionality and toolbar actions.

| Icon | Component | Meaning |
|---|---|---|
| `<Icons.Input />` | `Image Input` | Represents a node that provides image data. |
| `<Icons.Architecture />` | `Text Input` | Represents a node that provides text data. |
| `<Icons.Classify />` | `Image Classifier` | A task involving classification or analysis. |
| `<Icons.Coordinate />` | `Decision Logic` | Control flow, branching, or logical operations. |
| `<Icons.Solution />` | `Generate Report` | Generating a final output or solution (e.g., via LLM). |
| `<Icons.Resource />` | `Data Hub` | A repository or source of datasets. |
| `<Icons.Inference />` | `Model Hub` | A repository or source of pre-trained models. |
| `<Icons.Run />` | Toolbar Button | Triggers the execution of the entire workflow. |
| `<Icons.Save />` | Toolbar Button | Saves the current workflow state. |
| `<Icons.Load />` | Toolbar Button | Loads a saved workflow from storage. |
| `<Icons.Trash />` | Toolbar Button | Clears the canvas of all nodes and edges. |
| `<Icons.Close />` | Modal Button | Closes the modal window. |
