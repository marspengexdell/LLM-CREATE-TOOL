# Module-to-Node Mapping Specification

## Introduction
This document serves as a bridge between the high-level design of the Multi-Modal LLM system and the practical implementation within the Visual Workflow Builder. It defines the relationship between abstract system modules and concrete UI nodes, specifies node configurations, and establishes a clear data contract for inputs and outputs to ensure seamless execution by the backend engine.

---

## 1. Module to Node Mapping
This section maps the conceptual modules from the system design documents to the visual nodes available in the application's toolbox.

| Design Module (模块) | Corresponding UI Node(s) | Notes |
|---|---|---|
| `输入处理模块` (Input Processing) | `image_input`, `text_input`, `data_hub` | These nodes are responsible for introducing data into the workflow, either through direct user input or from pre-existing datasets. |
| `任务分类模块` (Task Classification) | `image_classifier` | This node is a concrete implementation of the task classification module, specialized for image inputs. |
| `推理模块` (Inference) | `image_classifier`, `generate_report` | These nodes perform core reasoning tasks. `image_classifier` executes classification, while `generate_report` leverages a generative model for more complex, instructed inference. |
| `解决方案模块` (Solution) | `generate_report` | This node acts as a dynamic solution generator, using the Gemini model to produce outputs based on structured prompts and input data. |
| `跨区域协调模块` (Cross-Region Coordination) | **The Canvas & Backend Execution Engine** | This module is not a single node but the overarching system that orchestrates the data flow between nodes as defined by the user. The "Run" button triggers this module's execution via the `/api/v1/workflow/run` endpoint. |
| `动态资源调度模块` (Dynamic Resource Scheduling) | **Backend Infrastructure** | This is a non-functional requirement managed by the backend infrastructure and is not represented by a visual node. |
| `空间感知与方向感模块` (Spatial Perception) | (Future Extension) | Currently, no nodes directly map to this module. Future additions could include nodes like `SLAM_Processor` or `Depth_Map_Input`. |
| Utility & Control Flow | `decision_logic`, `model_hub`, `data_hub` | These nodes provide essential functionalities. `model_hub` and `data_hub` are used to configure processing nodes, while `decision_logic` enables conditional workflow paths. |

---

## 2. Node Configuration & Data Contracts
This section provides a detailed specification for each node available in the toolbox.

---

#### **Node: `image_input` (Image Input)**
- **Maps to Module**: `输入处理模块` (Input Processing Module)
- **Description**: Provides an image to the workflow via direct user upload.
- **Configuration (via UI)**:
  - `data.image`: The user-uploaded image.
- **Data Contract**:
  - **Inputs**: None
  - **Outputs**:
    - `out (Image)`:
      - **Type**: `string`
      - **Format**: Base64 encoded data URI (e.g., `data:image/jpeg;base64,...`). This ensures the image data is self-contained within the workflow's JSON payload.

---

#### **Node: `text_input` (Text Input)**
- **Maps to Module**: `输入处理模块` (Input Processing Module)
- **Description**: Provides a block of text to the workflow.
- **Configuration (via UI)**:
  - `data.text`: User-entered text content.
- **Data Contract**:
  - **Inputs**: None
  - **Outputs**:
    - `out (Text)`:
      - **Type**: `string`
      - **Format**: Plain text.

---

#### **Node: `image_classifier` (Image Classifier)**
- **Maps to Module**: `任务分类模块` (Task Classification), `推理模块` (Inference)
- **Description**: Classifies an image using a specified model.
- **Configuration (via UI)**: None directly on the node. The model is supplied via the `Model` input port.
- **Data Contract**:
  - **Inputs**:
    - `image (Image)`:
      - **Type**: `string`
      - **Format**: Base64 encoded data URI, typically from an `image_input` or `data_hub` node.
    - `model (Model)`:
      - **Type**: `object`
      - **Format**: JSON object specifying the model to use, provided by a `model_hub` node. Example: `{ "model_id": "resnet50_v2" }`.
  - **Outputs**:
    - `out (Classification)`:
      - **Type**: `object`
      - **Format**: JSON object containing the classification results. Example: `{ "label": "cat", "confidence": 0.98, "all_scores": [{"label": "cat", "score": 0.98}, {"label": "dog", "score": 0.02}] }`.

---

#### **Node: `decision_logic` (Decision Logic)**
- **Maps to Module**: Utility / Control Flow
- **Description**: Evaluates a condition on input data to route the workflow.
- **Configuration (via UI)**:
  - `data.condition`: A string expression to be evaluated by the backend. The expression can reference the input data using a variable like `input`. Example: `input.confidence > 0.9`.
- **Data Contract**:
  - **Inputs**:
    - `in (Data)`:
      - **Type**: `any`
      - **Format**: Any valid JSON data from an upstream node.
  - **Outputs**:
    - `true (If True)`:
      - **Type**: `any`
      - **Format**: The original input data is passed through if the condition evaluates to true.
    - `false (If False)`:
      - **Type**: `any`
      - **Format**: The original input data is passed through if the condition evaluates to false.

---

#### **Node: `generate_report` (Generate Report (Gemini))**
- **Maps to Module**: `推理模块` (Inference), `解决方案模块` (Solution)
- **Description**: Generates a text report using the Gemini API based on a prompt template and input data. The underlying model is `gemini-2.5-flash`.
- **Configuration (via UI)**:
  - `data.prompt`: A prompt template. The backend execution engine should support placeholder substitution (e.g., using `{{...}}` syntax) to inject data from the input. Example: `"Summarize the following object: {{input.label}}."`
- **Data Contract**:
  - **Inputs**:
    - `in (Data)`:
      - **Type**: `any`
      - **Format**: Any valid JSON data, which can be referenced in the prompt template.
  - **Outputs**:
    - `out (Report)`:
      - **Type**: `string`
      - **Format**: The plain text report generated by the Gemini model.

---

#### **Node: `data_hub` (Data Hub)**
- **Maps to Module**: `输入处理模块` (Input Processing Module)
- **Description**: Provides a reference to a pre-uploaded dataset.
- **Configuration (via UI)**:
  - `data.datasetId`: The unique identifier for the selected dataset.
  - `data.datasetName`: The display name for the selected dataset.
- **Data Contract**:
  - **Inputs**: None
  - **Outputs**:
    - `out (Dataset)`:
      - **Type**: `object`
      - **Format**: A JSON object referencing the dataset. The backend will use this reference to load the appropriate data for connected nodes. Example: `{ "dataset_id": "ds_xyz_789", "type": "image" }`.

---

#### **Node: `model_hub` (Model Hub)**
- **Maps to Module**: Utility
- **Description**: Provides a reference to a pre-trained model.
- **Configuration (via UI)**:
  - `data.modelId`: The unique identifier for the selected model.
- **Data Contract**:
  - **Inputs**: None
  - **Outputs**:
    - `out (Model)`:
      - **Type**: `object`
      - **Format**: A JSON object referencing the model. This object is passed to the `Model` input of nodes like `image_classifier`. Example: `{ "model_id": "image-classifier-v1" }`.
