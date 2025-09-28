
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

const DEFAULT_ERROR_MESSAGE = 'Request failed, please retry.';

codex/add-lightweight-toast-mechanism
const TOAST_DURATION = 5000;

const ToastContext = React.createContext(() => {});

const useToast = () => React.useContext(ToastContext);

const ToastViewport = ({ toasts, onDismiss }) => (
    <div className="toast-viewport" role="region" aria-live="polite" aria-atomic="true" aria-label="Notifications">
        {toasts.map((toast) => (
            <div
                key={toast.id}
                className={`toast-card toast-${toast.tone}`}
                role={toast.tone === 'error' ? 'alert' : 'status'}
                aria-live={toast.tone === 'error' ? 'assertive' : 'polite'}
            >
                <div className="toast-copy">
                    <h4>{toast.title}</h4>
                    {toast.description ? <p>{toast.description}</p> : null}
                </div>
                <button
                    type="button"
                    className="toast-dismiss-btn"
                    onClick={() => onDismiss(toast.id)}
                    aria-label="Dismiss notification"
                >
                    <Icons.Close />
                </button>
            </div>
        ))}
    </div>
);

const handleApiError = (enqueueToast, error, fallbackMessage = DEFAULT_ERROR_MESSAGE, context = 'API request failed') => {
    console.error(context, error);
    enqueueToast({
        tone: 'error',
        title: 'We hit a snag',
        description: fallbackMessage,
    });

const extractErrorInfoFromPayload = (payload, fallbackMessage = DEFAULT_ERROR_MESSAGE) => {
    const messageCandidate = payload?.error?.message;
    const normalizedMessage = typeof messageCandidate === 'string' && messageCandidate.trim().length > 0
        ? messageCandidate.trim()
        : fallbackMessage;

    const errorCodeCandidate = payload?.error?.details?.errorCode;
    const normalizedCode = typeof errorCodeCandidate === 'string' && errorCodeCandidate.trim().length > 0
        ? errorCodeCandidate.trim()
        : undefined;

    return { message: normalizedMessage, code: normalizedCode };
};

const createErrorInfo = (error, fallbackMessage = DEFAULT_ERROR_MESSAGE) => {
    if (error && typeof error === 'object') {
        const messageCandidate = 'message' in error ? error.message : undefined;
        if (typeof messageCandidate === 'string' && messageCandidate.trim().length > 0) {
            return { message: messageCandidate.trim() };
        }
    }

    if (typeof error === 'string' && error.trim().length > 0) {
        return { message: error.trim() };
    }

    return { message: fallbackMessage };
};

const handleApiError = (errorInfo, context = 'API request failed') => {
    const { message = DEFAULT_ERROR_MESSAGE, code } = errorInfo || {};
    const displayMessage = code ? `${message} (Code: ${code})` : message;
    console.error(context, { message: displayMessage, code });
    alert(displayMessage);
main
};

// --- ICONS ---
const Icons = {
    Architecture: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 21v-1.5a2.25 2.25 0 0 1 2.25-2.25h12a2.25 2.25 0 0 1 2.25 2.25V21m-15-9.75v-1.5a2.25 2.25 0 0 1 2.25-2.25h12a2.25 2.25 0 0 1 2.25 2.25v1.5m-15-9.75V3.75a2.25 2.25 0 0 1 2.25-2.25h12a2.25 2.25 0 0 1 2.25 2.25v1.5" /><path strokeLinecap="round" strokeLinejoin="round" d="M3 12h18M3 17.25h18M3 6.75h18" /></svg>,
    Input: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M9 8.25H7.5a2.25 2.25 0 0 0-2.25 2.25v9a2.25 2.25 0 0 0 2.25 2.25h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25H15m0-3-3-3m0 0-3 3m3-3v11.25" /></svg>,
    Classify: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-1.003 1.11-1.226a2.25 2.25 0 0 1 2.58 1.226l.121.542a2.25 2.25 0 0 0 1.082 1.082l.542.121a2.25 2.25 0 0 1 1.226 2.58l-.121.542a2.25 2.25 0 0 0-1.082 1.082l-.542.121a2.25 2.25 0 0 1-2.58 1.226l-.121-.542a2.25 2.25 0 0 0-1.082-1.082L9.594 14.44m-5.132 5.132a2.25 2.25 0 0 1 3.182 0l2.121 2.121a2.25 2.25 0 0 1 0 3.182L9.75 24.75l-3.182 0L2.25 20.432l2.121-2.121z" /></svg>,
    Inference: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Z" /></svg>,
    Coordinate: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M13.5 10.5V6.75a4.5 4.5 0 1 1 9 0v3.75M3.75 21.75h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H3.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" /></svg>,
    Solution: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="m3.75 13.5 10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75Z" /></svg>,
    Resource: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" /></svg>,
    Save: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M9 3.75H6.75A2.25 2.25 0 0 0 4.5 6v12a2.25 2.25 0 0 0 2.25 2.25h10.5A2.25 2.25 0 0 0 19.5 18V9.75l-4.5-4.5H9Z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 3.75v4.5a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75v-4.5" /></svg>,
    Load: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9.75h16.5v10.5a1.5 1.5 0 0 1-1.5 1.5H5.25a1.5 1.5 0 0 1-1.5-1.5V9.75Z" /><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9.75V4.5a1.5 1.5 0 0 1 1.5-1.5h4.5l2.25 2.25H18.75a1.5 1.5 0 0 1 1.5 1.5v3" /></svg>,
    Trash: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.134-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.067-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" /></svg>,
    Run: () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.647c1.295.742 1.295 2.545 0 3.286L7.279 20.99c-1.25.717-2.779-.217-2.779-1.643V5.653Z" clipRule="evenodd" /></svg>,
    Close: () => <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" /></svg>
};

// --- MODULE DEFINITIONS ---
const TOOLBOX_MODULES = [
  { type: 'image_input', name: 'Image Input', icon: Icons.Input, inputs: [], outputs: [{ id: 'out', name: 'Image' }] },
  { type: 'text_input', name: 'Text Input', icon: Icons.Architecture, inputs: [], outputs: [{ id: 'out', name: 'Text' }] },
  { type: 'image_classifier', name: 'Image Classifier', icon: Icons.Classify, inputs: [{ id: 'image', name: 'Image' }, { id: 'model', name: 'Model' }], outputs: [{ id: 'out', name: 'Classification' }] },
  { type: 'decision_logic', name: 'Decision Logic', icon: Icons.Coordinate, inputs: [{ id: 'in', name: 'Data' }], outputs: [{ id: 'true', name: 'If True' }, { id: 'false', name: 'If False' }] },
  { type: 'generate_report', name: 'Generate Report (Gemini)', icon: Icons.Solution, inputs: [{ id: 'in', name: 'Data' }], outputs: [{ id: 'out', name: 'Report' }] },
  { type: 'data_hub', name: 'Data Hub', icon: Icons.Resource, inputs: [], outputs: [{ id: 'out', name: 'Dataset' }] },
  { type: 'model_hub', name: 'Model Hub', icon: Icons.Inference, inputs: [], outputs: [{ id: 'out', name: 'Model' }] },
];

const getEdgePath = (startPos, endPos) => {
    const dx = endPos.x - startPos.x;
    const curveX = Math.abs(dx) * 0.6;
    return `M ${startPos.x},${startPos.y} C ${startPos.x + curveX},${startPos.y} ${endPos.x - curveX},${endPos.y} ${endPos.x},${endPos.y}`;
};

// --- UI COMPONENTS ---

const Modal = ({ isOpen, onClose, title, children }) => {
    if (!isOpen) return null;
    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>{title}</h3>
                    <button onClick={onClose} className="modal-close-btn"><Icons.Close /></button>
                </div>
                <div className="modal-body">{children}</div>
            </div>
        </div>
    );
};

const ModelHubConfigurator = ({ onSelectModel }) => {
    const enqueueToast = useToast();
    const [models, setModels] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');

    const fetchModels = useCallback(async () => {
        try {
            setIsLoading(true);
            const response = await fetch('/api/v1/models');
            if (!response.ok) {
                throw new Error('Network response was not ok');
codex/add-lightweight-toast-mechanism


    useEffect(() => {
        const fetchModels = async () => {
            try {
                setIsLoading(true);
                const response = await fetch('/api/v1/models');
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setModels(data);
            } catch (error) {
                handleApiError(createErrorInfo(error), 'Failed to fetch models');
            } finally {
                setIsLoading(false);
main
main
            }
            const data = await response.json();
            setModels(data);
        } catch (error) {
            handleApiError(enqueueToast, error, DEFAULT_ERROR_MESSAGE, 'Failed to fetch models');
        } finally {
            setIsLoading(false);
        }
    }, [enqueueToast]);

    useEffect(() => {
        fetchModels();
    }, [fetchModels]);

    const searchQuery = searchTerm.trim().toLowerCase();
    const filteredModels = models.filter(model => {
        if (!searchQuery) return true;
        const fields = [model?.name, model?.source, model?.description];
        return fields
            .filter(value => typeof value === 'string')
            .some(value => value.toLowerCase().includes(searchQuery));
    });

    const formatRegisteredAt = (value) => {
        if (!value) return null;
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) {
            return value;
        }
        return parsed.toLocaleString();
    };

    return (
        <div className="configurator-container">
            <input
                type="text"
                placeholder="Search models..."
                className="modal-search-input"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
            />
            <div className="item-list">
                {isLoading ? (
                    <p>Loading models...</p>
                ) : filteredModels.length === 0 ? (
                    <div className="empty-state">
                        <h4>No models available</h4>
                        <p>Try refreshing to see the latest models.</p>
                        <div className="empty-state-actions">
                            <button onClick={fetchModels} className="select-item-btn">Retry</button>
                        </div>
                    </div>
                ) : (
                    filteredModels.map(model => {
                        const registeredLabel = formatRegisteredAt(model.registeredAt);
                        return (
                        <div key={model.id} className="model-item item-card">
                            <div className="item-info">
                                <h4>{model.name}</h4>
                                {model.description ? <p>{model.description}</p> : null}
                                <dl className="model-meta">
                                    <div>
                                        <dt>Source</dt>
                                        <dd>{model.source || '—'}</dd>
                                    </div>
                                    <div>
                                        <dt>Quantization</dt>
                                        <dd>{model.quantization || model.metadata?.quantizationScheme || '—'}</dd>
                                    </div>
                                    {registeredLabel ? (
                                        <div>
                                            <dt>Registered</dt>
                                            <dd>{registeredLabel}</dd>
                                        </div>
                                    ) : null}
                                </dl>
                            </div>
                            <button onClick={() => onSelectModel(model.id)} className="select-item-btn">Select</button>
                        </div>
                    );
                    })
                )}
            </div>
        </div>
    );
};

const DataHubConfigurator = ({ onSelectDataset }) => {
    const enqueueToast = useToast();
    const [datasets, setDatasets] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const fileInputRef = useRef(null);
    const [maxUploadBytes, setMaxUploadBytes] = useState(null);

    const fetchDatasets = useCallback(async () => {
        try {
            setIsLoading(true);
            const response = await fetch('/api/v1/datasets');
            const data = await response.json();
            setDatasets(data);
        } catch (error) {
codex/add-lightweight-toast-mechanism
            handleApiError(enqueueToast, error, DEFAULT_ERROR_MESSAGE, 'Failed to fetch datasets');

            handleApiError(createErrorInfo(error), 'Failed to fetch datasets');
main
        } finally {
            setIsLoading(false);
        }
    }, [enqueueToast]);

    useEffect(() => {
        fetchDatasets();
    }, [fetchDatasets]);

    useEffect(() => {
        const fetchMetadata = async () => {
            try {
                const response = await fetch('/api/v1/metadata');
                if (!response.ok) {
                    throw new Error('Failed to fetch metadata');
                }
                const data = await response.json();
                const limit = data?.datasetUpload?.maxBytes;
                if (typeof limit === 'number' && Number.isFinite(limit) && limit > 0) {
                    setMaxUploadBytes(limit);
                }
            } catch (error) {
                console.warn('Failed to fetch backend metadata', error);
            }
        };

        fetchMetadata();
    }, []);

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (typeof maxUploadBytes === 'number' && file.size > maxUploadBytes) {
            const maxSizeMb = Math.ceil(maxUploadBytes / (1024 * 1024));
            alert(`File exceeds maximum allowed size of ${maxSizeMb} MB.`);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/v1/datasets/upload', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            // Refresh dataset list after successful upload
            fetchDatasets();
        } catch (error) {
codex/add-lightweight-toast-mechanism
            handleApiError(enqueueToast, error, DEFAULT_ERROR_MESSAGE, 'Failed to upload dataset file');

             handleApiError(createErrorInfo(error), 'Failed to upload dataset file');
main
        }
    };

    const handleUploadAction = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="configurator-container">
            <div className="upload-area">
                <h4>Upload New Dataset</h4>
                {typeof maxUploadBytes === 'number' && (
                    <p className="upload-hint">Maximum file size: {Math.ceil(maxUploadBytes / (1024 * 1024))} MB</p>
                )}
                <label className="upload-btn-large" onClick={handleUploadAction}>
                    Click to select or drag and drop a file
                    <input
                        ref={fileInputRef}
                        type="file"
                        aria-label="Dataset file input"
                        onChange={handleFileUpload}
                        style={{ display: 'none' }}
                    />
                </label>
            </div>
             <div className="item-list">
                {isLoading ? (
                    <p>Loading datasets...</p>
                ) : datasets.length === 0 ? (
                    <div className="empty-state">
                        <h4>No datasets available</h4>
                        <p>Upload a new dataset or try fetching again.</p>
                        <div className="empty-state-actions">
                            <button onClick={fetchDatasets} className="select-item-btn">Retry</button>
                            <button onClick={handleUploadAction} className="configure-btn">Upload dataset</button>
                        </div>
                    </div>
                ) : (
                    datasets.map(dataset => (
                        <div key={dataset.id} className="dataset-item item-card">
                            <div className="dataset-preview">
                                {dataset.type === 'image' && <img src={dataset.preview} alt={dataset.name} />}
                            </div>
                            <div className="item-info">
                                <h4>{dataset.name}</h4>
                                <p>Type: {dataset.type}</p>
                            </div>
                            <button onClick={() => onSelectDataset(dataset.id, dataset.name)} className="select-item-btn">Select</button>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

const LoadWorkflowModal = ({ onLoadWorkflow, closeModal, onCreateNewWorkflow }) => {
    const enqueueToast = useToast();
    const [workflows, setWorkflows] = useState([]);
    const [isLoading, setIsLoading] = useState(true);

    const fetchWorkflows = useCallback(async () => {
        try {
            setIsLoading(true);
            const response = await fetch('/api/v1/workflows');
            if (!response.ok) throw new Error('Failed to fetch workflows');
            const data = await response.json();
            setWorkflows(data);
        } catch (error) {
            handleApiError(enqueueToast, error, DEFAULT_ERROR_MESSAGE, 'Failed to fetch workflows');
        } finally {
            setIsLoading(false);
        }
    }, [enqueueToast]);

    useEffect(() => {
codex/add-lightweight-toast-mechanism

codex/add-empty-state-handling-for-various-components

        const fetchWorkflows = async () => {
            try {
                setIsLoading(true);
                const response = await fetch('/api/v1/workflows');
                if (!response.ok) throw new Error('Failed to fetch workflows');
                const data = await response.json();
                setWorkflows(data);
            } catch (error) {
                handleApiError(createErrorInfo(error), 'Failed to fetch workflows');
            } finally {
                setIsLoading(false);
            }
        };
main
main
        fetchWorkflows();
    }, [fetchWorkflows]);

    const handleLoad = (workflowId) => {
        onLoadWorkflow(workflowId);
        closeModal();
    };

    const handleCreateNew = () => {
        if (onCreateNewWorkflow) {
            onCreateNewWorkflow();
        }
        closeModal();
    };

    return (
        <div className="configurator-container">
            <div className="item-list">
                {isLoading ? (
                    <p>Loading workflows...</p>
                ) : workflows.length === 0 ? (
                    <div className="empty-state">
                        <h4>No saved workflows found</h4>
                        <p>Try fetching again or start a new workflow.</p>
                        <div className="empty-state-actions">
                            <button onClick={fetchWorkflows} className="select-item-btn">Retry</button>
                            <button onClick={handleCreateNew} className="configure-btn">Create new workflow</button>
                        </div>
                    </div>
                ) : (
                    workflows.map(workflow => (
                        <div key={workflow.id} className="item-card">
                            <div className="item-info">
                                <h4>{workflow.name}</h4>
                            </div>
                            <button onClick={() => handleLoad(workflow.id)} className="select-item-btn">Load</button>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};


const SidebarItem = ({ module, onDragStart }) => (
    <div
        className="sidebar-item"
        draggable="true"
        onDragStart={(e) => onDragStart(e, module.type)}
        role="button"
    >
        <module.icon />
        <span>{module.name}</span>
    </div>
);

const NodeBody = ({ type, data, onDataChange, onConfigure }) => {
    const handleImageUpload = (e) => {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = (event) => {
                onDataChange({ ...data, image: event.target.result });
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    };

    switch (type) {
        case 'image_input':
            return (
                <div className="node-content">
                    {data.image ? (
                        <img src={data.image} alt="Upload preview" className="image-preview" />
                    ) : (
                        <label className="upload-btn">
                            Upload Image
                            <input type="file" accept="image/*" onChange={handleImageUpload} style={{ display: 'none' }} />
                        </label>
                    )}
                </div>
            );
        case 'text_input':
            return (
                <div className="node-content">
                    <textarea 
                        value={data.text || ''} 
                        onChange={(e) => onDataChange({ ...data, text: e.target.value })} 
                        placeholder="Enter text..."
                    />
                </div>
            );
        case 'decision_logic':
             return (
                <div className="node-content">
                    <input 
                        type="text"
                        value={data.condition || ''} 
                        onChange={(e) => onDataChange({ ...data, condition: e.target.value })} 
                        placeholder="e.g., input > 0.5"
                    />
                </div>
            );
        case 'generate_report':
            return (
                <div className="node-content column">
                    <textarea 
                        value={data.prompt || ''} 
                        onChange={(e) => onDataChange({ ...data, prompt: e.target.value })} 
                        placeholder="Prompt template..."
                        rows={3}
                    />
                    {data.report && <div className="report-output">{data.report}</div>}
                </div>
            );
         case 'model_hub':
            return (
                <div className="node-content column">
                    <div className="hub-display">
                        <span>Model:</span>
                        <code>{data.modelId || 'Not Selected'}</code>
                    </div>
                    <button className="configure-btn" onClick={onConfigure}>Configure Model</button>
                </div>
            );
        case 'data_hub':
            return (
                <div className="node-content column">
                    <div className="hub-display">
                        <span>Dataset:</span>
                        <code>{data.datasetName || 'Not Selected'}</code>
                    </div>
                    <button className="configure-btn" onClick={onConfigure}>Configure Dataset</button>
                </div>
            );
        default:
            return null;
    }
};

const Node = ({ node, module, onMouseDown, onPortMouseDown, onNodeDataChange, onConfigureNode }) => {
    return (
        <div
            id={`node-${node.id}`}
            className={`node status-${node.status || 'idle'}`}
            style={{ transform: `translate(${node.x}px, ${node.y}px)` }}
            onMouseDown={(e) => onMouseDown(e, node.id)}
        >
            <div className="node-header">
                <module.icon />
                <span>{module.name}</span>
            </div>
            <div className="node-body">
                 <NodeBody 
                    type={node.type} 
                    data={node.data || {}} 
                    onDataChange={(newData) => onNodeDataChange(node.id, newData)}
                    onConfigure={() => onConfigureNode(node.id, node.type)}
                 />
            </div>
            <div className="node-ports">
                <div className="node-inputs">
                    {module.inputs.map(port => (
                        <div key={port.id} className="port-label-group">
                            <div
                                className="port input-port"
                                data-node-id={node.id}
                                data-port-id={port.id}
                                onMouseDown={(e) => onPortMouseDown(e, node.id, port.id, 'input')}
                            ></div>
                            <span>{port.name}</span>
                        </div>
                    ))}
                </div>
                <div className="node-outputs">
                    {module.outputs.map(port => (
                        <div key={port.id} className="port-label-group">
                            <span>{port.name}</span>
                            <div
                                className="port output-port"
                                data-node-id={node.id}
                                data-port-id={port.id}
                                onMouseDown={(e) => onPortMouseDown(e, node.id, port.id, 'output')}
                            ></div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

const Toolbar = ({ onRun, onSave, onLoad, onClear }) => (
    <div className="canvas-toolbar">
        <button onClick={onRun} className="run-btn"><Icons.Run/> Run</button>
        <button onClick={onSave}><Icons.Save/> Save</button>
        <button onClick={onLoad}><Icons.Load/> Load</button>
        <button onClick={onClear} className="danger-btn"><Icons.Trash/> Clear Canvas</button>
    </div>
);


// --- MAIN APP COMPONENT ---
export const App = () => {
    const [nodes, setNodes] = useState([]);
    const [edges, setEdges] = useState([]);
    const [draggedItem, setDraggedItem] = useState(null);
    const [connectionPreview, setConnectionPreview] = useState(null);
    const [modalState, setModalState] = useState({ isOpen: false, type: null, nodeId: null });
    const [lastRunId, setLastRunId] = useState(null);
    const [toasts, setToasts] = useState([]);

    const canvasRef = useRef(null);
    const toastTimersRef = useRef(new Map());

    const dismissToast = useCallback((id) => {
        setToasts(prev => prev.filter(toast => toast.id !== id));
        const timeoutId = toastTimersRef.current.get(id);
        if (timeoutId) {
            clearTimeout(timeoutId);
            toastTimersRef.current.delete(id);
        }
    }, []);

    const enqueueToast = useCallback(({ tone = 'success', title, description }) => {
        const id = crypto.randomUUID();
        setToasts(prev => [...prev, { id, tone, title, description }]);
        const timeoutId = setTimeout(() => {
            setToasts(prev => prev.filter(toast => toast.id !== id));
            toastTimersRef.current.delete(id);
        }, TOAST_DURATION);
        toastTimersRef.current.set(id, timeoutId);
    }, []);

    useEffect(() => {
        return () => {
            toastTimersRef.current.forEach(clearTimeout);
            toastTimersRef.current.clear();
        };
    }, []);

    const handleDragStart = (e, nodeType) => {
        e.dataTransfer.setData('application/reactflow', nodeType);
        e.dataTransfer.effectAllowed = 'move';
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const canvasBounds = canvasRef.current.getBoundingClientRect();
        const nodeType = e.dataTransfer.getData('application/reactflow');
        if (!nodeType) return;
        
        const position = {
            x: e.clientX - canvasBounds.left,
            y: e.clientY - canvasBounds.top,
        };

        const newNode = {
            id: crypto.randomUUID(),
            type: nodeType,
            status: 'idle',
            data: {},
            ...position
        };
        setNodes(prev => [...prev, newNode]);
    };
    
    const handleNodeMouseDown = (e, nodeId) => {
        if (e.target.closest('.node-body') || e.target.classList.contains('port')) {
             return;
        }
        e.stopPropagation();
        const node = nodes.find(n => n.id === nodeId);
        setDraggedItem({
            type: 'node',
            id: nodeId,
            offsetX: e.clientX - node.x,
            offsetY: e.clientY - node.y
        });
    };

    const handlePortMouseDown = (e, nodeId, portId, portType) => {
        e.stopPropagation();
        if (portType === 'input') return; 
        
        const canvasBounds = canvasRef.current.getBoundingClientRect();
        const startPos = { 
            x: e.clientX - canvasBounds.left, 
            y: e.clientY - canvasBounds.top 
        };

        setConnectionPreview({
            startNodeId: nodeId,
            startPortId: portId,
            ...startPos,
            endX: startPos.x,
            endY: startPos.y,
        });
    };
    
    const handleMouseMove = useCallback((e) => {
        if (draggedItem?.type === 'node') {
            const newX = e.clientX - draggedItem.offsetX;
            const newY = e.clientY - draggedItem.offsetY;
            setNodes(prev => prev.map(n => n.id === draggedItem.id ? {...n, x: newX, y: newY} : n));
        } else if (connectionPreview) {
             const canvasBounds = canvasRef.current.getBoundingClientRect();
            setConnectionPreview(prev => ({
                ...prev, 
                endX: e.clientX - canvasBounds.left, 
                endY: e.clientY - canvasBounds.top 
            }));
        }
    }, [draggedItem, connectionPreview]);

    const handleMouseUp = useCallback((e) => {
        setDraggedItem(null);
        if (connectionPreview) {
            const targetEl = e.target;
            if (targetEl?.classList.contains('input-port')) {
                const endNodeId = targetEl.dataset.nodeId;
                const endPortId = targetEl.dataset.portId;
                const startNodeId = connectionPreview.startNodeId;
                
                if (endNodeId !== startNodeId) {
                    const newEdge = {
                        id: crypto.randomUUID(),
                        fromNode: startNodeId,
                        fromPort: connectionPreview.startPortId,
                        toNode: endNodeId,
                        toPort: endPortId,
                    };
                    setEdges(prev => [...prev, newEdge]);
                }
            }
            setConnectionPreview(null);
        }
    }, [connectionPreview]);
    
    useEffect(() => {
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [handleMouseMove, handleMouseUp]);
    
    const moduleMap = new Map(TOOLBOX_MODULES.map(m => [m.type, m]));

    const handleNodeDataChange = (nodeId, newData) => {
        setNodes(prev => prev.map(n => n.id === nodeId ? { ...n, data: newData } : n));
    };

    const openModal = (type, nodeId = null) => {
        setModalState({ isOpen: true, type, nodeId });
    };

    const closeModal = () => {
        setModalState({ isOpen: false, type: null, nodeId: null });
    };

    const handleConfigureNode = (nodeId, type) => {
        if (type === 'model_hub' || type === 'data_hub') {
            openModal(type, nodeId);
        }
    };
    
    const handleSelectModel = (modelId) => {
        const { nodeId } = modalState;
        const node = nodes.find(n => n.id === nodeId);
        if (node) {
            handleNodeDataChange(nodeId, { ...node.data, modelId });
        }
        closeModal();
    };
    
    const handleSelectDataset = (datasetId, datasetName) => {
        const { nodeId } = modalState;
        const node = nodes.find(n => n.id === nodeId);
        if (node) {
            handleNodeDataChange(nodeId, { ...node.data, datasetId, datasetName });
        }
        closeModal();
    };

    const clearCanvas = () => {
        setNodes([]);
        setEdges([]);
    };

    const handleSaveWorkflow = async () => {
        const name = prompt("Enter a name for your workflow:");
        if (!name || name.trim() === '') {
            enqueueToast({
                tone: 'error',
                title: 'Name required',
                description: 'Workflow name cannot be empty.',
            });
            return;
        }

        const workflowPayload = {
            name,
            nodes: nodes.map(node => {
                const fallbackX = node.x ?? node.position?.x ?? 0;
                const fallbackY = node.y ?? node.position?.y ?? 0;
                return {
                    id: node.id,
                    type: node.type,
                    position: { x: fallbackX, y: fallbackY },
                    data: node.data,
                };
            }),
            edges,
        };

        try {
            const response = await fetch('/api/v1/workflows/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(workflowPayload),
            });
            if (!response.ok) {
                let errorInfo = { message: DEFAULT_ERROR_MESSAGE };
                try {
                    const payload = await response.json();
                    errorInfo = extractErrorInfoFromPayload(payload);
                } catch (parseError) {
                    console.warn('Failed to parse save workflow error payload', parseError);
                }
                handleApiError(errorInfo, 'Failed to save workflow');
                return;
            }
            enqueueToast({
                tone: 'success',
                title: 'All set!',
                description: 'Workflow saved successfully.',
            });
        } catch (error) {
codex/add-lightweight-toast-mechanism
            handleApiError(enqueueToast, error, DEFAULT_ERROR_MESSAGE, 'Failed to save workflow');

            handleApiError(createErrorInfo(error), 'Failed to save workflow');
main
        }
    };

    const handleLoadWorkflow = async (workflowId) => {
        try {
            const response = await fetch(`/api/v1/workflows/${workflowId}`);
            if (!response.ok) {
                let errorInfo = { message: DEFAULT_ERROR_MESSAGE };
                try {
                    const payload = await response.json();
                    errorInfo = extractErrorInfoFromPayload(payload);
                } catch (parseError) {
                    console.warn('Failed to parse load workflow error payload', parseError);
                }
                handleApiError(errorInfo, 'Failed to load workflow');
                return;
            }
            const data = await response.json();
            if (data.nodes && data.edges) {
                const normalizedNodes = data.nodes.map(node => {
                    const { position, ...rest } = node;
                    const fallbackX = node.x ?? position?.x ?? 0;
                    const fallbackY = node.y ?? position?.y ?? 0;
                    return {
                        ...rest,
                        x: fallbackX,
                        y: fallbackY,
                    };
                });
                setNodes(normalizedNodes);
                setEdges(data.edges);
            } else {
                throw new Error('Invalid workflow data received from server.');
            }
        } catch (error) {
codex/add-lightweight-toast-mechanism
            handleApiError(enqueueToast, error, DEFAULT_ERROR_MESSAGE, 'Failed to load workflow');

            handleApiError(createErrorInfo(error), 'Failed to load workflow');
main
        }
    };

    const runWorkflow = async () => {
        const workflowPayload = {
            nodes: nodes.map(node => {
                const fallbackX = node.x ?? node.position?.x ?? 0;
                const fallbackY = node.y ?? node.position?.y ?? 0;
                return {
                    id: node.id,
                    type: node.type,
                    position: { x: fallbackX, y: fallbackY },
                    data: node.data,
                };
            }),
            edges: edges,
        };

        // Provide immediate feedback by setting all nodes to running
        setNodes(prev => prev.map(n => ({...n, status: 'running', data: {...n.data, report: undefined}})));

        try {
            const response = await fetch('/api/v1/workflow/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(workflowPayload),
            });

            if (!response.ok) {
                let errorInfo = { message: DEFAULT_ERROR_MESSAGE };
                try {
                    const payload = await response.json();
                    errorInfo = extractErrorInfoFromPayload(payload);
                } catch (parseError) {
                    console.warn('Failed to parse run workflow error payload', parseError);
                }
                handleApiError(errorInfo, 'Failed to run workflow');
                setNodes(prev => prev.map(n => ({...n, status: 'idle'})));
                setLastRunId(null);
                return;
            }

            const result = await response.json();
            // Update nodes with the final state from the backend
            setNodes(result.nodes);
            if (result.run_id) {
                setLastRunId(result.run_id);
                console.log(`Workflow run ${result.run_id} completed`);
            } else {
                setLastRunId(null);
            }

        } catch (error) {
codex/add-lightweight-toast-mechanism
            handleApiError(enqueueToast, error, DEFAULT_ERROR_MESSAGE, 'Failed to run workflow');

            handleApiError(createErrorInfo(error), 'Failed to run workflow');
main
            // Revert status to idle on failure
            setNodes(prev => prev.map(n => ({...n, status: 'idle'})));
            setLastRunId(null);
        }
    };
    
    const MODAL_TITLES = {
        'model_hub': 'Model Hub',
        'data_hub': 'Data Hub',
        'load_workflow': 'Load Workflow'
    };

    const renderModalContent = () => {
        switch (modalState.type) {
            case 'model_hub':
                return <ModelHubConfigurator onSelectModel={handleSelectModel} />;
            case 'data_hub':
                return <DataHubConfigurator onSelectDataset={handleSelectDataset} />;
            case 'load_workflow':
                return <LoadWorkflowModal onLoadWorkflow={handleLoadWorkflow} closeModal={closeModal} onCreateNewWorkflow={clearCanvas} />;
            default:
                return null;
        }
    };


    return (
        <ToastContext.Provider value={enqueueToast}>
            <>
                <header className="app-header">
                    <h1>Visual Workflow Builder</h1>
                </header>
                <main className="main-container">
                    <aside className="sidebar">
                        <h2>Toolbox</h2>
                        <div className="sidebar-items-container">
                            {TOOLBOX_MODULES.map(module => (
                                <SidebarItem key={module.type} module={module} onDragStart={handleDragStart} />
                            ))}
                        </div>
                    </aside>
                    <section
                        className="canvas-area"
                        ref={canvasRef}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                    >
                        <Toolbar onRun={runWorkflow} onSave={handleSaveWorkflow} onLoad={() => openModal('load_workflow')} onClear={clearCanvas} />
                        {lastRunId && (
                            <div className="run-status">Last run ID: {lastRunId}</div>
                        )}
                        <svg className="edge-layer">
                            {edges.map(edge => {
                           const fromNode = nodes.find(n => n.id === edge.fromNode);
                           const toNode = nodes.find(n => n.id === edge.toNode);
                           if (!fromNode || !toNode) return null;
                           
                           const fromModule = moduleMap.get(fromNode.type);
                           const toModule = moduleMap.get(toNode.type);
                           if(!fromModule || !toModule) return null;

                           const fromPort = fromModule.outputs.find(p => p.id === edge.fromPort);
                           const toPort = toModule.inputs.find(p => p.id === edge.toPort);
                           if (!fromPort || !toPort) return null;

                           const fromPortIndex = fromModule.outputs.indexOf(fromPort);
                           const toPortIndex = toModule.inputs.indexOf(toPort);
                           
                           // --- Improved Edge Rendering Logic ---
                           // These constants are based on the CSS to make calculations more accurate
                           const NODE_HEADER_HEIGHT = 40; // .node-header height
                           const PORT_AREA_TOP_PADDING = 8; // .node-ports top padding
                           const PORT_GROUP_HEIGHT = 14; // Approximate height of a .port-label-group
                           const PORT_GROUP_SPACING = 21; // Approximate vertical distance between port centers

                           const fromNodeEl = document.getElementById(`node-${fromNode.id}`);
                           const fromBodyHeight = fromNodeEl?.querySelector('.node-body')?.clientHeight || 80;

                           const toNodeEl = document.getElementById(`node-${toNode.id}`);
                           const toBodyHeight = toNodeEl?.querySelector('.node-body')?.clientHeight || 80;
                           
                           const fromBaseY = fromNode.y + NODE_HEADER_HEIGHT + fromBodyHeight + PORT_AREA_TOP_PADDING;
                           const fromY = fromBaseY + (PORT_GROUP_HEIGHT / 2) + (fromPortIndex * PORT_GROUP_SPACING);

                           const toBaseY = toNode.y + NODE_HEADER_HEIGHT + toBodyHeight + PORT_AREA_TOP_PADDING;
                           const toY = toBaseY + (PORT_GROUP_HEIGHT / 2) + (toPortIndex * PORT_GROUP_SPACING);
                           
                           const startPos = { x: fromNode.x + 250, y: fromY};
                           const endPos = { x: toNode.x, y: toY };
                           
                           return <path key={edge.id} className="edge" d={getEdgePath(startPos, endPos)} />;
                        })}
                        {connectionPreview && (
                             <path className="edge-preview" d={getEdgePath(
                                { x: connectionPreview.x, y: connectionPreview.y },
                                { x: connectionPreview.endX, y: connectionPreview.endY }
                             )} />
                        )}
                    </svg>
                    {nodes.map(node => {
                        const module = moduleMap.get(node.type);
                        return module ? <Node key={node.id} node={node} module={module} onMouseDown={handleNodeMouseDown} onPortMouseDown={handlePortMouseDown} onNodeDataChange={handleNodeDataChange} onConfigureNode={handleConfigureNode} /> : null;
                    })}
                    </section>
                </main>
                 <Modal isOpen={modalState.isOpen} onClose={closeModal} title={MODAL_TITLES[modalState.type] || 'Configuration'}>
                    {renderModalContent()}
                </Modal>
                <ToastViewport toasts={toasts} onDismiss={dismissToast} />
            </>
        </ToastContext.Provider>
    );
};

if (typeof document !== 'undefined') {
    const container = document.getElementById('root');
    if (container) {
        const root = createRoot(container);
        root.render(<App />);
    }
}