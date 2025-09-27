import { fireEvent, screen } from '@testing-library/react';

type MutableDataTransfer = {
  data: Record<string, string>;
  setData: (type: string, value: string) => void;
  getData: (type: string) => string;
  clearData: () => void;
  dropEffect: string;
  effectAllowed: string;
  files: File[];
  items: DataTransferItem[];
  types: string[];
};

export const createDataTransfer = (): DataTransfer => {
  const store: Record<string, string> = {};
  const dataTransfer: Partial<MutableDataTransfer> = {
    data: store,
    setData: (type: string, value: string) => {
      store[type] = value;
    },
    getData: (type: string) => store[type] ?? '',
    clearData: () => {
      Object.keys(store).forEach((key) => delete store[key]);
    },
    dropEffect: 'move',
    effectAllowed: 'all',
    files: [],
    items: [] as unknown as DataTransferItem[],
    types: [],
  };

  return dataTransfer as DataTransfer;
};

export const addModuleToCanvas = async (label: string) => {
  const module = await screen.findByText(label);
  const canvas = document.querySelector('.canvas-area') as HTMLElement;
  if (!canvas) {
    throw new Error('Canvas element not found');
  }

  const dataTransfer = createDataTransfer();
  fireEvent.dragStart(module, { dataTransfer });
  fireEvent.dragOver(canvas, { dataTransfer });
  fireEvent.drop(canvas, { dataTransfer });
};
