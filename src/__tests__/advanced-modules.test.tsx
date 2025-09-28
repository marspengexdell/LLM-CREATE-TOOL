import { render, screen, waitFor, within, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { vi, describe, test, beforeEach } from 'vitest';
import { App } from '@/index';
import { server } from './setup';
import { addModuleToCanvas } from './test-utils';

const mockCanvasRect = () => {
  const canvas = document.querySelector('.canvas-area');
  if (!canvas) {
    throw new Error('Canvas not found');
  }
  vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
    x: 0,
    y: 0,
    top: 0,
    left: 0,
    right: 1200,
    bottom: 800,
    width: 1200,
    height: 800,
    toJSON() {
      return {};
    },
  } as DOMRect);
};

describe('advanced workflow modules', () => {
  beforeEach(() => {
    server.use(
      http.get('/api/v1/datasets', () =>
        HttpResponse.json([
          { id: 'dataset-1', name: 'Demo Dataset', type: 'text' },
        ])
      ),
      http.get('/api/v1/models', () =>
        HttpResponse.json([
          { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', description: 'Fast model' },
        ])
      )
    );
  });

  test('configures dataset build and downstream modules', async () => {
    render(<App />);

    await addModuleToCanvas('Dataset Build');
    await addModuleToCanvas('Tokenize Dataset');
    await addModuleToCanvas('Train LoRA (SFT)');
    await addModuleToCanvas('Quantize & Export');
    await addModuleToCanvas('Eval (LMEval)');
    await addModuleToCanvas('Registry Publish');

    const buildConfigure = await screen.findByRole('button', { name: /configure build/i });
    await userEvent.click(buildConfigure);
    const datasetSelect = await screen.findByLabelText(/source dataset/i);
    await userEvent.selectOptions(datasetSelect, 'dataset-1');
    const saveButtons = screen.getAllByRole('button', { name: /save configuration/i });
    await userEvent.click(saveButtons[saveButtons.length - 1]);

    const buildNode = (await screen.findByText('Dataset Build')).closest('.node');
    if (!buildNode) throw new Error('Dataset Build node not found');
    await waitFor(() => {
      expect(within(buildNode).getByText('Demo Dataset')).toBeInTheDocument();
    });

    const tokenizeConfigure = await screen.findByRole('button', { name: /configure tokenizer/i });
    await userEvent.click(tokenizeConfigure);
    const tokenizeDatasetSelect = await screen.findByLabelText(/dataset/i);
    await userEvent.selectOptions(tokenizeDatasetSelect, 'dataset-1');
    const tokenizerInput = screen.getByLabelText(/tokenizer/i);
    await userEvent.clear(tokenizerInput);
    await userEvent.type(tokenizerInput, 'tiktoken');
    await userEvent.click(screen.getAllByRole('button', { name: /save configuration/i }).slice(-1)[0]);

    const tokenizeNode = (await screen.findByText('Tokenize Dataset')).closest('.node');
    if (!tokenizeNode) throw new Error('Tokenize node not found');
    await waitFor(() => {
      expect(within(tokenizeNode).getByText('tiktoken')).toBeInTheDocument();
    });

    const trainConfigure = await screen.findByRole('button', { name: /configure training/i });
    await userEvent.click(trainConfigure);
    const baseModelSelect = await screen.findByLabelText(/base model/i);
    await userEvent.selectOptions(baseModelSelect, 'gemini-1.5-flash');
    const trainDatasetSelect = screen.getByLabelText(/training dataset/i);
    await userEvent.selectOptions(trainDatasetSelect, 'dataset-1');
    const epochsInput = screen.getByLabelText(/epochs/i);
    await userEvent.clear(epochsInput);
    await userEvent.type(epochsInput, '3');
    await userEvent.click(screen.getAllByRole('button', { name: /save configuration/i }).slice(-1)[0]);

    const trainNode = (await screen.findByText('Train LoRA (SFT)')).closest('.node');
    if (!trainNode) throw new Error('Train node not found');
    await waitFor(() => {
      expect(within(trainNode).getByText(/Gemini 1.5 Flash/)).toBeInTheDocument();
      expect(within(trainNode).getByText('3')).toBeInTheDocument();
    });

    const quantizeConfigure = await screen.findByRole('button', { name: /configure quantization/i });
    await userEvent.click(quantizeConfigure);
    const bitsInput = await screen.findByLabelText(/bits/i);
    await userEvent.clear(bitsInput);
    await userEvent.type(bitsInput, '8');
    await userEvent.click(screen.getAllByRole('button', { name: /save configuration/i }).slice(-1)[0]);

    const quantizeNode = (await screen.findByText('Quantize & Export')).closest('.node');
    if (!quantizeNode) throw new Error('Quantize node not found');
    await waitFor(() => {
      expect(within(quantizeNode).getByText('8')).toBeInTheDocument();
    });

    const evalConfigure = await screen.findByRole('button', { name: /configure evaluation/i });
    await userEvent.click(evalConfigure);
    const evalDatasetSelect = await screen.findByLabelText(/evaluation dataset/i);
    await userEvent.selectOptions(evalDatasetSelect, 'dataset-1');
    const benchmarkSelect = screen.getByLabelText(/benchmark/i);
    await userEvent.selectOptions(benchmarkSelect, 'gsm8k');
    await userEvent.click(screen.getAllByRole('button', { name: /save configuration/i }).slice(-1)[0]);

    const evalNode = (await screen.findByText('Eval (LMEval)')).closest('.node');
    if (!evalNode) throw new Error('Eval node not found');
    await waitFor(() => {
      expect(within(evalNode).getByText('GSM8K')).toBeInTheDocument();
    });

    const publishConfigure = await screen.findByRole('button', { name: /configure publish/i });
    await userEvent.click(publishConfigure);
    const visibilitySelect = await screen.findByLabelText(/visibility/i);
    await userEvent.selectOptions(visibilitySelect, 'public');
    await userEvent.click(screen.getAllByRole('button', { name: /save configuration/i }).slice(-1)[0]);

    const publishNode = (await screen.findByText('Registry Publish')).closest('.node');
    if (!publishNode) throw new Error('Publish node not found');
    await waitFor(() => {
      expect(within(publishNode).getByText('public')).toBeInTheDocument();
    });
  });

  test('validates port compatibility and prevents duplicate connections', async () => {
    render(<App />);
    await addModuleToCanvas('Text Input');
    await addModuleToCanvas('Model Hub');
    await addModuleToCanvas('Data Hub');
    await addModuleToCanvas('Train LoRA (SFT)');

    mockCanvasRect();

    const textNode = (await screen.findByText('Text Input')).closest('.node');
    const trainNode = (await screen.findByText('Train LoRA (SFT)')).closest('.node');
    const modelNode = (await screen.findByText('Model Hub')).closest('.node');
    const dataNode = (await screen.findByText('Data Hub')).closest('.node');
    if (!textNode || !trainNode || !modelNode || !dataNode) {
      throw new Error('Required nodes not found');
    }

    const textOutput = textNode.querySelector('.output-port[data-port-id="out"]');
    const trainDatasetInput = trainNode.querySelector('.input-port[data-port-id="dataset"]');
    if (!textOutput || !trainDatasetInput) {
      throw new Error('Ports not found for invalid connection test');
    }

    fireEvent.mouseDown(textOutput, { clientX: 10, clientY: 10 });
    fireEvent.mouseUp(trainDatasetInput, { clientX: 20, clientY: 20 });

    await waitFor(() => {
      expect(document.querySelectorAll('.edge').length).toBe(0);
    });

    const modelOutput = modelNode.querySelector('.output-port[data-port-id="out"]');
    const trainModelInput = trainNode.querySelector('.input-port[data-port-id="model"]');
    if (!modelOutput || !trainModelInput) {
      throw new Error('Ports not found for model connection test');
    }

    fireEvent.mouseDown(modelOutput, { clientX: 30, clientY: 30 });
    fireEvent.mouseUp(trainModelInput, { clientX: 40, clientY: 40 });

    const dataOutput = dataNode.querySelector('.output-port[data-port-id="out"]');
    if (!dataOutput) {
      throw new Error('Data hub output port not found');
    }

    fireEvent.mouseDown(dataOutput, { clientX: 50, clientY: 50 });
    fireEvent.mouseUp(trainDatasetInput, { clientX: 60, clientY: 60 });

    await waitFor(() => {
      expect(document.querySelectorAll('.edge').length).toBe(2);
    });

    fireEvent.mouseDown(dataOutput, { clientX: 70, clientY: 70 });
    fireEvent.mouseUp(trainDatasetInput, { clientX: 80, clientY: 80 });

    await waitFor(() => {
      expect(document.querySelectorAll('.edge').length).toBe(2);
    });
  });
});
