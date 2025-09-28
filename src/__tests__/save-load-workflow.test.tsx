import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { vi } from 'vitest';
import { App } from '@/index';
import { server } from './setup';
import { addModuleToCanvas } from './test-utils';

describe('save and load workflow flow', () => {
  test('saves the current workflow and loads a saved workflow', async () => {
    const promptMock = vi.spyOn(window, 'prompt').mockReturnValue('My workflow');
    const alertMock = vi.spyOn(window, 'alert').mockImplementation(() => {});

    let savedPayload: any = null;
    let runPayload: any = null;

    server.use(
      http.post('/api/v1/workflows/save', async ({ request }) => {
        savedPayload = await request.json();
        return HttpResponse.json({ id: 'workflow-1' });
      }),
      http.get('/api/v1/workflows', () =>
        HttpResponse.json([
          {
            id: 'workflow-1',
            name: 'Workflow 1',
          },
        ])
      ),
      http.get('/api/v1/workflows/workflow-1', () =>
        HttpResponse.json({
          nodes: [
            {
              id: 'node-loaded',
              type: 'data_hub',
              position: { x: 120, y: 120 },
              data: { datasetName: 'Loaded dataset' },
            },
          ],
          edges: [],
        })
      ),
      http.post('/api/v1/workflow/run', async ({ request }) => {
        runPayload = await request.json();
        return HttpResponse.json({
          nodes: [
            {
              id: 'node-loaded',
              type: 'data_hub',
              x: 120,
              y: 120,
              data: { datasetName: 'Loaded dataset' },
              status: 'idle',
            },
          ],
          edges: [],
          run_id: 'run-123',
        });
      })
    );

    render(<App />);
    await addModuleToCanvas('Text Input');

    const toolbar = document.querySelector('.canvas-toolbar') as HTMLElement;
    const saveButton = within(toolbar).getByRole('button', { name: /^Save$/i });
    await userEvent.click(saveButton);

    await waitFor(() => {
      expect(promptMock).toHaveBeenCalled();
      expect(alertMock).toHaveBeenCalledWith('Workflow saved successfully!');
      expect(savedPayload?.nodes).toHaveLength(1);
    });

    const loadButton = within(toolbar).getByRole('button', { name: /^Load$/i });
    await userEvent.click(loadButton);

    const modalHeading = await screen.findByRole('heading', { name: 'Load Workflow' });
    const modal = modalHeading.closest('.modal-content');
    if (!modal) {
      throw new Error('Modal container not found');
    }

    const workflowLoadButton = within(modal).getByRole('button', { name: /^Load$/i });
    await userEvent.click(workflowLoadButton);

    await screen.findByText('Loaded dataset');

    const runButton = within(toolbar).getByRole('button', { name: /^Run$/i });
    await userEvent.click(runButton);

    await waitFor(() => {
      expect(runPayload?.nodes?.[0]?.position).toEqual({ x: 120, y: 120 });
    });
  });
});
