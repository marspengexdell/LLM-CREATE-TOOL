import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { App } from '@/index';
import { server } from './setup';
import { addModuleToCanvas } from './test-utils';

describe('run workflow flow', () => {
  test('runs the workflow and displays the run identifier', async () => {
    let runPayload: any = null;

    server.use(
      http.post('/api/v1/workflow/run', async ({ request }) => {
        runPayload = await request.json();
        const responseNodes = (runPayload?.nodes ?? []).map((node: any) => ({
          ...node,
          status: 'completed',
        }));
        return HttpResponse.json({
          run_id: 'run-123',
          nodes: responseNodes,
          edges: runPayload?.edges ?? [],
        });
      })
    );

    render(<App />);
    await addModuleToCanvas('Text Input');

    const runButton = screen.getByRole('button', { name: /Run/i });
    await userEvent.click(runButton);

    await screen.findByText(/Last run ID: run-123/i);

    await waitFor(() => {
      expect(runPayload?.nodes).toHaveLength(1);
      const completedNode = document.querySelector('.node.status-completed');
      expect(completedNode).not.toBeNull();
    });
  });
});
