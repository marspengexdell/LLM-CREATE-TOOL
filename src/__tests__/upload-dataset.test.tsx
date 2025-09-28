import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { App } from '@/index';
import { server } from './setup';
import { addModuleToCanvas } from './test-utils';
import { vi } from 'vitest';

describe('upload dataset flow', () => {
  test('uploads a new dataset and refreshes the list', async () => {
    let uploadCount = 0;

    server.use(
      http.get('/api/v1/datasets', () => {
        if (uploadCount > 0) {
          return HttpResponse.json([
            {
              id: 'dataset-1',
              name: 'Uploaded Dataset',
              type: 'image',
              preview: '/preview.png',
            },
          ]);
        }
        return HttpResponse.json([]);
      }),
      http.get('/api/v1/metadata', () =>
        HttpResponse.json({ datasetUpload: { maxBytes: 200 * 1024 * 1024 } })
      ),
      http.post('/api/v1/datasets/upload', async () => {
        uploadCount += 1;
        return HttpResponse.json({ success: true });
      })
    );

    render(<App />);
    await addModuleToCanvas('Data Hub');

    const configureButton = await screen.findByRole('button', { name: /configure dataset/i });
    await userEvent.click(configureButton);

    const modalHeading = await screen.findByRole('heading', { name: 'Data Hub' });
    const modal = modalHeading.closest('.modal-content');
    if (!modal) {
      throw new Error('Modal container not found');
    }

    const fileInput = within(modal).getByLabelText('Dataset file input');
    const file = new File(['id,value\n1,foo'], 'dataset.csv', { type: 'text/csv' });
    await userEvent.upload(fileInput, file);

    await screen.findByText('Uploaded Dataset');
  });

  test('prevents uploading files that exceed the configured limit', async () => {
    const alertMock = vi.spyOn(window, 'alert').mockImplementation(() => {});
    let uploadCount = 0;

    server.use(
      http.get('/api/v1/datasets', () => HttpResponse.json([])),
      http.get('/api/v1/metadata', () =>
        HttpResponse.json({ datasetUpload: { maxBytes: 5 * 1024 * 1024 } })
      ),
      http.post('/api/v1/datasets/upload', () => {
        uploadCount += 1;
        return HttpResponse.json({ success: true });
      })
    );

    render(<App />);
    await addModuleToCanvas('Data Hub');

    const configureButton = await screen.findByRole('button', { name: /configure dataset/i });
    await userEvent.click(configureButton);

    const modalHeading = await screen.findByRole('heading', { name: 'Data Hub' });
    const modal = modalHeading.closest('.modal-content');
    if (!modal) {
      throw new Error('Modal container not found');
    }

    const fileInput = within(modal).getByLabelText('Dataset file input');
    const oversizedFile = new File([new Uint8Array(6 * 1024 * 1024)], 'large.bin', {
      type: 'application/octet-stream',
    });

    await userEvent.upload(fileInput, oversizedFile);

    await waitFor(() => {
      expect(alertMock).toHaveBeenCalledWith('File exceeds maximum allowed size of 5 MB.');
    });
    expect(uploadCount).toBe(0);
  });
});
