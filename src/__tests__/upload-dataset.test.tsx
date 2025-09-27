import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { App } from '@/index';
import { server } from './setup';
import { addModuleToCanvas } from './test-utils';

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
});
