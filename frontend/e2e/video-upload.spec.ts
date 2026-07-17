import { expect, test } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const VIDEO_FIXTURE = path.resolve(__dirname, '../../test_videos/rachel_32frames.mp4')

test.describe('video upload + job status', () => {
  test('selects a video and starts upload', async ({ page }) => {
    await page.goto('/videos')
    await expect(page.getByRole('heading', { name: 'Video Tanıma' })).toBeVisible()

    await page.setInputFiles('[data-testid="video-dropzone-input"]', VIDEO_FIXTURE)
    await expect(page.getByText('rachel_32frames.mp4')).toBeVisible()

    await page.getByTestId('upload-video-button').click()
    await expect(page).toHaveURL(/\/videos\/[0-9a-f-]+\/jobs\/[0-9a-f-]+/)

    const statusBadge = page.locator('span').filter({ hasText: /Bekliyor|İşleniyor|Tamamlandı|Hata|İptal Edildi/ }).first()
    await expect(statusBadge).toBeVisible()
  })

  test('shows unavailable overlay placeholder until the job completes', async ({ page }) => {
    await page.goto('/videos')
    await page.setInputFiles('[data-testid="video-dropzone-input"]', VIDEO_FIXTURE)
    await page.getByTestId('upload-video-button').click()
    await expect(page).toHaveURL(/\/videos\/[0-9a-f-]+\/jobs\/[0-9a-f-]+/)

    await expect(page.getByText('Video hazır değil')).toBeVisible()
  })
})
