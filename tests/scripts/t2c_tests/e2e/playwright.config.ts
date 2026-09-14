import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  timeout: 180_000,               // LLM + Cypher can be slow
  expect: { timeout: 120_000 },
  retries: 1,
  workers: 1,                     // serialise - most chat backends rate-limit
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.CHAT_URL ?? 'http://localhost:3000',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure',
  },
});
