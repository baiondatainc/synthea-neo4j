/**
 * UI smoke tests for the IA text2cypher chat.
 * Purpose: prove the UI end-to-end works for a representative subset.
 * Correctness of every question is covered by the pytest suite (API layer).
 *
 * Selectors come from env so a UI change is a one-line fix:
 *   CHAT_INPUT_SELECTOR     e.g. textarea, [data-testid=chat-input]
 *   CHAT_SEND_SELECTOR      e.g. button[type=submit]
 *   CHAT_RESPONSE_SELECTOR  e.g. [data-role=assistant], .assistant-message
 */
import { test, expect, Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

const INPUT = process.env.CHAT_INPUT_SELECTOR ?? 'textarea';
const SEND = process.env.CHAT_SEND_SELECTOR ?? 'button[type=submit]';
const RESPONSE = process.env.CHAT_RESPONSE_SELECTOR ?? '[data-role=assistant]';

// Representative subset: one per category + the two that prove the model.
const SMOKE_IDS = ['Q01', 'Q05', 'Q19', 'Q27', 'Q31', 'Q34', 'Q37', 'Q43', 'Q44', 'Q56'];

type Q = { id: string; category: string; question: string; compare: string };

function loadQuestions(): Q[] {
  const file = path.resolve(__dirname, '..', 'questions.yaml');
  const doc = yaml.load(fs.readFileSync(file, 'utf8')) as any;
  const values: Record<string, string> = doc.values ?? {};
  return (doc.questions as Q[])
    .filter(q => SMOKE_IDS.includes(q.id))
    .map(q => ({ ...q, question: q.question.replace(/\$\{(\w+)\}/g, (_, k) => String(values[k])) }));
}

async function ask(page: Page, question: string): Promise<string> {
  const before = await page.locator(RESPONSE).count();
  await page.locator(INPUT).fill(question);
  await page.locator(SEND).click();
  // wait for a new assistant message to appear...
  await expect(page.locator(RESPONSE)).toHaveCount(before + 1);
  const msg = page.locator(RESPONSE).nth(before);
  // ...and for streaming to settle (text unchanged for 2s)
  let last = '';
  for (let i = 0; i < 60; i++) {
    const now = (await msg.innerText()).trim();
    if (now && now === last) break;
    last = now;
    await page.waitForTimeout(2000);
  }
  return last;
}

test.describe('IA chat - smoke', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator(INPUT)).toBeVisible();
  });

  for (const q of loadQuestions()) {
    test(`${q.id} ${q.question}`, async ({ page }) => {
      const text = await ask(page, q.question);

      // Generic assertions - loose on purpose, LLM wording varies run to run.
      expect(text.length, 'empty response').toBeGreaterThan(0);
      expect(text, 'backend error surfaced in chat').not.toMatch(/error|exception|traceback|failed to/i);

      if (q.compare === 'count') {
        expect(text, 'count answer should contain a number').toMatch(/\d/);
      }
      if (q.compare === 'rows' || q.compare === 'topk') {
        // either a rendered table or at least several lines of results
        const hasTable = (await page.locator(`${RESPONSE} table`).count()) > 0;
        expect(hasTable || text.split('\n').length >= 2, 'expected table or list').toBeTruthy();
      }
      if (q.id === 'Q31') {
        // model integrity: orphan claims must be zero
        expect(text).toMatch(/\b(0|zero|no claims)\b/i);
      }
      if (q.id === 'Q34') {
        expect(text).toContain('CTM5DQ3K');
      }
    });
  }

  test('multi-turn follow-up keeps context', async ({ page }) => {
    await ask(page, 'How many claims?');
    const follow = await ask(page, 'Of those, how many are fraud-flagged?');
    expect(follow).toMatch(/\d/);
  });
});
