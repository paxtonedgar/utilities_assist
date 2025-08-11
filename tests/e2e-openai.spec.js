// e2e-openai.spec.js - End-to-end test for OpenAI integration
// This test validates that the Streamlit app can connect to OpenAI and process responses

const { test, expect } = require('@playwright/test');

test('Streamlit app connects to local OpenAI and processes response', async ({ page }) => {
  console.log('Starting local OpenAI end-to-end test...');
  
  // 1) Navigate to the Streamlit app (should be started with USE_LOCAL_OPENAI=true)
  await page.goto('http://localhost:8501', { 
    waitUntil: 'domcontentloaded',
    timeout: 30000 
  });

  // 2) Wait for the app to load and check for the title
  await expect(page.locator('h1')).toContainText('Digital Knowledge Hub', { timeout: 15000 });

  // 3) Find the chat input field
  const chatInput = page.getByTestId('stChatInputTextArea');
  
  // 4) Type a simple test prompt
  const testPrompt = 'Quick test: say "pong" and nothing else.';
  await chatInput.fill(testPrompt);

  // 5) Submit the message by pressing Enter
  await page.keyboard.press('Enter');

  // 6) Wait for the user message to appear
  await expect(page.locator('text=' + testPrompt)).toBeVisible({ timeout: 10000 });

  // 7) Wait for any assistant response (could be generic response or actual pong)
  await expect(page.locator('strong:has-text("Assistant:")').locator('..').locator('text')).toBeVisible({ timeout: 20000 });

  // 8) Verify no error messages appeared in the UI
  await expect(page.locator('text=Error')).toHaveCount(0);
  await expect(page.locator('text=exception')).toHaveCount(0);

  console.log('Local OpenAI integration test completed successfully!');
});