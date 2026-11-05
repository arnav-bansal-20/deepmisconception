/**
 * scratch_populate_ui.js
 * Interacts with the live DeepMisconception UI (http://localhost:3000) using Chrome
 * to author real problems, submit real student code, answer viva questions,
 * and view the synthesized multimodal report purely through browser user actions.
 */
const { chromium } = require('playwright-core');

async function run() {
  console.log('🚀 Launching Chrome to interact with the dashboard...');
  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 960 }
  });
  const page = await context.newPage();

  try {
    // 1. Open Dashboard
    console.log('1. Navigating to http://localhost:3000...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // If currently logged in, log out first to test clean flow
    const logoutBtn = await page.$('button:has-text("Sign Out")');
    if (logoutBtn && await logoutBtn.isVisible()) {
      await logoutBtn.click();
      await page.waitForTimeout(1000);
    }

    // 2. Log in as Student (Arnav Bansal)
    console.log('2. Logging in as Student (Arnav Bansal, 1RV24CS051)...');
    await page.click('button:has-text("Student")');
    await page.waitForTimeout(300);
    await page.fill('input[placeholder="Enter Student ID"]', '1RV24CS051');
    await page.fill('input[placeholder="Enter Full Name"]', 'Arnav Bansal');
    
    // Check classroom selector
    const selectEl = await page.$('select');
    if (selectEl) {
      const options = await page.$$('select option');
      if (options.length > 0) {
        await page.selectOption('select', { index: 0 });
      }
    }
    await page.click('button[type="submit"]');
    await page.waitForTimeout(2000);
    console.log('   -> Logged into student learning portal.');

    // 3. Navigate to Workspace tab
    console.log('3. Navigating to Workspace tab...');
    await page.click('nav button:has-text("Workspace")');
    await page.waitForTimeout(2000);

    // 4. Check if problem catalog is visible and launch problem
    const startProblemBtn = await page.$('button:has-text("Start Problem")');
    if (startProblemBtn && await startProblemBtn.isVisible()) {
      console.log('4. Launching problem from catalog...');
      await startProblemBtn.click();
      await page.waitForTimeout(2000);
    } else {
      console.log('4. Problem workspace already active.');
    }

    // 5. Insert Python code into Monaco Editor
    console.log('5. Inserting Python solution into Monaco Editor...');
    const pythonCode = `def sum_of_squares_skipping_threes(n: int) -> int:
    total = 0
    for i in range(1, n + 1):
        if i % 3 == 0:
            continue
        total += i * i
    return total

result = sum_of_squares_skipping_threes(5)
print(f"Computed total: {result}")
`;

    await page.click('.monaco-editor');
    await page.keyboard.press('Meta+A');
    await page.keyboard.press('Backspace');
    await page.waitForTimeout(300);
    await page.keyboard.insertText(pythonCode);
    await page.waitForTimeout(1500);

    // 6. Run Code in Sandbox
    console.log('6. Running code in sandbox via Workspace UI...');
    await page.click('button:has-text("Run Code")');
    await page.waitForTimeout(3500);

    // 7. Submit Code for Viva
    console.log('7. Submitting code snapshot for AI Viva questioning...');
    const submitBtn = await page.$('button:has-text("Submit for Viva")');
    if (submitBtn && await submitBtn.isVisible()) {
      await submitBtn.click();
      await page.waitForTimeout(4000);
    }

    // 8. Handle AI Viva Interface
    console.log('8. Interacting with AI Viva interface...');
    // If not automatically switched to Viva, click Viva Verification or wait
    const genQuestionsBtn = await page.$('button:has-text("Generate Dynamic Questions (AI)")');
    if (genQuestionsBtn && await genQuestionsBtn.isVisible()) {
      console.log('   -> Clicking "Generate Dynamic Questions (AI)" to synthesize viva questions...');
      await genQuestionsBtn.click();
      console.log('   -> Waiting for AI questions to be generated...');
      await page.waitForTimeout(10000);
    }

    // Wait for viva question textarea
    await page.waitForSelector('textarea', { timeout: 25000 });
    console.log('   -> Viva questions loaded! Answering Question 1...');

    const vivaAnswer1 = 'The variable total maintains the accumulator state. In each iteration of the loop from 1 to n, the condition i % 3 == 0 checks if i is a multiple of 3. If so, continue skips the squaring step, ensuring only non-multiples of 3 are squared and added.';
    await page.fill('textarea', vivaAnswer1);
    await page.waitForTimeout(500);
    await page.click('button:has-text("Submit Viva Explanation")');
    console.log('   -> Waiting for AI assessment of Question 1...');
    await page.waitForTimeout(8000);

    // Check Question 2 in sidebar
    const questionButtons = await page.$$('.lg\\:col-span-4 .space-y-2 > div');
    if (questionButtons.length > 1) {
      console.log('   -> Selecting Question 2 from sidebar...');
      await questionButtons[1].click();
      await page.waitForTimeout(1500);

      const textarea2 = await page.$('textarea');
      if (textarea2 && await textarea2.isVisible()) {
        const vivaAnswer2 = 'The range(1, n + 1) generates integers starting at 1 and stopping before n + 1, including n. The % operator calculates integer remainder, which is 0 exclusively when i is divisible by 3.';
        await textarea2.fill(vivaAnswer2);
        await page.waitForTimeout(500);
        await page.click('button:has-text("Submit Viva Explanation")');
        console.log('   -> Waiting for AI assessment of Question 2...');
        await page.waitForTimeout(8000);
      }
    }

    // 9. Synthesize / View Comprehensive Diagnosis Report
    console.log('9. Synthesizing & Viewing Multimodal Comprehension Report...');
    const viewReportBtn = await page.waitForSelector('button:has-text("View Comprehension Report")', { timeout: 15000 });
    if (viewReportBtn) {
      await viewReportBtn.click();
      await page.waitForTimeout(6000);
    }

    // Capture full-page screenshot of the student diagnostic report
    await page.screenshot({ path: 'scratch_report_screenshot.png', fullPage: true });
    console.log('   -> Student report screenshot captured: scratch_report_screenshot.png');

    // 10. Check Instructor view
    console.log('10. Logging into Instructor portal to verify instructor analytics...');
    await page.click('button:has-text("Sign Out")');
    await page.waitForTimeout(1000);

    await page.click('button:has-text("Teacher / Faculty")');
    await page.fill('input[placeholder="Enter Faculty ID"]', 'FAC-001');
    await page.fill('input[placeholder="Enter Faculty Name"]', 'Dr. Ada Smith');
    await page.fill('input[placeholder="Enter Department"]', 'Computer Science & Engineering');
    await page.click('button[type="submit"]');
    await page.waitForTimeout(3000);

    // Capture instructor view screenshot
    await page.screenshot({ path: 'scratch_instructor_screenshot.png', fullPage: true });
    console.log('   -> Instructor dashboard screenshot captured: scratch_instructor_screenshot.png');

    console.log('\n🎉 ALL ACTIONS COMPLETED SUCCESSFULLY THROUGH THE DASHBOARD UI!');
  } catch (err) {
    console.error('Error during dashboard automation:', err);
    await page.screenshot({ path: 'scratch_error_screenshot.png' });
  } finally {
    await browser.close();
  }
}

run();
