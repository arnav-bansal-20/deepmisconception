const { chromium } = require('playwright-core');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 950 } });
  const page = await context.newPage();

  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  // Check if on login page
  const studentBtn = page.locator('button:has-text("Student")');
  if (await studentBtn.isVisible()) {
    console.log('Logging in as student Arnav Bansal...');
    await studentBtn.click();
    await page.waitForTimeout(300);

    const rollInput = page.locator('input[placeholder="Enter Student ID"]');
    await rollInput.fill('1RV24CS051');

    const nameInput = page.locator('input[placeholder="Enter Full Name"]');
    await nameInput.fill('Arnav Bansal');

    // Submit login
    await page.locator('button[type="submit"]:has-text("Enter Guided Workspace")').click();
    await page.waitForTimeout(1500);
  }

  // Ensure on session history tab
  const historyTab = page.locator('button:has-text("My Coding Sessions")');
  if (await historyTab.isVisible()) {
    await historyTab.click();
    await page.waitForTimeout(1000);
  }

  const screenshotPath = '/Users/arnavbansal/.gemini/antigravity/brain/98ae6918-36f8-411b-8d12-1f8d44ec6c79/student_sessions_no_delete.png';
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`Saved screenshot to ${screenshotPath}`);

  await browser.close();
})();
