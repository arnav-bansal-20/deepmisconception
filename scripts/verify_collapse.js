const { chromium } = require('playwright-core');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 950 } });
  const page = await context.newPage();

  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  const teacherBtn = page.locator('button:has-text("Teacher / Faculty")');
  if (await teacherBtn.isVisible()) {
    await teacherBtn.click();
    await page.locator('input[placeholder="Enter Faculty ID"]').fill('FAC-001');
    await page.locator('input[placeholder="Enter Faculty Name"]').fill('Dr. Ada Smith');
    await page.locator('input[placeholder="Enter Department"]').fill('Computer Science & Engineering');
    await page.locator('button[type="submit"]:has-text("Open Instructor Studio")').click();
    await page.waitForTimeout(2000);
  }

  // Click student header to toggle collapse
  const studentHeader = page.locator('div:has-text("Arnav Bansal"):has-text("1RV24CS051")').first();
  await studentHeader.click();
  await page.waitForTimeout(500);

  const collapsedScreenshotPath = '/Users/arnavbansal/.gemini/antigravity/brain/98ae6918-36f8-411b-8d12-1f8d44ec6c79/clubbed_student_collapsed.png';
  await page.screenshot({ path: collapsedScreenshotPath, fullPage: true });
  console.log(`Saved collapsed screenshot to ${collapsedScreenshotPath}`);

  await browser.close();
})();

