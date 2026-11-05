const { chromium } = require('playwright-core');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 950 } });
  const page = await context.newPage();

  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  // Check if on login page
  const teacherBtn = page.locator('button:has-text("Teacher / Faculty")');
  if (await teacherBtn.isVisible()) {
    console.log('Clicking Teacher / Faculty role...');
    await teacherBtn.click();
    await page.waitForTimeout(400);

    // Faculty ID
    const idInput = page.locator('input[placeholder="Enter Faculty ID"]');
    await idInput.fill('FAC-001');

    // Faculty Name
    const nameInput = page.locator('input[placeholder="Enter Faculty Name"]');
    await nameInput.fill('Dr. Ada Smith');

    // Department
    const deptInput = page.locator('input[placeholder="Enter Department"]');
    await deptInput.fill('Computer Science & Engineering');

    // Submit login
    await page.locator('button[type="submit"]:has-text("Open Instructor Studio")').click();
    await page.waitForTimeout(2000);
  }

  // Ensure on reports tab
  const reportsTab = page.locator('button:has-text("Student Comprehension Reports")');
  if (await reportsTab.isVisible()) {
    await reportsTab.click();
    await page.waitForTimeout(1000);
  }

  const screenshotPath = '/Users/arnavbansal/.gemini/antigravity/brain/98ae6918-36f8-411b-8d12-1f8d44ec6c79/clubbed_student_view.png';
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`Saved screenshot to ${screenshotPath}`);

  await browser.close();
})();
