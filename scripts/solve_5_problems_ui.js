/**
 * scripts/solve_5_problems_ui.js
 * Logs in as student Arnav Bansal in classroom CS101-ALPHA,
 * solves 5 different coding problems via the interactive dashboard UI,
 * intentionally introducing varied errors in code and viva explanations,
 * generating diverse diagnostic outcomes:
 *   - Possible Misconception (Problems 1 & 2)
 *   - Partial Understanding (Problem 3)
 *   - Understood (Problems 4 & 5)
 */
const { chromium } = require('playwright-core');

const tasks = [
  {
    name: 'Problem 1: Recursive Factorial (Missing Base Case -> Misconception)',
    title: 'Recursive Factorial with Non-Negative Invariant',
    code: `def factorial(n: int) -> int:
    # Missing base case: infinite recursion on factorial(0) or n <= 0
    return n * factorial(n - 1)

print(factorial(5))
`,
    vivaAnswer: "Recursion does not require an explicit base case in Python because the execution call stack automatically terminates when the counter parameter decrements to zero without a base case.",
    targetDiagnosis: 'Possible Misconception'
  },
  {
    name: 'Problem 2: List Partitioning (Mutable Reference Aliasing -> Misconception)',
    title: 'List Partitioning and Mutable Aliasing',
    code: `def split_evens_odds(numbers: list[int]) -> tuple[list[int], list[int]]:
    evens = numbers
    odds = numbers
    for x in numbers:
        if x % 2 == 0:
            odds.remove(x)
    return (evens, odds)

print(split_evens_odds([1, 2, 3, 4, 5, 6]))
`,
    vivaAnswer: "Assigning evens = numbers creates two completely independent copies in memory, so removing items from odds has no effect on evens.",
    targetDiagnosis: 'Possible Misconception'
  },
  {
    name: 'Problem 3: Palindrome Validator (Minor Normalization Flaw -> Partial Understanding)',
    title: 'Palindrome Validator with Slicing Invariant',
    code: `def is_palindrome(text: str) -> bool:
    cleaned = text.replace(" ", "")
    return cleaned == cleaned[::-1]

print(is_palindrome("Race car"))
`,
    vivaAnswer: "The slice [::-1] steps backwards through the string to reverse it, but I didn't lowercase the characters so it fails on mixed case words like Race car.",
    targetDiagnosis: 'Partial Understanding'
  },
  {
    name: 'Problem 4: Fibonacci Sequence Memoization (Flawless -> Understood)',
    title: 'Fibonacci Sequence Memoization',
    code: `def fibonacci(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 0:
        return 0
    if n == 1:
        return 1
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

result = fibonacci(10)
print(f"Computed total: {result}")
`,
    vivaAnswer: "The memo dictionary caches calculated subproblems. Before recursive branching, we look up n in O(1) time. The base cases n<=0 returning 0 and n==1 returning 1 anchor the recursion, preventing redundant exponential computations and reducing time complexity to O(n).",
    targetDiagnosis: 'Understood'
  },
  {
    name: 'Problem 5: Binary Search with Strict Bounds (Rigorous Reasoning -> Understood)',
    title: 'Binary Search with Strict Monotonic Bounds',
    code: `def binary_search(arr: list[int], target: int) -> int:
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
result = binary_search(arr, 23)
print(f"Index found: {result}")
`,
    vivaAnswer: "Binary search maintains the search invariant across the closed interval [low, high]. At each iteration, calculating mid = (low + high) // 2 splits the monotonic sequence in half. Updating low = mid + 1 or high = mid - 1 guarantees strict progress, converging in O(log n) time.",
    targetDiagnosis: 'Understood'
  }
];

async function solveProblems() {
  console.log('🚀 Launching Chrome to solve 5 problems via Student Learning Portal...');
  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 960 }
  });
  const page = await context.newPage();

  try {
    // 1. Open Dashboard & Clean Login
    console.log('1. Navigating to http://localhost:3000...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    const logoutBtn = await page.$('button:has-text("Sign Out")');
    if (logoutBtn && await logoutBtn.isVisible()) {
      await logoutBtn.click();
      await page.waitForTimeout(1000);
    }

    // 2. Log in as Student (Arnav Bansal, 1RV24CS051, Group: CS101-ALPHA)
    console.log('2. Logging in as Student Arnav Bansal (1RV24CS051) under CS101-ALPHA...');
    await page.click('button:has-text("Student")');
    await page.waitForTimeout(300);
    await page.fill('input[placeholder="Enter Student ID"]', '1RV24CS051');
    await page.fill('input[placeholder="Enter Full Name"]', 'Arnav Bansal');

    // Select Classroom CS101-ALPHA
    const selectClass = await page.$('select');
    if (selectClass) {
      await page.selectOption('select', { index: 0 });
    }
    await page.click('button[type="submit"]');
    await page.waitForTimeout(2500);
    console.log('   -> Logged in successfully!');

    // Loop through the 5 tasks
    for (let t = 0; t < tasks.length; t++) {
      const task = tasks[t];
      console.log(`\n======================================================`);
      console.log(`[Task ${t + 1}/5] ${task.name}`);
      console.log(`Targeting Diagnosis: ${task.targetDiagnosis}`);
      console.log(`======================================================`);

      // 1. Navigate to Workspace tab
      await page.click('nav button:has-text("Workspace")');
      await page.waitForTimeout(1000);

      // Check if we need to open the catalog:
      const changeProblemBtn = await page.$('button:has-text("Change Problem")');
      if (changeProblemBtn && await changeProblemBtn.isVisible()) {
        console.log('   -> Clicking "Change Problem" to open catalog...');
        await changeProblemBtn.click();
        await page.waitForTimeout(1000);
      }

      const startAnotherBtn = await page.$('button:has-text("Start Another Problem")');
      if (startAnotherBtn && await startAnotherBtn.isVisible()) {
        console.log('   -> Clicking "Start Another Problem" to open catalog...');
        await startAnotherBtn.click();
        await page.waitForTimeout(1000);
      }

      // Now wait for the problem card for task.title to be visible in the catalog
      console.log(`   -> Locating problem card for "${task.title}"...`);
      const specificStartBtn = page.locator(`//h3[contains(text(), "${task.title}")]/ancestor::div[contains(@class, "rounded-2xl")]//button[contains(., "Start Problem")]`);
      await specificStartBtn.first().waitFor({ state: 'visible', timeout: 15000 });
      console.log(`   -> Launching "${task.title}" from catalog...`);
      await specificStartBtn.first().click();
      await page.waitForTimeout(2000);

      // Wait for Monaco Editor
      await page.waitForSelector('.monaco-editor', { timeout: 15000 });
      console.log('   -> Inserting solution code into Monaco editor...');
      await page.click('.monaco-editor');
      await page.keyboard.press('Meta+A');
      await page.keyboard.press('Backspace');
      await page.waitForTimeout(300);
      await page.keyboard.insertText(task.code);
      await page.waitForTimeout(1000);

      // Run Code in sandbox
      console.log('   -> Running code in sandbox...');
      await page.click('button:has-text("Run Code")');
      await page.waitForTimeout(3000);

      // Submit Code for Viva
      console.log('   -> Submitting code for Viva...');
      await page.click('button:has-text("Submit for Viva")');
      await page.waitForTimeout(2000);

      // Wait for either the "Generate Dynamic Questions (AI)" button OR textarea to appear
      console.log('   -> Waiting for Viva interface to load...');
      await page.waitForSelector('button:has-text("Generate Dynamic Questions (AI)"), textarea', { timeout: 25000 });

      const genQuestionsBtn = await page.$('button:has-text("Generate Dynamic Questions (AI)")');
      if (genQuestionsBtn && await genQuestionsBtn.isVisible()) {
        console.log('      -> Generating dynamic questions...');
        await genQuestionsBtn.click();
        await page.waitForTimeout(8000);
      }

      // Wait for question textarea
      await page.waitForSelector('textarea', { timeout: 25000 });
      console.log('      -> Answering Viva question...');
      await page.fill('textarea', task.vivaAnswer);
      await page.waitForTimeout(500);
      await page.click('button:has-text("Submit Viva Explanation")');
      console.log('      -> Waiting for AI assessment...');
      await page.waitForTimeout(7000);

      // View Report
      console.log('   -> Synthesizing Multimodal Diagnosis Report...');
      const viewReportBtn = await page.waitForSelector('button:has-text("View Comprehension Report")', { timeout: 20000 });
      if (viewReportBtn) {
        await viewReportBtn.click();
        await page.waitForTimeout(5000);
      }

      // Read resulting verdict on screen
      const verdict = await page.evaluate(() => {
        const badges = document.querySelectorAll('span, strong, h3, div');
        for (const b of badges) {
          const txt = b.textContent.trim();
          if (txt === 'Understood' || txt === 'Possible Misconception' || txt === 'Partial Understanding') {
            return txt;
          }
        }
        return 'Report Generated';
      });
      console.log(`   -> Diagnosed Verdict: "${verdict}"`);

      // Capture screenshot of each report
      const screenshotName = `scratch_report_task_${t + 1}.png`;
      await page.screenshot({ path: screenshotName, fullPage: true });
      console.log(`   -> Captured screenshot: ${screenshotName}`);
    }

    console.log('\n🎉 ALL 5 PROBLEMS COMPLETED THROUGH DASHBOARD UI!');
  } catch (err) {
    console.error('Error during 5 problems execution:', err);
    await page.screenshot({ path: 'scratch_error_5tasks.png' });
  } finally {
    await browser.close();
  }
}

solveProblems();
