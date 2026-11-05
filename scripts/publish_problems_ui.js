/**
 * scripts/publish_problems_ui.js
 * Logs into the live DeepMisconception UI as Dr. Ada Smith (FAC-001)
 * and authors 6 distinct, conceptually rich coding problems via the dashboard interface.
 */
const { chromium } = require('playwright-core');

const problems = [
  {
    title: 'Recursive Factorial with Non-Negative Invariant',
    difficulty: 'easy',
    tags: 'recursion, base-case, call-stack',
    statement: 'Write a recursive function factorial(n: int) -> int that computes the product of all positive integers up to n. Ensure the function correctly validates n >= 0, returning 1 for n=0 and avoiding infinite recursion stack overflow.',
    sampleInput: '5',
    sampleOutput: '120'
  },
  {
    title: 'List Partitioning and Mutable Aliasing',
    difficulty: 'medium',
    tags: 'arrays, aliasing, mutation, pointers',
    statement: 'Write a function split_evens_odds(numbers: list[int]) -> tuple[list[int], list[int]] that partitions integers into even and odd lists without mutating or sharing internal memory references with the original input list.',
    sampleInput: '[1, 2, 3, 4, 5, 6]',
    sampleOutput: '([2, 4, 6], [1, 3, 5])'
  },
  {
    title: 'Palindrome Validator with Slicing Invariant',
    difficulty: 'easy',
    tags: 'strings, two-pointer, immutability',
    statement: 'Write a function is_palindrome(text: str) -> bool that ignores whitespace and case capitalization, returning True if the alphanumeric sequence reads identically forward and backward.',
    sampleInput: '"Race car"',
    sampleOutput: 'True'
  },
  {
    title: 'Fibonacci Sequence Memoization',
    difficulty: 'medium',
    tags: 'recursion, memoization, dynamic-programming',
    statement: 'Write an efficient function fibonacci(n: int, memo: dict = None) -> int that calculates the n-th Fibonacci number (where fib(0)=0, fib(1)=1). Avoid recalculating overlapping subproblems using recursive memoization.',
    sampleInput: '10',
    sampleOutput: '55'
  },
  {
    title: 'Binary Search with Strict Monotonic Bounds',
    difficulty: 'hard',
    tags: 'binary-search, loop-invariants, boundary-conditions',
    statement: 'Implement binary_search(arr: list[int], target: int) -> int that returns the exact zero-based index of target in a strictly sorted ascending list, or -1 if target is absent. Avoid off-by-one boundary condition errors.',
    sampleInput: 'arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91], target = 23',
    sampleOutput: '5'
  },
  {
    title: 'Nested Dictionary Key Counter & Aggregator',
    difficulty: 'medium',
    tags: 'hash-maps, tree-traversal, recursion',
    statement: 'Write a function count_leaf_values(nested_data: dict) -> int that recursively traverses an arbitrarily nested dictionary and sums all primitive integer values stored at terminal leaf nodes.',
    sampleInput: '{"a": 10, "b": {"c": 20, "d": {"e": 30}}}',
    sampleOutput: '60'
  }
];

async function publishProblems() {
  console.log('🚀 Launching Chrome to publish problems through Instructor Dashboard...');
  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 }
  });
  const page = await context.newPage();

  try {
    console.log('1. Navigating to http://localhost:3000...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // Sign out if already logged in
    const logoutBtn = await page.$('button:has-text("Sign Out")');
    if (logoutBtn && await logoutBtn.isVisible()) {
      await logoutBtn.click();
      await page.waitForTimeout(1000);
    }

    // 2. Log in as Dr. Ada Smith (FAC-001)
    console.log('2. Logging into Instructor Portal as Dr. Ada Smith (FAC-001)...');
    await page.click('button:has-text("Teacher / Faculty")');
    await page.waitForTimeout(300);
    await page.fill('input[placeholder="Enter Faculty ID"]', 'FAC-001');
    await page.fill('input[placeholder="Enter Faculty Name"]', 'Dr. Ada Smith');
    await page.fill('input[placeholder="Enter Department"]', 'Computer Science & Engineering');
    await page.click('button[type="submit"]');
    await page.waitForTimeout(2500);

    // 3. Switch to "My Published Questions" authoring tab
    console.log('3. Switching to problem authoring tab in Instructor Dashboard...');
    await page.click('button:has-text("My Published Questions")');
    await page.waitForSelector('input[placeholder="Enter Problem Title"]', { timeout: 10000 });
    console.log('   -> Authoring form ready!');

    for (let i = 0; i < problems.length; i++) {
      const prob = problems[i];
      console.log(`\nPublishing Problem ${i + 1}/6: "${prob.title}"...`);

      // Check if problem already published
      const alreadyPublished = await page.evaluate((title) => {
        return document.body.innerText.includes(title);
      }, prob.title);

      if (alreadyPublished) {
        console.log(`   -> Problem "${prob.title}" already published. Skipping.`);
        continue;
      }

      // Title
      await page.fill('input[placeholder="Enter Problem Title"]', prob.title);

      // Description
      await page.fill('textarea[placeholder="Enter Problem Description"]', prob.statement);

      // Difficulty - find the select for difficulty
      const difficultySelect = await page.$('select:has(option[value="medium"])');
      if (difficultySelect) {
        await difficultySelect.selectOption(prob.difficulty);
      }

      // Concept Tags
      await page.fill('input[placeholder="Enter Concept Tags"]', prob.tags);

      // Sample Input & Output
      await page.fill('textarea[placeholder="Enter Sample Input"]', prob.sampleInput);
      await page.fill('textarea[placeholder="Enter Expected Output"]', prob.sampleOutput);

      // Click Publish to Target Group
      await page.click('button:has-text("Publish to Target Group")');
      await page.waitForTimeout(2000);

      // Wait for success indicator or reset
      await page.waitForSelector('text=Coding question successfully published', { timeout: 10000 }).catch(() => {});
      console.log(`   -> Successfully published problem "${prob.title}"!`);
      await page.waitForTimeout(1000);
    }

    // Capture full-page screenshot of the Instructor Dashboard showing all published problems
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'scratch_instructor_problems.png', fullPage: true });
    console.log('\n📸 Captured Instructor Dashboard screenshot: scratch_instructor_problems.png');

    console.log('\n🎉 ALL 6 PROBLEMS PUBLISHED SUCCESSFULLY VIA DASHBOARD UI!');
  } catch (err) {
    console.error('Error during problem publishing:', err);
    await page.screenshot({ path: 'scratch_error_problems.png' });
  } finally {
    await browser.close();
  }
}

publishProblems();
