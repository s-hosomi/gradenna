// Screenshot each tab of the dev server for visual review (not shipped).
import { chromium } from "playwright";

const url = process.env.URL ?? "http://localhost:5173";
const outDir = process.env.OUT ?? "/tmp/gradenna_shots";
const tabs = ["Optimization", "Live FDTD", "Far Field 3D", "S11"];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on("console", (m) => {
  if (m.type() === "error") console.log("[console.error]", m.text());
});
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(url);
await page.waitForTimeout(1500);

for (const label of tabs) {
  await page.getByRole("button", { name: label }).click();
  // Let data load / wasm warm up / animations settle.
  await page.waitForTimeout(label === "Live FDTD" ? 4500 : 2500);
  const slug = label.toLowerCase().replace(/\s+/g, "_");
  await page.screenshot({ path: `${outDir}/${slug}.png` });
  console.log("shot:", slug);
}
await browser.close();
