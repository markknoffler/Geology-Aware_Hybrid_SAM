const { chromium } = require("playwright");
const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

async function convert() {
  const inputPath = path.join(__dirname, "LandslideArchitectureDiagram.tsx");
  const bundledPath = path.join(__dirname, "temp_landslide_bundle.js");

  console.log("Transpiling TSX...");
  await esbuild.build({
    entryPoints: [inputPath],
    bundle: true,
    outfile: bundledPath,
    loader: { ".tsx": "tsx", ".ts": "ts" },
    format: "iife",
    globalName: "LandslideArchitectureDiagram",
    external: ["react", "react-dom"],
  });

  console.log("Launching browser...");
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 2400, height: 1500 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  const htmlContent = `
    <!DOCTYPE html>
    <html>
      <head>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <style>body { margin: 0; background: #eef2f7; }</style>
      </head>
      <body>
        <div id="root"></div>
        <script>${fs.readFileSync(bundledPath, "utf8")}</script>
        <script>
          const root = ReactDOM.createRoot(document.getElementById("root"));
          root.render(React.createElement(LandslideArchitectureDiagram.default));
        </script>
      </body>
    </html>
  `;

  await page.setContent(htmlContent);
  await page.waitForSelector("svg", { timeout: 60000 });
  await page.waitForTimeout(400);

  console.log("Capturing screenshot...");
  const outputPath = path.join(__dirname, "Landslide_Architecture.png");
  const element = await page.$("svg");
  await element.screenshot({ path: outputPath });

  await context.close();
  await browser.close();
  fs.unlinkSync(bundledPath);

  console.log(`\nSuccess! Image saved to:\n${outputPath}`);
}

convert().catch(console.error);
