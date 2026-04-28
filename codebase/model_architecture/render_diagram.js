const { chromium } = require('playwright');
const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');

async function convert() {
  const inputPath = path.join(__dirname, 'NMTDiagram.jsx');
  const bundledPath = path.join(__dirname, 'temp_bundle.js');

  console.log("Transpiling JSX...");
  // 1. Transpile JSX into standard JS that the browser understands
  await esbuild.build({
    entryPoints: [inputPath],
    bundle: true,
    outfile: bundledPath,
    loader: { '.jsx': 'jsx' },
    format: 'iife',
    globalName: 'NMTDiagram',
    external: ['react', 'react-dom'], // We'll provide these in the HTML
  });

  console.log("Launching browser...");
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // 2. Create a simple HTML wrapper to render your component
  const htmlContent = `
    <!DOCTYPE html>
    <html>
      <head>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <style>body { margin: 0; background: #080c16; }</style>
      </head>
      <body>
        <div id="root"></div>
        <script>${fs.readFileSync(bundledPath, 'utf8')}</script>
        <script>
          const root = ReactDOM.createRoot(document.getElementById('root'));
          // Accessing the default export from the esbuild bundle
          root.render(React.createElement(NMTDiagram.default));
        </script>
      </body>
    </html>
  `;

  await page.setContent(htmlContent);
  
  // Wait for the SVG to render (adjust timeout if your diagram is huge)
  await page.waitForSelector('svg');

  console.log("Capturing screenshot...");
  const outputPath = path.join(__dirname, 'NMT_Architecture.png');
  
  // Target the SVG specifically for a clean crop
  const element = await page.$('svg');
  await element.screenshot({ path: outputPath });

  await browser.close();
  fs.unlinkSync(bundledPath); // Clean up temporary file
  
  console.log(`\n✨ Success! Image saved to:\n${outputPath}`);
}

convert().catch(console.error);
