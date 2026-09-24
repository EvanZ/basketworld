// Render the actual, styled board DOM. In particular, there is no separately
// maintained scoreboard layout: flex sizing, lights, fonts, and banners all
// come from the same elements that the player sees.
import scoreboardFontUrl from 'dseg/fonts/DSEG7-Classic/DSEG7Classic-Regular.woff2?url';

const SVG_NS = 'http://www.w3.org/2000/svg';
let embeddedFontPromise;

function asDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

async function embeddedFontCss() {
  // SVG images cannot fetch external fonts. Embed the same bundled face used
  // by main.css, once per page, so every animation frame stays self-contained.
  if (!embeddedFontPromise) {
    embeddedFontPromise = fetch(scoreboardFontUrl).then(async (response) => {
      if (!response.ok) throw new Error(`Cannot load scoreboard font: ${response.status}`);
      const source = await asDataUrl(await response.blob());
      return `@font-face { font-family: 'DSEG7 Classic'; src: url("${source}") format("woff2"); font-weight: 400; font-style: normal; }`;
    }).catch((error) => {
      embeddedFontPromise = undefined;
      throw error;
    });
  }
  return embeddedFontPromise;
}

function inlineStyles(source, target) {
  const style = getComputedStyle(source);
  for (const property of style) {
    // Computed SVG paint references can become absolute URLs. Keep local
    // gradients/markers/masks pointing into the cloned SVG instead.
    const value = style.getPropertyValue(property).replace(
      /url\(["']?[^)"']*#([^"')]+)["']?\)/g,
      'url(#$1)',
    );
    target.style.setProperty(property, value);
  }
  target.style.animation = 'none';
  target.style.transition = 'none';
  for (let index = 0; index < source.children.length; index += 1) {
    inlineStyles(source.children[index], target.children[index]);
  }
}

async function renderBoardToCanvas(board, { scale = 2, width: targetWidth } = {}) {
  if (!board) throw new Error('Game board is unavailable for capture');
  await document.fonts.ready;
  const fontCss = await embeddedFontCss();
  const boardRect = board.getBoundingClientRect();
  if (!(boardRect.width > 0 && boardRect.height > 0)) throw new Error('Game board has no visible dimensions');
  // At narrow widths, the live scoreboard/legacy clock can extend beyond the
  // court container. Include their actual bounds without reflowing the board.
  const bounds = [boardRect, ...Array.from(board.querySelectorAll(
    '.game-scoreboard, .shot-clock-wrapper, .clearance-required-banner, .shot-attempt-banner, .fast-mode-warming-banner',
  ), (node) => node.getBoundingClientRect()).filter((rect) => rect.width > 0 && rect.height > 0)];
  const left = Math.min(...bounds.map((rect) => rect.left));
  const top = Math.min(...bounds.map((rect) => rect.top));
  const width = Math.max(...bounds.map((rect) => rect.right)) - left;
  const height = Math.max(...bounds.map((rect) => rect.bottom)) - top;
  // GIFs are often shared at a much smaller size than the live dev board.
  // Render directly at the requested width instead of capturing a large image
  // and relying on a second resize step in the GIF encoder.
  const requestedWidth = Number(targetWidth);
  const outputWidth = Number.isFinite(requestedWidth) && requestedWidth > 0
    ? Math.round(requestedWidth)
    : Math.round(width * scale);
  const outputHeight = Math.max(1, Math.round(outputWidth * height / width));

  const clone = board.cloneNode(true);
  inlineStyles(board, clone);
  Object.assign(clone.style, {
    margin: '0', position: 'relative', top: '0', left: '0',
    transform: `translate(${boardRect.left - left}px, ${boardRect.top - top}px)`,
    width: `${boardRect.width}px`, height: `${boardRect.height}px`,
  });
  // Keep the measured layout but omit interactive export/turn controls.
  clone.querySelectorAll('.board-toolbar, .shot-clock-controls').forEach((node) => {
    node.style.opacity = '0';
  });

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const foreignObject = document.createElementNS(SVG_NS, 'foreignObject');
  foreignObject.setAttribute('width', '100%');
  foreignObject.setAttribute('height', '100%');
  const fonts = document.createElement('style');
  fonts.textContent = fontCss;
  clone.prepend(fonts);
  foreignObject.append(clone);
  svg.append(foreignObject);

  // A blob URL containing foreignObject taints the canvas in Chromium even
  // when everything is local. A self-contained SVG data URL stays origin-clean.
  const source = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(svg))}`;
  const image = await new Promise((resolve, reject) => {
    const img = new Image();
    const timer = setTimeout(() => reject(new Error('Timed out rendering game board')), 10000);
    img.onload = () => { clearTimeout(timer); resolve(img); };
    img.onerror = () => { clearTimeout(timer); reject(new Error('Could not render game board image')); };
    img.src = source;
  });
  const canvas = document.createElement('canvas');
  canvas.width = outputWidth;
  canvas.height = outputHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas rendering context is unavailable');
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export async function captureBoardPngBlob(board, options = {}) {
  const canvas = await renderBoardToCanvas(board, options);
  const blob = await new Promise((resolve, reject) => {
    canvas.toBlob((result) => {
      if (result) resolve(result);
      else reject(new Error('Could not encode game board PNG'));
    }, 'image/png');
  });
  return blob;
}

export async function captureBoardPng(board, options = {}) {
  return asDataUrl(await captureBoardPngBlob(board, options));
}
