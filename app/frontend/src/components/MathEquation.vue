<script setup>
import { nextTick, onMounted, ref, watch } from 'vue';

const props = defineProps({
  tex: {
    type: String,
    required: true,
  },
  display: {
    type: Boolean,
    default: false,
  },
});

const rootEl = ref(null);
let renderVersion = 0;

function getMathSource() {
  return props.display ? `\\[${props.tex}\\]` : `\\(${props.tex}\\)`;
}

async function ensureMathJax() {
  if (typeof window === 'undefined') return null;
  if (window.MathJax?.tex2chtmlPromise || window.MathJax?.typesetPromise) return window.MathJax;
  if (!window.__basketworldMathJaxPromise) {
    window.__basketworldMathJaxPromise = new Promise((resolve) => {
      let attempts = 0;
      const check = () => {
        const mathJax = window.MathJax;
        if (mathJax?.startup?.promise) {
          mathJax.startup.promise.then(() => resolve(window.MathJax || null));
          return;
        }
        if (mathJax?.tex2chtmlPromise || mathJax?.typesetPromise) {
          resolve(mathJax);
          return;
        }
        attempts += 1;
        if (attempts > 200) {
          console.warn('[MathEquation] MathJax script did not become ready');
          resolve(null);
          return;
        }
        window.setTimeout(check, 50);
      };
      check();
    });
  }
  return window.__basketworldMathJaxPromise;
}

async function typesetEquation() {
  const version = ++renderVersion;
  await nextTick();
  const el = rootEl.value;
  if (!el) return;
  el.textContent = getMathSource();
  const mathJax = await ensureMathJax();
  if ((!mathJax?.tex2chtmlPromise && !mathJax?.typesetPromise) || version !== renderVersion) return;

  const runTypeset = async () => {
    if (version !== renderVersion || !rootEl.value) return;
    if (mathJax.tex2chtmlPromise) {
      const node = await mathJax.tex2chtmlPromise(props.tex, { display: props.display });
      if (version !== renderVersion || !rootEl.value) return;
      el.replaceChildren(node);
      return;
    }
    mathJax.typesetClear?.([el]);
    await mathJax.typesetPromise([el]);
  };
  const previous = window.__basketworldMathJaxTypesetQueue || Promise.resolve();
  window.__basketworldMathJaxTypesetQueue = previous
    .catch(() => {})
    .then(runTypeset)
    .catch((err) => {
      console.warn('[MathEquation] Failed to render equation', err);
    });
  await window.__basketworldMathJaxTypesetQueue;
}

onMounted(typesetEquation);
watch(() => [props.tex, props.display], typesetEquation, { flush: 'post' });
</script>

<template>
  <span ref="rootEl" class="math-equation"></span>
</template>
