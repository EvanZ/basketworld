<script setup>
const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
});

const emit = defineEmits(['play-again']);

function playAgain() {
  emit('play-again');
}
</script>

<template>
  <div class="game-over-overlay">
    <div class="game-over-modal">
      <h2>Game Over</h2>
      <p v-if="results && results.shots && Object.keys(results.shots).length > 0">
        Shot outcome: <strong>{{ results.shots[Object.keys(results.shots)[0]].success ? 'Made!' : 'Missed!' }}</strong>
      </p>
      <p v-else-if="results && results.out_of_bounds_turnover">
        Reason: <strong>Turnover (Player Stepped Out)</strong>
      </p>
      <p v-else-if="results && results.passes && Object.values(results.passes).some(p => p.turnover)">
        Reason: <strong>
          {{ Object.values(results.passes).find(p => p.turnover).reason === 'intercepted' 
              ? 'Turnover (Intercepted Pass)' 
              : 'Turnover (Pass Out of Bounds)' 
          }}
        </strong>
      </p>
      <p v-else>
        Reason: <strong>Shot Clock Expired</strong>
      </p>
      <button @click="playAgain">Play Again</button>
    </div>
  </div>
</template>

<style scoped>
.game-over-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}
.game-over-modal {
  background-color: white;
  padding: 2rem 3rem;
  border-radius: 8px;
  text-align: center;
}
h2 {
  margin-bottom: 1rem;
}
button {
  margin-top: 1.5rem;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  cursor: pointer;
}
</style> 