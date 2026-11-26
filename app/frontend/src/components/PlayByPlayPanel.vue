<script setup>
defineProps({ rows: { type: Array, default: () => [] } });
</script>

<template>
  <section class="play-by-play" aria-labelledby="dev-play-by-play-title">
    <header>
      <h3 id="dev-play-by-play-title">Play-By-Play</h3>
      <span>Newest first · Latest 240 events</span>
    </header>
    <div class="event-scroll" tabindex="0" aria-label="Play-by-play events">
      <table>
        <thead><tr><th>Clock</th><th>Score<br /><small>Blue–Red</small></th><th>Team</th><th>Event</th></tr></thead>
        <tbody>
          <tr v-if="!rows.length"><td colspan="4" class="empty">No events yet. Shots, blocks, turnovers, steals, and violations will appear here.</td></tr>
          <tr v-for="row in rows" :key="row.id">
            <td class="clock"><span>{{ row.period }}</span>{{ row.clock }}</td>
            <td class="score">{{ row.score }}</td>
            <td class="team">{{ row.team === 'Team A' ? 'Blue' : row.team === 'Team B' ? 'Red' : row.team }}</td>
            <td>{{ row.event }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<style scoped>
.play-by-play { margin: 12px 0 0; padding: 0; background: rgba(255,255,255,.95); border: 1px solid #ddd; border-radius: 8px; color: #333; overflow: hidden; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,.1); }
header { padding: 10px; display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 10px; }
h3 { margin: 0; color: #333; font-size: 14px; font-weight: 600; text-transform: none; letter-spacing: normal; }
header span { font-size: 11px; color: #666; }
.event-scroll { max-height: 340px; overflow: auto; }
table { border-collapse: collapse; width: 100%; text-align: left; }
th, td { padding: 8px; vertical-align: top; border-top: 1px solid #e5e7eb; }
th { position: sticky; top: 0; background: #f3f4f6; z-index: 1; font-weight: 600; }
tbody tr:nth-child(even) { background: #f9fafb; }
.clock, .score, .team { white-space: nowrap; }
.clock, .score { font-variant-numeric: tabular-nums; }
.clock span { display: block; font-size: 10px; color: #666; margin-bottom: 2px; }
.empty { color: #666; padding: 14px 10px; }
</style>
