# Board/GIF visual regression check

With the dev backend on port 8080 and a recorded episode available:

```sh
cd app/frontend
npm run dev -- --config tests/capture-vite.config.js --port 5174
```

Open `http://localhost:5174/tests/capture.html`.

The page mounts the real `GameBoard` and uses its public `renderStateToPng`
method plus the real GIF encoder. It reads the recorded replay, without
initializing, resetting, or stepping the user's game. Nothing in this test
entry point is included in the production app.

1. Capture a recorded state; compare the live board and decoded GIF. Check the
   blue/red panels, scores, active possession light, clock, banners, and court.
2. Check a shot/rebound state and the final End Game state.
3. Check the explicitly labeled inbound fixture (double-digit scores, inbound
   countdown, and lit lane-warning lights) and the legacy-clock fixture. These
   vary the recorded state for display testing; they are not simulated plays.
4. Click **Check stable banner layout**. This measures court position and frame
   bounds through every recorded state and explicit clearance, shot/rebound,
   steal, turnover, inbound, and end-game fixtures, at both 640px and 440px board
   widths, with live animations and export mode. It also verifies that the
   scoreboard scales down with the board, stays inside it, and export banners
   are fully opaque. Repeat captures at compact width; nothing should crop.
5. Export the full recorded episode. This captures every recorded state and
   asserts the decoded GIF canvas covers the largest frame. Banner changes
   should no longer change the frame dimensions or move the court. Compare the
   scoreboard glows in the lossless PNG and decoded GIF. This harness samples
   one settled frame per step, not the main app's movement interpolation.

`BW_CAPTURE_MEDIA_URL` can point the encoder request to an isolated test server
while replay data continues to come from the existing backend on port 8080.

Backend regressions cover both GIF endpoints, variable frame dimensions,
padding, and the one-second final-frame hold:

```sh
app/backend/.env/bin/python -m pytest app/backend/tests/test_media_routes.py -q
```

## Full Save Episode integration (including animation subframes)

The settled-frame comparison above is not a full Save Episode test. To exercise
the actual `App.vue` Save Episode button, start a GIF-only server from the repo
root in a fresh temporary directory:

```sh
export_test_dir=$(mktemp -d /tmp/basketworld-export-test-XXXXXX)
app/backend/.env/bin/python -m app.backend.tests.gif_export_server "$export_test_dir"
```

In another terminal:

```sh
cd app/frontend
BW_CAPTURE_SEED_REPLAY=1 BW_CAPTURE_MEDIA_URL=http://localhost:8081 \
VITE_API_BASE_URL=http://localhost:5174 \
npm run dev -- --config tests/capture-vite.config.js --port 5174
```

Open `http://localhost:5174/tests/save-episode.html` and click **Save Episode**.
This test-only Vite configuration restores the recorded replay into the real
application instead of initializing a new game. All capture timing,
interpolation, frame uploads, completion, and state restoration use the real
application handler. The status at the top displays its success/error alert.

The isolated server saves GIFs under the temporary directory. Its
`/test/export_stats` endpoint reports total bytes, largest request, active export
count, and saved paths. Verify the GIF decodes every expected animation frame,
has a consistent canvas, retains the scoreboard, and holds the final frame for
one second. Active exports should return to zero after success or failure.

The browser upload regression exercises more than the JavaScript maximum string
length cumulatively, without ever constructing such a string:

```sh
node --test app/frontend/tests/episode-export.test.mjs
```
