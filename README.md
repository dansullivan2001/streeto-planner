# Street-O Planner

Point your phone at a Street-O map, get a high-scoring route.

Street-O is score orienteering: controls are scattered across a town, each worth
points, and you have a fixed time to collect as many as you can. Choosing which
ones to visit and in what order is the orienteering problem — a travelling
salesman who is allowed to skip stops, and paid for the ones he keeps.

## Status

Early. The course reader, the route solver and the map-ID recovery all work from
the command line and from a Streamlit web app. Distances are approximated
unless you supply a routing key.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

For the web app, add the `streamlit` extra:

```bash
.venv/bin/pip install -e ".[streamlit]"
```

## Use

Summarise a course straight from the ID printed on the map:

```bash
.venv/bin/streeto course 698d11dd9c78c
```

Plan routes across a range of distance budgets:

```bash
.venv/bin/streeto solve 698d11dd9c78c --km 10 13 15
```

With real pedestrian routing (`OPENROUTESERVICE_API_KEY` or `MAPBOX_ACCESS_TOKEN`
must be set):

```bash
.venv/bin/streeto solve 698d11dd9c78c --km 10 13 15 --routed --provider mapbox
```

Recover an ID you could not quite read, marking the uncertain characters:

```bash
.venv/bin/streeto resolve '698d11dd[9g]c78c'
```

## Web app

```bash
.venv/bin/streamlit run streamlit_app.py
```

Enter a map ID and a distance budget, or leave the budget blank to plan the
optimum route through every control. The map shows the real routed path (not
just straight lines between stops), control numbers, the printed sheet's
actual boundary, and — toggleable — OOM's own rendered map image (the one
with the purple control circles), draped over real geography at its true
position and rotation.

Copy `.env.example` to `.env` and fill in the keys you have; the same file
also holds `STREETO_APP_PASSWORD`, which gates the whole app behind a shared
password if set (leave blank for local use). Deployed to
[Streamlit Community Cloud](https://streamlit.io/cloud), the same variable
names go in that app's *Secrets* settings instead of `.env`.

The map picker remembers every map ID solved successfully, in `known_maps.json`,
so past events build up into a dropdown instead of needing to be retyped.

If a course was also run as a MapRun event, entering its event name (or
letting it prefill from `known_maps.json`'s `maprun_event` field) pulls real
results from [MapRun's public results API](https://maprunners.weebly.com/access-to-results.html)
and compares a chosen runner's actual score and distance against what the
solver finds for that same distance budget, control by control. That API has
no GPS trace data — only each runner's score, distance, time and the order
they punched controls — and asks for "reasonable use" (no more than one call
every 10 seconds, not bulk-fetching every event), which the app respects by
caching rather than refetching on every interaction.

## How it works

**Read the corner, not the controls.** Every OpenOrienteeringMap sheet prints
its own map ID next to the credits. Reading that one string beats detecting
forty purple circles and georeferencing a photograph, and it yields exact
coordinates rather than estimates. Feed the ID to OOM and you get every control,
its score, and the map geometry.

**Map IDs carry a checksum of sorts.** They are PHP `uniqid()` values: eight hex
digits of Unix seconds, then five of microseconds. So an ID decodes to the
moment the map was made, which is normally a few weeks before the event. That
makes most misreadings detectable without a network round trip. The microsecond
half is a much weaker filter — only about 5% of values are impossible — so
`resolve` leans on the date and on constraining which characters were doubtful.

**Scores follow the numbering.** A control's points are the first digit of its
number times ten: 47 is worth 40. This has held for every control on every map
checked so far, so `course` reports any control that breaks it — a useful signal
that something was misread or that an organiser has done something unusual.

**Distances are approximated by default.** Real planning needs pedestrian
routing. Set `OPENROUTESERVICE_API_KEY` and pass `--routed` for that, or set
`MAPBOX_ACCESS_TOKEN` and pass `--routed --provider mapbox` to use Mapbox's
Matrix API instead. Mapbox caps a single request at 25 coordinates, well under
a typical course's control count, so that path chunks the course into blocks
and stitches the full matrix together from several requests. Without a key,
distances are great-circle inflated by a detour index. Measured against real
routed answers on two solved London courses, the index came out at 1.40 in the
inner city and 1.47 in the suburbs, and the approximation flatters the solver
by 5–9% — it picks legs that are cheap in a straight line but blocked by a
river or a railway. Treat approximate scores as ±10% indicative.

**The solver is a heuristic.** Randomised greedy insertion, 2-opt cleanup,
restarted. Forty-six controls across eleven budgets solves in under half a
second of pure Python, which is the main reason a live, in-browser version looks
practical. Each route reports a `gap` against an upper bound built from which
controls are reachable out-and-back at all. The bound is loose, especially at
small budgets, but it is honest: it never claims optimality it has not proven.

## Known limits

- Uncrossable walls, railways and fences (`c_cross` in OOM) are not yet fed to
  the router, so a route may cross something the event forbids.
- The budget is distance, not time. Street-O is scored on time with a penalty
  for finishing late, so the right model is points minus penalty, given a
  runner's pace. That is the next thing worth fixing.
- Terrain is ignored. On a map with contours, hills cost time that a distance
  budget cannot see.
- A route can be routed outside the printed map's boundary. OOM's `rotation`
  field has no documented sign convention, so it's derived per course by
  testing both signs against that course's own controls — which must lie
  inside the printed sheet by construction — and using whichever one actually
  contains them (validated against a real course; see `course_boundary` in
  `distances.py`). This is a warning, not a hard block: if a future course
  doesn't fit either sign, it falls back to a looser box around the course's
  own controls rather than trust a boundary it can't confirm.
- Some controls sit near roads OpenStreetMap tags `access=private` with no
  `foot=yes`/`foot=permissive` override (common in Cobham's gated estate
  roads) — real, walkable roads that OpenRouteService's foot profile excludes
  anyway, since it can't assume pedestrian access just because vehicle access
  is restricted. It searches outward for the nearest road it's sure permits
  pedestrians, which can be over 100m away, corrupting the distances that
  touch that control. `solve --routed` and the app flag any control snapped
  more than 50m from its real position, and replace distances touching it
  with the straight-line approximation rather than trust the corrupted
  number (see `_patch_unreliable_legs` in `distances.py`). There's no
  request-level API option to override this; the real fix is upstream, in
  OpenStreetMap's own tagging.

## Credits

Inspired by [pbatch/StreetoSolver](https://github.com/pbatch/StreetoSolver),
which solves the same problem with a precomputed static site. That is a separate
project and none of its code is used here, though `oom.render_image_url`'s
request to OOM's map-image renderer was worked out by reading how that
project's `pull_map.py` builds the same request.

Course data comes from [OpenOrienteeringMap](https://oomap.co.uk/), created by
Oliver O'Brien, with OOM v4 developed by David Dixon. Map data is
© OpenStreetMap contributors, available under the
[Open Database Licence](https://www.openstreetmap.org/copyright).

## Troubleshooting

`CERTIFICATE_VERIFY_FAILED` on macOS means your Python has no CA bundle. Run the
`Install Certificates.command` that ships in your `/Applications/Python 3.x/`
folder, or `pip install certifi` and export
`SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')`.

## Licence

MIT.
