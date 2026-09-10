# Street-O Planner

Point your phone at a Street-O map, get a high-scoring route.

Street-O is score orienteering: controls are scattered across a town, each worth
points, and you have a fixed time to collect as many as you can. Choosing which
ones to visit and in what order is the orienteering problem — a travelling
salesman who is allowed to skip stops, and paid for the ones he keeps.

## Status

Early. The course reader, the route solver and the map-ID recovery all work from
the command line. There is no web front end yet, and distances are approximated
unless you supply a routing key.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
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

Recover an ID you could not quite read, marking the uncertain characters:

```bash
.venv/bin/streeto resolve '698d11dd[9g]c78c'
```

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
routing. Set `OPENROUTESERVICE_API_KEY` and pass `--routed` for that. Without a
key, distances are great-circle inflated by a detour index. Measured against
real routed answers on two solved London courses, the index came out at 1.40 in
the inner city and 1.47 in the suburbs, and the approximation flatters the
solver by 5–9% — it picks legs that are cheap in a straight line but blocked by
a river or a railway. Treat approximate scores as ±10% indicative.

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

## Credits

Inspired by [pbatch/StreetoSolver](https://github.com/pbatch/StreetoSolver),
which solves the same problem with a precomputed static site. That is a separate
project and none of its code is used here.

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
