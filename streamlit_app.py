"""Streamlit front end for Streeto Planner.

Deploy target: Streamlit Community Cloud. Set OPENROUTESERVICE_API_KEY,
MAPBOX_ACCESS_TOKEN and/or STREETO_APP_PASSWORD via that app's secrets (they
also work from a local .env for development) -- see README.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path

import folium
import streamlit as st
from dotenv import load_dotenv
from folium.elements import JSCSSMixin
from folium.template import Template
from streamlit_folium import st_folium

from streeto_planner import distances, maprun, oom
from streeto_planner.solve import solve

load_dotenv()


def _bridge_secrets_into_environ() -> None:
    """Make st.secrets (Streamlit Cloud's secrets store) reach os.environ.

    distances.py and oom.py are framework-agnostic and read plain
    environment variables, so they work the same from the CLI or the app.
    Streamlit's docs confirm root-level secrets.toml keys are also exposed
    as environment variables for local development, but don't document the
    same for Community Cloud's secrets UI -- rather than rely on that,
    copy them across explicitly. .env (via load_dotenv above) wins if a key
    is set both ways, since it ran first and this only fills in gaps.
    """
    try:
        secrets = st.secrets
    except Exception:
        return
    try:
        items = dict(secrets)
    except Exception:
        return
    for key, value in items.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)


_bridge_secrets_into_environ()

st.set_page_config(page_title="Streeto Planner", page_icon="\U0001f9ed", layout="wide")

_ROTATED_OVERLAY_JS_URL = (
    "https://cdn.jsdelivr.net/npm/leaflet-imageoverlay-rotated@0.2.1/"
    "Leaflet.ImageOverlay.Rotated.min.js"
)


class RotatedImageOverlay(JSCSSMixin, folium.MacroElement):
    """Drape an image over three corners rather than an axis-aligned box.

    Leaflet's own imageOverlay only supports axis-aligned bounds, so a
    rotated printed sheet needs the Leaflet.ImageOverlay.Rotated plugin.
    Loading it via a plain header <script src=...> doesn't work here:
    streamlit_folium's header handling strips any script tag with a src
    attribute before sending it to the frontend (see randyzwitch/
    streamlit-folium's _get_header) -- but it separately collects every
    JSCSSMixin subclass's own `default_js` and awaits each one as a real
    <script> element before running the map's own script, which is the
    supported path folium's own plugins (MarkerCluster, Draw, ...) use for
    exactly this. Hence subclassing JSCSSMixin instead of hand-rolling the
    header injection.
    """

    default_js = [("leaflet-imageoverlay-rotated", _ROTATED_OVERLAY_JS_URL)]

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        L.imageOverlay.rotated(
            {{ this.image_url|tojson }},
            {{ this.top_left|tojson }},
            {{ this.top_right|tojson }},
            {{ this.bottom_left|tojson }},
            {{ this.options|tojson }}
        ).addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """
    )

    def __init__(
        self,
        image_url: str,
        top_left: tuple[float, float],
        top_right: tuple[float, float],
        bottom_left: tuple[float, float],
        opacity: float = 0.85,
    ) -> None:
        super().__init__()
        self._name = "RotatedImageOverlay"
        self.image_url = image_url
        self.top_left = list(top_left)
        self.top_right = list(top_right)
        self.bottom_left = list(bottom_left)
        self.options = {"opacity": opacity, "interactive": False}


def _check_password() -> bool:
    """Gate the app behind a shared password, if one is configured.

    Off by default, so local use is unaffected. Set STREETO_APP_PASSWORD
    before any deployment reachable from the open internet.
    """
    password = os.environ.get("STREETO_APP_PASSWORD")
    if not password:
        return True
    if st.session_state.get("authed"):
        return True

    entered = st.text_input("Password", type="password")
    if entered:
        if entered == password:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


KNOWN_MAPS_PATH = Path(__file__).parent / "known_maps.json"
NEW_MAP_OPTION = "__new__"


def _load_known_maps() -> list[dict[str, str]]:
    """Previously-used map IDs, most recently added first.

    Read fresh (not cached) since _remember_known_map can change the file
    within the same running app. A local JSON file persists fully when
    running locally; on Streamlit Cloud it lasts until the next redeploy,
    then resets to whatever's committed -- fine for a personal convenience
    list, not meant as durable multi-user storage.
    """
    try:
        raw = json.loads(KNOWN_MAPS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, dict) and "id" in m]


def _remember_known_map(map_id: str, title: str) -> None:
    """Add a successfully-loaded map to the known list, if it's new."""
    known = _load_known_maps()
    if any(m["id"] == map_id for m in known):
        return
    known.insert(0, {"id": map_id, "label": title or map_id})
    with contextlib.suppress(OSError):
        # Best-effort: a read-only filesystem shouldn't break solving.
        KNOWN_MAPS_PATH.write_text(
            json.dumps(known, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


@st.cache_data(show_spinner="Loading course...", ttl=600)
def _load_course(map_id: str) -> oom.Course:
    return oom.fetch_course(map_id)


@st.cache_data(show_spinner=False, ttl=600)
def _render_image_url(map_id: str) -> str | None:
    """The real OOM map image's URL, or None if this course has no usable one."""
    try:
        return oom.render_image_url(map_id)
    except (ValueError, LookupError, OSError):
        return None


@st.cache_data(show_spinner="Computing distances...", ttl=600)
def _matrix(map_id: str, routed: bool, provider: str, detour: float):
    course = _load_course(map_id)
    nodes = course.nodes
    if not routed:
        matrix = distances.approximate_matrix(
            nodes, detour, no_access_points=course.no_access_points
        )
        caveat = (
            f"approximate: straight line x {detour}, historically "
            f"~{distances.APPROXIMATION_OPTIMISM:.0%} optimistic"
        )
        return matrix, [], caveat
    if provider == "mapbox":
        matrix, warnings = distances.mapbox_matrix(nodes)
        caveat = "real pedestrian routing (Mapbox)"
    else:
        matrix, warnings = distances.routed_matrix(nodes)
        caveat = "real pedestrian routing (OpenRouteService)"
    return matrix, warnings, caveat


@st.cache_data(show_spinner="Fetching the real route...", ttl=600)
def _geometry(map_id: str, visit_numbers: tuple[str, ...]) -> dict:
    course = _load_course(map_id)
    by_number = {c.number: c for c in course.controls}
    waypoints = [
        course.start_finish,
        *(by_number[n] for n in visit_numbers),
        course.start_finish,
    ]
    return distances.route_geometry(
        waypoints, avoid_points=course.no_access_points or None
    )


@st.cache_data(show_spinner="Fetching MapRun results...", ttl=600)
def _maprun_results(event_name: str) -> list[maprun.Result]:
    """Cached wrapper around maprun.fetch_results.

    MapRun's API asks for "reasonable use" -- no faster than one call every
    10 seconds, and not for bulk/every event -- so this is cached rather
    than refetched on every rerun a selectbox interaction triggers.
    """
    return maprun.fetch_results(event_name)


def _build_map(
    course: oom.Course,
    route,
    map_id: str,
    show_real_map: bool,
) -> folium.Map:
    fmap = folium.Map(
        location=[course.centre_latitude, course.centre_longitude],
        tiles="OpenStreetMap",
    )

    boundary = distances.course_boundary(course)
    if show_real_map and boundary is not None:
        image_url = _render_image_url(map_id)
        if image_url is not None:
            # boundary corners are ordered (bottom-left, bottom-right,
            # top-right, top-left) -- see course_boundary in distances.py.
            RotatedImageOverlay(
                image_url,
                top_left=boundary[3],
                top_right=boundary[2],
                bottom_left=boundary[0],
            ).add_to(fmap)
        else:
            st.info("Real map image unavailable for this course.")

    if boundary is not None:
        folium.Polygon(
            locations=boundary,
            color="#1c1b19",
            weight=2,
            dash_array="4 6",
            fill=False,
            tooltip="Printed map edge",
        ).add_to(fmap)

    folium.CircleMarker(
        [course.start_finish.latitude, course.start_finish.longitude],
        radius=8, color="#1c1b19", fill=True, fill_color="#1c1b19", fill_opacity=1,
        tooltip="Start/Finish",
    ).add_to(fmap)
    for c in course.controls:
        folium.CircleMarker(
            [c.latitude, c.longitude],
            radius=5, color="#b3541e", fill=True, fill_color="#f2c9ab",
            fill_opacity=0.9, weight=2,
            popup=f"Control {c.number}: {c.points} points",
        ).add_to(fmap)
        folium.Marker(
            [c.latitude, c.longitude],
            icon=folium.DivIcon(html=(
                '<div style="font-size:11px;font-weight:700;color:#1c1b19;'
                'text-shadow:0 0 3px #fff,0 0 3px #fff;'
                'transform:translate(-50%,-150%)">'
                f"{c.number}</div>"
            )),
        ).add_to(fmap)
    for lat, lon in course.no_access_points:
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=(
                '<div style="font-size:16px;font-weight:900;color:#8b1e8b;'
                'text-shadow:0 0 3px #fff,0 0 3px #fff;'
                'transform:translate(-50%,-50%)">&times;</div>'
            )),
            tooltip="No access",
        ).add_to(fmap)

    visits = [c.number for c in route.controls]
    strayed = False
    geometry_warnings: list[distances.SnapWarning] = []
    try:
        geometry = _geometry(map_id, tuple(visits))
        folium.GeoJson(
            geometry, style_function=lambda _: {"color": "#b3541e", "weight": 4}
        ).add_to(fmap)
        strayed = distances.route_strays_outside_course(geometry, course)
        by_number = {c.number: c for c in course.controls}
        waypoints = [
            course.start_finish,
            *(by_number[n] for n in visits),
            course.start_finish,
        ]
        geometry_warnings = distances.geometry_snap_warnings(geometry, waypoints)
    except (ValueError, RuntimeError, OSError):
        by_number = {c.number: c for c in course.controls}
        coords = [[course.start_finish.latitude, course.start_finish.longitude]]
        coords += [[by_number[n].latitude, by_number[n].longitude] for n in visits]
        coords.append([course.start_finish.latitude, course.start_finish.longitude])
        folium.PolyLine(
            coords, color="#b3541e", weight=3, dash_array="6 6"
        ).add_to(fmap)
        st.caption("Showing a straight-line approximation (no route geometry).")

    if strayed:
        st.warning(
            "The routed path strays outside the course's control area — "
            "possibly routing off the printed map."
        )
    if geometry_warnings:
        st.warning(
            "The drawn route snapped these controls far from their mapped "
            "position, so the path shown near them may look wrong: "
            + ", ".join(
                f"{w.number} ({w.snapped_distance:.0f}m)" for w in geometry_warnings
            )
        )

    bounds = [[n.latitude, n.longitude] for n in course.nodes]
    if boundary is not None:
        bounds += [list(corner) for corner in boundary]
    fmap.fit_bounds(bounds, padding=(30, 30))
    return fmap


def main() -> None:
    if not _check_password():
        st.stop()

    st.title("Streeto Planner")
    st.caption("Point a map ID at it, get a route.")

    known_maps = _load_known_maps()
    options = [m["id"] for m in known_maps] + [NEW_MAP_OPTION]
    labels = {NEW_MAP_OPTION: "+ Enter a new map ID..."} | {
        m["id"]: f"{m['label']} ({m['id']})" for m in known_maps
    }
    selected = st.selectbox("Map", options, format_func=lambda v: labels[v])
    if selected == NEW_MAP_OPTION:
        map_id_input = st.text_input("New map ID", placeholder="e.g. 698d11dd9c78c")
    else:
        map_id_input = selected

    with st.form("solve_form"):
        km_text = st.text_input(
            "Distance budget (km)",
            placeholder="leave blank for the optimum route through every control",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            routed = st.checkbox("Real pedestrian routing", value=False)
        with col2:
            provider_label = st.selectbox(
                "Provider", ["OpenRouteService", "Mapbox"], disabled=not routed
            )
        with col3:
            show_real_map = st.checkbox("Show the real map", value=True)
        submitted = st.form_submit_button("Solve", type="primary")

    if submitted:
        map_id = map_id_input.strip()
        if not map_id:
            st.error("Enter a map ID.")
            st.stop()

        try:
            course = _load_course(map_id)
        except (ValueError, LookupError, OSError) as error:
            st.error(str(error))
            st.stop()
        _remember_known_map(map_id, course.title)

        provider_key = "mapbox" if provider_label == "Mapbox" else "ors"
        try:
            matrix, warnings, caveat = _matrix(
                map_id, routed, provider_key, distances.DEFAULT_DETOUR
            )
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error))
            st.stop()

        if km_text.strip():
            try:
                budget = float(km_text) * 1000
            except ValueError:
                st.error("Distance budget must be a number.")
                st.stop()
        else:
            budget = math.inf

        route = solve(course.nodes, matrix, budget, iterations=60, seed=0)

        st.session_state["result"] = {
            "map_id": map_id,
            "course": course,
            "route": route,
            "matrix": matrix,
            "caveat": caveat,
            "warnings": warnings,
            "unbounded": budget == math.inf,
            "show_real_map": show_real_map,
            "routed": routed,
        }

    result = st.session_state.get("result")
    if not result:
        return

    course = result["course"]
    route = result["route"]

    st.subheader(course.title or "(untitled)")
    st.write(f"{course.event_date} · {course.club} · scale 1:{course.scale}")

    if result["unbounded"]:
        st.success(
            f"Optimum route through all {len(course.controls)} controls: "
            f"{round(route.distance)} m"
        )
    else:
        st.success(
            f"{route.points} / {course.total_points} points "
            f"({100 * route.points / course.total_points:.0f}%), "
            f"{len(route.controls)} stops, {round(route.distance)} m, "
            f"gap ≤ {route.gap:.0%}"
        )
    st.caption(result["caveat"])

    if course.no_access_points:
        note = (
            "the drawn route avoids them, but the route itself was chosen "
            "without accounting for them (OpenRouteService's Matrix API "
            "ignores avoid_polygons entirely -- confirmed directly, not "
            "assumed)"
            if result["routed"]
            else "legs crossing too close to one are penalised when "
            "choosing the route"
        )
        st.caption(
            f"{len(course.no_access_points)} marked no-access point(s) on "
            f"this course — {note}."
        )

    anomalies = course.scoring_anomalies()
    if anomalies:
        st.warning(
            "Scoring anomalies: "
            + ", ".join(
                f"{c.number}={c.points} (expected {c.implied_points})"
                for c in anomalies
            )
        )
    if result["warnings"]:
        st.warning(
            "Snapped far from mapped position, legs touching them may be "
            "unreliable: "
            + ", ".join(
                f"{w.number} ({w.snapped_distance:.0f}m)" for w in result["warnings"]
            )
        )

    st.write("Route: " + " → ".join(c.number for c in route.controls))

    fmap = _build_map(course, route, result["map_id"], result["show_real_map"])
    st_folium(fmap, width=None, height=650, returned_objects=[])

    _render_maprun_comparison(course, result)


def _render_maprun_comparison(course: oom.Course, result: dict) -> None:
    """A section comparing a real MapRun runner's result against the solver."""
    st.divider()
    st.subheader("Compare with a real MapRun result")

    known_maps = _load_known_maps()
    known_entry = next((m for m in known_maps if m["id"] == result["map_id"]), None)
    default_event_name = (known_entry or {}).get("maprun_event", "")

    event_name = st.text_input(
        "MapRun event name",
        value=default_event_name,
        placeholder="e.g. Cobham Mar26 PXAS ScoreQ60",
    )
    if st.button("Fetch MapRun results"):
        if not event_name.strip():
            st.error("Enter a MapRun event name.")
        else:
            try:
                fetched = _maprun_results(event_name.strip())
            except (LookupError, ValueError, OSError) as error:
                st.error(str(error))
            else:
                st.session_state["maprun"] = {
                    "event_name": event_name.strip(),
                    "results": fetched,
                }

    maprun_state = st.session_state.get("maprun")
    if not maprun_state or maprun_state["event_name"] != event_name.strip():
        return

    entries = maprun_state["results"]
    if not entries:
        st.warning("No results found for that event name.")
        return

    ordered = sorted(entries, key=lambda r: (r.classifier != "OK", -r.gross_score))

    def _label(i: int) -> str:
        r = ordered[i]
        return (
            f"{r.name} — {r.classifier} — {r.distance_km:.2f}km — "
            f"{r.gross_score}pts — {r.total_time_hhmmss}"
        )

    chosen_index = st.selectbox(
        "Runner", range(len(ordered)), format_func=_label
    )
    chosen = ordered[chosen_index]

    if chosen.distance_km <= 0:
        st.info("This result has no usable distance to compare against.")
        return

    solver_route = solve(
        course.nodes, result["matrix"], chosen.distance_km * 1000,
        iterations=60, seed=0,
    )

    course_control_numbers = {c.number for c in course.controls}
    their_controls = set(chosen.visits) & course_control_numbers
    our_controls = {c.number for c in solver_route.controls}

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            f"{chosen.name}'s actual score",
            chosen.gross_score,
            help=f"{chosen.distance_km:.2f}km in {chosen.total_time_hhmmss}, "
                 f"{len(their_controls)} controls",
        )
    with col2:
        st.metric(
            "Solver's score at the same distance",
            solver_route.points,
            delta=solver_route.points - chosen.gross_score,
            help=f"{round(solver_route.distance)}m, "
                 f"{len(solver_route.controls)} controls",
        )

    missed_by_solver = sorted(their_controls - our_controls, key=int)
    missed_by_runner = sorted(our_controls - their_controls, key=int)
    if missed_by_solver:
        st.write(
            "Controls they got that the solver's route skips: "
            + ", ".join(missed_by_solver)
        )
    if missed_by_runner:
        st.write(
            "Controls the solver's route adds that they didn't get: "
            + ", ".join(missed_by_runner)
        )
    if not missed_by_solver and not missed_by_runner:
        st.write("Same set of controls, just possibly a different order.")


main()
