"""Streamlit front end for Streeto Planner.

Deploy target: Streamlit Community Cloud. Set OPENROUTESERVICE_API_KEY,
MAPBOX_ACCESS_TOKEN and/or STREETO_APP_PASSWORD via that app's secrets (they
also work from a local .env for development) -- see README.
"""

from __future__ import annotations

import math
import os

import folium
import streamlit as st
from dotenv import load_dotenv
from folium.elements import JSCSSMixin
from folium.template import Template
from streamlit_folium import st_folium

from streeto_planner import distances, oom
from streeto_planner.solve import solve

load_dotenv()

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
        matrix = distances.approximate_matrix(nodes, detour)
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
    return distances.route_geometry(waypoints)


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

    visits = [c.number for c in route.controls]
    strayed = False
    try:
        geometry = _geometry(map_id, tuple(visits))
        folium.GeoJson(
            geometry, style_function=lambda _: {"color": "#b3541e", "weight": 4}
        ).add_to(fmap)
        strayed = distances.route_strays_outside_course(geometry, course)
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

    with st.form("solve_form"):
        map_id = st.text_input("Map ID", placeholder="e.g. 698d11dd9c78c")
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
        if not map_id.strip():
            st.error("Enter a map ID.")
            st.stop()
        map_id = map_id.strip()

        try:
            course = _load_course(map_id)
        except (ValueError, LookupError, OSError) as error:
            st.error(str(error))
            st.stop()

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
            "caveat": caveat,
            "warnings": warnings,
            "unbounded": budget == math.inf,
            "show_real_map": show_real_map,
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


main()
