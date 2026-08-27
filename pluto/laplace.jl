### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 378b5d23-5fb6-4bf8-be1d-a29aabdf4d61
using HypertextLiteral, PlutoUI, JSON, SparseArrays, Plots

# ╔═╡ 81fd7c66-5b5d-46e4-896e-8c9d0847ace6
md"""
# Interactive Laplace equation

Draw conductors with the mouse. The Laplace equation ∇²V = 0 is solved on everything they leave free, and the potential is redrawn on every stroke.

### Drawing

Each stroke becomes a conductor held at the voltage you set. Closed shapes can be drawn filled, so the interior is held at the same voltage as the boundary.

### Units

Everything in this notebook is dimensionless. Laplace's equation is scale invariant, so only ratios matter. Two plates 10 units long and 1 unit apart give the same potential whether that unit is a millimeter or a meter, and doubling every voltage doubles the whole solution.

### Model

* The drawing pad converts from canvas pixels to `x` and `y` before handing a stroke over, so everything after it works in the same units as `xs` and `ys`.
* Arrays are `V[i,j]` with `i` running over `ys` (row) and `j` over `xs` (column).
* `xs` and `ys` are ascending ranges with **equal spacing**. The solver averages the four neighbors of a cell with equal weight, which is only correct when the spacing in `x` matches the spacing in `y`.

### Boundary of the domain

The outer edge needs a condition of its own, and there are two to choose from.

**Grounded** holds it at V = 0, like a metal box around everything.

**Open** imposes ∂V/∂n = 0 instead, which approximates an unbounded domain. Use this one for an isolated charged conductor.
"""

# ╔═╡ 38f4ccdb-a719-470d-a149-ab62ad3fda66
md"""
### Grid

The physical domain. Everything downstream reads `xs`, `ys` and `h` from here, so this is the one place to change resolution or extent.

| Parameter | Description |
|:----------|:-----------|
| `n`       | cells per side; odd puts a grid point exactly at the origin |
| `L`       | half width of the square domain, so it spans `[-L, L]` |
| `h`       | grid spacing, and the natural unit for conductor thickness |

At `n = 201` the solve is fast enough to redraw on every stroke; drop to 101 if drawing feels sluggish.
"""

# ╔═╡ 560381af-e4f7-4c0e-a898-77bc97bf3157
begin
    n = 201
    L = 5.0
    xlims = (-L, L)
    ylims = (-L, L)

    xs = range(xlims[1], xlims[2]; length = n)
    ys = range(ylims[1], ylims[2]; length = n)
    h  = step(xs)

    isapprox(step(xs), step(ys); rtol = 1e-12) || error("the 5-point stencil assumes equal spacing in x and y")

    h
end;

# ╔═╡ 0457e038-2d49-4162-9a3e-452ae472bcc1
begin
    """
        Seg(a, b)

    One segment of a stroke, from `a` to `b`, with the constants that point-to-segment distance needs worked out once up front: the origin `a`, the direction `d = b - a`, and `invL2 = 1/|d|²`.

    A zero-length segment gets `invL2 = 0`, which makes `dist2` measure the distance to `a` without needing a special case.
    """
    struct Seg
        ax::Float64
        ay::Float64
        dx::Float64
        dy::Float64
        invL2::Float64
    end

    function Seg(a::NTuple{2,Float64}, b::NTuple{2,Float64})
        dx, dy = b[1] - a[1], b[2] - a[2]
        L2 = muladd(dx, dx, dy * dy)
        Seg(a[1], a[2], dx, dy, isfinite(inv(L2)) ? inv(L2) : 0.0)
    end

    """
        dist2(s, x, y)

    Squared distance from `(x, y)` to the nearest point of `s`. Squared, because the callers only ever compare it against a threshold, and staying squared avoids a `sqrt` per grid cell. The reciprocal length is already stored, so there is no division either.
    """
    @inline function dist2(s::Seg, x::Float64, y::Float64)
        wx = x - s.ax
        wy = y - s.ay
        t  = clamp(muladd(wx, s.dx, wy * s.dy) * s.invL2, 0.0, 1.0)
        ux = muladd(-t, s.dx, wx)
        uy = muladd(-t, s.dy, wy)
        muladd(ux, ux, uy * uy)
    end

    """
        Edge(y0, y1, x0, slope)

    One edge of a filled shape, for the row-by-row fill. Edges are stored pointing upward and span the rows `y0 ≤ y < y1`, crossing row `y` at `x0 + slope * (y - y0)`.

    Leaving the top row out is what keeps a shared vertex from being counted twice by the two edges that meet there.
    """
    struct Edge
        y0::Float64
        y1::Float64
        x0::Float64
        slope::Float64
    end

    """
        idxwin(v, lo, hi)

    The range of indices where the ascending vector `v` falls inside `[lo, hi]`, clamped to `v`'s own bounds so the result is always safe to index with. Used to turn a physical interval into the grid rows or columns it covers.

    On a `range` the index is arithmetic rather than a search, so this is O(1), not O(log n).
    """
    @inline function idxwin(v::AbstractRange, lo::Float64, hi::Float64)
        inv_s = inv(step(v))
        a = ceil(Int, (lo - first(v)) * inv_s) + firstindex(v)
        b = floor(Int, (hi - first(v)) * inv_s) + firstindex(v)
        max(a, firstindex(v)):min(b, lastindex(v))
    end

    @inline function idxwin(v::AbstractVector, lo::Float64, hi::Float64)
        a = searchsortedfirst(v, lo)
        b = searchsortedlast(v, hi)
        max(a, firstindex(v)):min(b, lastindex(v))
    end

    md"""
    **Geometry primitives.** `Seg` and `dist2` measure how far a grid point is from a stroke, which is what decides whether that point is part of a conductor. `Edge` describes one side of a filled shape for the interior fill. `idxwin` converts a physical interval into grid indices. Used by `Conductor` and `stamp!`.
    """
end

# ╔═╡ d6456a51-dbe9-4c23-a079-6097ffdd4a7a
begin
    """
        Conductor(pts, value, halfwidth; filled, closed)

    One conductor held at `value`, built from the vertices `pts`. A cell counts as part of it if it lies within `halfwidth` of the path, and also anywhere inside, if `filled`.

    `closed` or `filled` appends the first point again so the loop is explicit. `segs` drives the outline, `edges` the interior fill, and `bbox` is the outline's extent padded by `halfwidth`, which lets `on_outline` reject distant points at once.
    """
    struct Conductor
        pts::Vector{NTuple{2,Float64}}
        segs::Vector{Seg}
        edges::Vector{Edge}
        value::Float64
        filled::Bool
        halfwidth::Float64
        hw2::Float64
        bbox::NTuple{4,Float64}
    end

    function Conductor(pts::AbstractVector{<:NTuple{2,Real}}, value::Real, halfwidth::Real;
                       filled::Bool = false, closed::Bool = false)
        length(pts) ≥ 2 || throw(ArgumentError("need at least 2 points"))
        halfwidth > 0 || throw(ArgumentError("halfwidth must be positive"))
        isfinite(value) || throw(ArgumentError("value must be finite"))

        P = Vector{NTuple{2,Float64}}(undef, length(pts))
        xlo = ylo = Inf
        xhi = yhi = -Inf
        @inbounds for k in eachindex(pts)
            x, y = Float64(pts[k][1]), Float64(pts[k][2])
            (isfinite(x) && isfinite(y)) || throw(ArgumentError("non-finite coordinate"))
            P[k] = (x, y)
            xlo = min(xlo, x); xhi = max(xhi, x)
            ylo = min(ylo, y); yhi = max(yhi, y)
        end

        (closed || filled) && P[end] != P[1] && push!(P, P[1])

        hw = Float64(halfwidth)
        segs = [Seg(P[k], P[k+1]) for k in 1:length(P)-1]
        edges = (filled && length(P) ≥ 4) ? _build_edges(P) : Edge[]

        Conductor(P, segs, edges, Float64(value), filled && !isempty(edges),
                  hw, hw * hw, (xlo - hw, xhi + hw, ylo - hw, yhi + hw))
    end

    """
        _build_edges(P)

    The non-horizontal edges of the closed path `P`, ready for the row-by-row fill. Horizontal edges span no rows at all under the half-open rule, so they are dropped. Sorting by the lower endpoint lets the fill walk up the rows adding edges with a single forward pointer instead of rescanning.
    """
    function _build_edges(P::Vector{NTuple{2,Float64}})
        edges = Edge[]
        sizehint!(edges, length(P) - 1)
        @inbounds for k in 1:length(P)-1
            (x1, y1), (x2, y2) = P[k], P[k+1]
            y1 == y2 && continue
            y1 < y2 ? push!(edges, Edge(y1, y2, x1, (x2 - x1) / (y2 - y1))) :
                      push!(edges, Edge(y2, y1, x2, (x1 - x2) / (y1 - y2)))
        end
        sort!(edges; by = e -> e.y0)
    end

    """
        on_outline(c, x, y)

    True if `(x, y)` lies within `halfwidth` of the path, ignoring any interior fill. The padded bounding box rejects distant points before any segment is touched, and the segment loop returns on the first hit.
    """
    function on_outline(c::Conductor, x::Real, y::Real)
        xf, yf = Float64(x), Float64(y)
        xlo, xhi, ylo, yhi = c.bbox
        (xlo ≤ xf ≤ xhi && ylo ≤ yf ≤ yhi) || return false
        @inbounds for s in c.segs
            dist2(s, xf, yf) ≤ c.hw2 && return true
        end
        false
    end

    md"""
    **Conductors.** `Conductor` turns one stroke into the geometry the solver needs: its segments, its edges if filled, and a bounding box. `_build_edges` prepares the interior fill and `on_outline` tests a single point against the path. `stamp!` below writes them onto the grid.
    """
end

# ╔═╡ b14cb25e-801b-4256-9875-ccb47e438646
begin
    """
        stamp!(V, fixed, xs, ys, c)

    Write `c.value` into every grid cell the conductor covers and mark those cells fixed.

    Cost scales with the number of cells actually covered, not with the size of the bounding box. Each segment is walked along whichever axis it runs furthest in: if it is more horizontal than vertical the outer loop steps over columns, and at each column only the rows the band could reach are tested. That window has half-height `halfwidth·L/|dx|`, which is exact for the straight part and slightly generous over the rounded ends, and choosing the longer axis keeps it below `√2·halfwidth`. The exact distance test then trims whatever the window overshot.

    Scanning the bounding box instead falls apart on diagonals: a stroke at 45° across the grid has the entire domain as its box.
    """
    function stamp!(V, fixed, xs, ys, c::Conductor)
        c.filled && _fill_interior!(V, fixed, xs, ys, c)
        hw, hw2, val = c.halfwidth, c.hw2, c.value
        @inbounds for s in c.segs
            if iszero(s.invL2)
                _stamp_disk!(V, fixed, xs, ys, s.ax, s.ay, hw, hw2, val)
                continue
            end
            bx, by = s.ax + s.dx, s.ay + s.dy
            L = sqrt(muladd(s.dx, s.dx, s.dy * s.dy))
            if abs(s.dx) ≥ abs(s.dy)
                half, invd = hw * L / abs(s.dx), inv(s.dx)
                for j in idxwin(xs, min(s.ax, bx) - hw, max(s.ax, bx) + hw)
                    x  = xs[j]
                    t  = clamp((x - s.ax) * invd, 0.0, 1.0)
                    yc = muladd(t, s.dy, s.ay)
                    rows = idxwin(ys, yc - half, yc + half)
                    # The band is convex in each column, so once the run of
                    # covered rows has started, the first miss ends it.
                    started = false
                    for i in rows
                        if dist2(s, x, ys[i]) ≤ hw2
                            V[i,j] = val
                            fixed[i,j] = true
                            started = true
                        elseif started
                            break
                        end
                    end
                end
            else
                half, invd = hw * L / abs(s.dy), inv(s.dy)
                for i in idxwin(ys, min(s.ay, by) - hw, max(s.ay, by) + hw)
                    y  = ys[i]
                    t  = clamp((y - s.ay) * invd, 0.0, 1.0)
                    xc = muladd(t, s.dx, s.ax)
                    cols = idxwin(xs, xc - half, xc + half)
                    started = false
                    for j in cols
                        if dist2(s, xs[j], y) ≤ hw2
                            V[i,j] = val
                            fixed[i,j] = true
                            started = true
                        elseif started
                            break
                        end
                    end
                end
            end
        end
        V, fixed
    end

    """
        _fill_interior!(V, fixed, xs, ys, c)

    Fill the inside of a closed conductor, row by row.

    For each grid row, the edges crossing it give a set of x positions, and sorting them pairs the crossings up: inside runs from the first to the second, the third to the fourth, and so on. Edges are pre-sorted by their lower endpoint, so walking up the rows adds each one with a single forward pointer, and finished edges are retired by swapping in the last active one.
    """
    function _fill_interior!(V, fixed, xs, ys, c::Conductor)
        edges = c.edges
        isempty(edges) && return
        # bbox carries the halfwidth padding, which the outline needs and the
        # fill does not; trimming it skips rows that have no crossings at all
        hw = c.halfwidth
        rows = idxwin(ys, c.bbox[3] + hw, c.bbox[4] - hw)
        isempty(rows) && return

        val    = c.value
        nedges = length(edges)
        ptr    = 1
        active = Int[]
        xc     = Float64[]
        sizehint!(active, nedges)
        sizehint!(xc, nedges)

        @inbounds for i in rows
            y = ys[i]
            while ptr ≤ nedges && edges[ptr].y0 ≤ y
                push!(active, ptr); ptr += 1
            end
            k = 1
            while k ≤ length(active)
                if edges[active[k]].y1 ≤ y
                    active[k] = active[end]; pop!(active)
                else
                    k += 1
                end
            end
            isempty(active) && continue

            empty!(xc)
            for a in active
                ed = edges[a]
                push!(xc, muladd(ed.slope, y - ed.y0, ed.x0))
            end
            # Almost every row of a convex shape has exactly two crossings, so
            # that case is a compare and swap; the general sort is insertion,
            # which beats the adaptive default on a handful of elements.
            if length(xc) == 2
                xc[1] > xc[2] && ((xc[1], xc[2]) = (xc[2], xc[1]))
            else
                sort!(xc; alg = InsertionSort)
            end
            for k in 1:2:length(xc)-1
                for j in idxwin(xs, xc[k], xc[k+1])
                    V[i,j] = val
                    fixed[i,j] = true
                end
            end
        end
    end
    
    """
        _stamp_disk!(V, fixed, xs, ys, cx, cy, hw, hw2, val)

    Fill a disk of radius `hw` about `(cx, cy)`. Used for the degenerate segments that a repeated point produces. The chord length is known exactly for each column, so no distance test is needed at all.
    """
    @inline function _stamp_disk!(V, fixed, xs, ys, cx, cy, hw, hw2, val)
        @inbounds for j in idxwin(xs, cx - hw, cx + hw)
            d = xs[j] - cx
            r = sqrt(max(hw2 - d * d, 0.0))
            for i in idxwin(ys, cy - r, cy + r)
                V[i,j] = val
                fixed[i,j] = true
            end
        end
    end

    md"**Stamping.** `stamp!` writes one conductor onto the grid, visiting only the cells near the stroke rather than scanning the whole region it spans."
end

# ╔═╡ ab257c68-8499-405b-8775-ed51a22b535a
begin
    """
        build_bcs(xs, ys, conductors) -> (; V, fixed)

    Fresh arrays with every conductor stamped onto them in order, so where two overlap the later one wins.

    `fixed` is a dense `Matrix{Bool}` rather than a `BitMatrix`. A bit array is eight times smaller, but every read in the relaxation inner loop then costs a shift and a mask instead of a single byte load, which is the wrong trade for the hottest loop in the notebook.
    """
    function build_bcs(xs, ys, conductors)
        V     = zeros(Float64, length(ys), length(xs))
        fixed = fill(false, length(ys), length(xs))
        for c in conductors
            stamp!(V, fixed, xs, ys, c)
        end
        (V = V, fixed = fixed)
    end

    md"""
    **Boundary conditions.** `build_bcs` allocates the potential grid and the mask of held cells, then stamps every conductor onto them. The solver takes it from there.
    """
end

# ╔═╡ 29c6ccbc-efda-4925-8cf2-b8d85e80c7f8
begin
    """
        drawpad(; xlims, ylims, px, gridstep)

    A canvas you draw conductors on, bound with `@bind` like any other Pluto widget. Set the voltage and shape in the toolbar, then drag on the canvas. The bond fires only when a stroke is completed, undone, or cleared, so fiddling with the toolbar costs nothing.

    The value is a JSON string, one object per stroke:

        [{"v": 1.0, "closed": false, "filled": false, "pts": [[x, y], ...]}, ...]

    Coordinates are physical, in the units of `xlims` and `ylims`, and rounded to 1e-4. `parse_strokes` turns this into `Conductor`s.

    Rendering is split across three stacked canvases: the grid is drawn once, finished strokes are stamped onto the middle layer and never touched again, and only the stroke in progress is redrawn, so cost per frame stays flat as conductors accumulate.

    If the pad is placed inside an element of class `pad-host` that also contains a checkbox under `.boundary`, the pad reads that checkbox to style its frame and caption. The pad only reads it, so the checkbox keeps its own separate bond and toggling it never re-renders the canvas or loses your strokes.
    """
    function drawpad(; xlims = (-5.0, 5.0), ylims = (-5.0, 5.0), px = 560, gridstep = 0.5)
        config = JSON.json(Dict(
            "xmin" => Float64(xlims[1]), "xmax" => Float64(xlims[2]),
            "ymin" => Float64(ylims[1]), "ymax" => Float64(ylims[2]),
            "W" => px, "H" => px, "gs" => Float64(gridstep)))

        @htl("""
        <div class="drawpad" data-config=$(config)
             style="display:inline-block;width:$(px)px;font-family:sans-serif;font-size:13px">
          <div class="bar" style="margin-bottom:6px;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
            <label>V <input class="val" type="number" value="1" step="0.25" style="width:4.5em"></label>
            <select class="mode">
              <option value="free">freehand</option>
              <option value="line">line</option>
              <option value="rect">rectangle</option>
              <option value="circle">circle</option>
            </select>
            <label><input class="fill" type="checkbox"> filled</label>
            <label><input class="snap" type="checkbox" checked> snap</label>
            <button class="undo" type="button">undo</button>
            <button class="clear" type="button">clear</button>
            <span class="pos" style="margin-left:auto;min-width:7.5em;text-align:right;white-space:nowrap;opacity:0.6;font-variant-numeric:tabular-nums"></span>
          </div>
          <div class="stack" style="position:relative;width:$(px)px;height:$(px)px;line-height:0">
            <canvas class="grid" style="display:block"></canvas>
            <canvas class="ink"  style="position:absolute;left:0;top:0"></canvas>
            <canvas class="live" style="position:absolute;left:0;top:0;cursor:crosshair;touch-action:none"></canvas>
          </div>
          <div class="foot" style="margin-top:5px;opacity:0.6;font-size:12px;line-height:1.3">
            <span class="bmode"></span>
          </div>
          <script>
          const wrapper = currentScript.parentElement
          const cfg = JSON.parse(wrapper.dataset.config)
          const bar = wrapper.querySelector(".bar")
          const stack = wrapper.querySelector(".stack")
          const gridC = wrapper.querySelector(".grid")
          const inkC = wrapper.querySelector(".ink")
          const liveC = wrapper.querySelector(".live")
          const bmode = wrapper.querySelector(".bmode")
          const valbox = wrapper.querySelector(".val")
          const modesel = wrapper.querySelector(".mode")
          const fillbox = wrapper.querySelector(".fill")
          const snapbox = wrapper.querySelector(".snap")
          const posout = wrapper.querySelector(".pos")

          // Toolbar input/change events bubble up to `wrapper`, where Pluto's
          // @bind listener reads them as a new bond value and re-runs the
          // notebook on every keystroke. Stop them here. publish() dispatches
          // straight on wrapper, so it still gets through.
          const stop = e => e.stopPropagation()
          bar.addEventListener("input", stop)
          bar.addEventListener("change", stop)

          const W = cfg.W, H = cfg.H, gs = cfg.gs
          const xmin = cfg.xmin, xmax = cfg.xmax, ymin = cfg.ymin, ymax = cfg.ymax
          const span = Math.max(xmax - xmin, ymax - ymin)
          const minStep2 = (span / 400) ** 2   // reject samples closer than this
          const flatTol2 = (span / 800) ** 2   // merge samples that barely bend the path
          const arcTol = span / 400            // circle faceting tolerance

          // Grays are semi-transparent so they read against a light or a dark
          // page without being told which one they are on.
          const GRID_FINE = "rgba(128,128,128,0.28)"
          const GRID_AXIS = "rgba(128,128,128,0.6)"
          const FRAME = "rgba(128,128,128,0.85)"

          const x2px = x => (x - xmin) / (xmax - xmin) * W
          const y2py = y => (ymax - y) / (ymax - ymin) * H
          const px2x = p => xmin + p / W * (xmax - xmin)
          const py2y = p => ymax - p / H * (ymax - ymin)

          // Half-pixel offset puts a 1px line on a pixel center so it stays
          // crisp; the clamp keeps the lines at the domain edges from landing
          // one pixel outside the canvas, where they vanish.
          const gx = x => Math.min(Math.max(Math.round(x2px(x)), 0), W - 1) + 0.5
          const gy = y => Math.min(Math.max(Math.round(y2py(y)), 0), H - 1) + 0.5

          let dpr = window.devicePixelRatio || 1
          let gctx, ictx, lctx
          function fit(c) {
              c.width = Math.round(W * dpr); c.height = Math.round(H * dpr)
              c.style.width = W + "px"; c.style.height = H + "px"
              const g = c.getContext("2d")
              g.setTransform(dpr, 0, 0, dpr, 0, 0)
              return g
          }
          function fitAll() { gctx = fit(gridC); ictx = fit(inkC); lctx = fit(liveC) }

          // points live interleaved: p = [x0,y0,x1,y1,...]
          let strokes = []
          let current = null
          let anchor = null
          let boundaryOpen = false

          const colorFor = v => v > 0 ? "#c0392b" : (v < 0 ? "#2471a3" : "#808080")
          const snapTo = (v, on) => on ? Math.round(v / gs) * gs : v

          function setPen(ctx, s) {
              ctx.strokeStyle = colorFor(s.v)
              ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.lineJoin = "round"
          }
          function drawStroke(ctx, s) {
              const p = s.p
              if (p.length < 4) return
              ctx.beginPath()
              ctx.moveTo(x2px(p[0]), y2py(p[1]))
              for (let k = 2; k < p.length; k += 2) ctx.lineTo(x2px(p[k]), y2py(p[k+1]))
              if (s.closed) ctx.closePath()
              if (s.filled) { ctx.fillStyle = colorFor(s.v) + "55"; ctx.fill() }
              setPen(ctx, s); ctx.stroke()
          }

          // Whatever is actually painted behind the canvas, found by walking up
          // until something is not transparent. The dash gaps are filled with
          // this, so the frame looks the same on a light or a dark page.
          function pageBg() {
              let el = stack
              while (el) {
                  const c = getComputedStyle(el).backgroundColor
                  if (c && c !== "transparent" && !/,\\s*0\\)\$/.test(c)) return c
                  el = el.parentElement
              }
              return "#ffffff"
          }

          // The frame is painted, not bordered: CSS gives each side its own dash
          // phase, so the corners never match. One rect path keeps the phase
          // continuous, and a period that divides the side exactly puts a dash
          // centered on every corner. The background pass underneath hides the
          // gridline ends that would otherwise show through the gaps.
          function drawFrame() {
              const lw = 2, o = lw / 2, side = W - lw, rest = H - lw
              gctx.save()
              gctx.lineWidth = lw
              gctx.lineJoin = "miter"
              if (boundaryOpen) {
                  gctx.strokeStyle = pageBg()
                  gctx.strokeRect(o, o, side, rest)
                  const period = side / Math.max(1, Math.round(side / 12))
                  const dash = period * 0.62
                  gctx.setLineDash([dash, period - dash])
                  gctx.lineDashOffset = -dash / 2
              }
              gctx.strokeStyle = FRAME
              gctx.strokeRect(o, o, side, rest)
              gctx.restore()
          }

          // static layer: two paths total, not two per gridline
          function drawGrid() {
              const line = (a, b, c, d) => { gctx.moveTo(a, b); gctx.lineTo(c, d) }
              gctx.clearRect(0, 0, W, H)
              gctx.lineWidth = 1; gctx.strokeStyle = GRID_FINE
              gctx.beginPath()
              for (let k = Math.ceil(xmin/gs); k <= Math.floor(xmax/gs); k++) line(gx(k*gs), 0, gx(k*gs), H)
              for (let k = Math.ceil(ymin/gs); k <= Math.floor(ymax/gs); k++) line(0, gy(k*gs), W, gy(k*gs))
              gctx.stroke()
              gctx.strokeStyle = GRID_AXIS; gctx.beginPath()
              line(gx(0), 0, gx(0), H); line(0, gy(0), W, gy(0))
              gctx.stroke()
              drawFrame()
          }

          // committed layer: stamped once per finished stroke, rebuilt only on
          // undo, clear, or a change of pixel ratio
          function rebuildInk() {
              ictx.clearRect(0, 0, W, H)
              for (const s of strokes) drawStroke(ictx, s)
          }

          // live layer: a freehand stroke only ever grows, so its new segments
          // are appended without clearing. A shape replaces its geometry, so it
          // needs a full redraw, but only when that geometry actually changed.
          let raf = 0, lastPos = null, drawnUpTo = 0, liveDirty = false, shapeDirty = false
          const schedule = () => { if (raf === 0) raf = requestAnimationFrame(frame) }
          function clearLive() {
              if (liveDirty) { lctx.clearRect(0, 0, W, H); liveDirty = false }
          }
          function frame() {
              raf = 0
              if (lastPos) posout.textContent =
                  "(" + lastPos[0].toFixed(2) + ", " + lastPos[1].toFixed(2) + ")"
              if (current === null) { clearLive(); drawnUpTo = 0; return }
              const p = current.p
              if (current.grow) {
                  if (p.length - 2 > drawnUpTo) {
                      setPen(lctx, current)
                      lctx.beginPath()
                      lctx.moveTo(x2px(p[drawnUpTo]), y2py(p[drawnUpTo+1]))
                      for (let k = drawnUpTo + 2; k < p.length; k += 2)
                          lctx.lineTo(x2px(p[k]), y2py(p[k+1]))
                      lctx.stroke()
                      drawnUpTo = p.length - 2
                      liveDirty = true
                  }
              } else if (shapeDirty) {
                  lctx.clearRect(0, 0, W, H)
                  drawStroke(lctx, current)
                  liveDirty = true; shapeDirty = false
              }
          }

          // Read-only view of a checkbox that lives outside this widget and owns
          // its own bond, so toggling it never re-renders the canvas.
          function showBoundary(open) {
              boundaryOpen = open
              drawGrid()
              bmode.innerHTML = "<b>" + (open
                  ? "open edge</b>: field lines pass through (∂V/∂n = 0)"
                  : "grounded edge</b>: field lines end on it (V = 0)")
          }
          let unhookBoundary = null, tries = 0
          function hookBoundary() {
              const host = wrapper.closest(".pad-host")
              const box = host && host.querySelector(".boundary input[type=checkbox]")
              if (!box) {
                  if (tries++ < 60) requestAnimationFrame(hookBoundary)
                  else showBoundary(false)
                  return
              }
              const upd = () => showBoundary(box.checked)
              box.addEventListener("change", upd)
              unhookBoundary = () => box.removeEventListener("change", upd)
              upd()
          }

          // getBoundingClientRect forces a layout flush, so cache it and drop
          // the cache whenever the page could have moved
          let rect = null
          const dropRect = () => { rect = null }
          function xy(e) {
              if (rect === null) rect = liveC.getBoundingClientRect()
              return [px2x(e.clientX - rect.left), py2y(e.clientY - rect.top)]
          }

          // Distances stay squared: both tests are comparisons against a
          // threshold, so no sqrt is needed, and cross-multiplying the
          // perpendicular distance removes the division too.
          function addPoint(x, y) {
              const p = current.p, n = p.length
              const lx = p[n-2], ly = p[n-1]
              const ex = x - lx, ey = y - ly
              if (ex*ex + ey*ey < minStep2) return
              if (n >= 4 && n - 2 > drawnUpTo) {
                  const ax = p[n-4], ay = p[n-3]
                  const dx = x - ax, dy = y - ay
                  const cross = (lx - ax)*dy - (ly - ay)*dx
                  if (cross*cross < flatTol2 * (dx*dx + dy*dy)) {
                      p[n-2] = x; p[n-1] = y   // previous sample was redundant
                      return
                  }
              }
              p.push(x, y)
          }

          function shapePts(mode, snap, a, bx, by) {
              if (mode === "line") return [a[0], a[1], bx, by]
              if (mode === "rect") return [a[0],a[1], bx,a[1], bx,by, a[0],by]
              let r = Math.hypot(bx - a[0], by - a[1])
              if (snap) r = Math.max(Math.round(r/gs)*gs, gs)
              if (r <= 0) return null
              const c = Math.max(-1, Math.min(1, 1 - arcTol/r))
              const N = Math.min(360, Math.max(24, Math.ceil(Math.PI/Math.acos(c))))
              const out = []
              for (let k = 0; k < N; k++) {
                  const t = 2*Math.PI*k/N
                  out.push(a[0] + r*Math.cos(t), a[1] + r*Math.sin(t))
              }
              return out
          }

          // each finished stroke serializes once; publish just reassembles
          function publish() {
              wrapper.value = JSON.stringify(strokes.map(s => s.out))
              wrapper.dispatchEvent(new CustomEvent("input"))
          }

          let lastBx = NaN, lastBy = NaN

          liveC.addEventListener("pointerdown", e => {
              e.preventDefault(); dropRect(); liveC.setPointerCapture(e.pointerId)
              // settings are frozen at pointerdown, so changing the toolbar
              // mid-drag cannot leave the stroke half in one mode
              const mode = modesel.value
              const filled = fillbox.checked
              const snap = snapbox.checked
              const v = parseFloat(valbox.value)
              const q = xy(e)
              anchor = mode === "free" ? q : [snapTo(q[0], snap), snapTo(q[1], snap)]
              current = {
                  v: isFinite(v) ? v : 0.0,
                  mode: mode, filled: filled, snap: snap,
                  closed: mode !== "line" && (mode !== "free" || filled),
                  grow: mode === "free" && !filled,
                  p: [anchor[0], anchor[1]]
              }
              drawnUpTo = 0; lastPos = q
              lastBx = NaN; lastBy = NaN; shapeDirty = true
              clearLive(); schedule()
          })

          liveC.addEventListener("pointermove", e => {
              if (current === null) { lastPos = xy(e); schedule(); return }
              if (current.mode === "free") {
                  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e]
                  for (const ev of evs) { const w = xy(ev); addPoint(w[0], w[1]); lastPos = w }
              } else {
                  const q = xy(e); lastPos = q
                  // With snap on, most moves land on the same lattice point, so
                  // the geometry is unchanged and there is nothing to redraw.
                  const bx = snapTo(q[0], current.snap), by = snapTo(q[1], current.snap)
                  if (bx !== lastBx || by !== lastBy) {
                      const pts = shapePts(current.mode, current.snap, anchor, bx, by)
                      if (pts) { current.p = pts; shapeDirty = true }
                      lastBx = bx; lastBy = by
                  }
              }
              schedule()
          }, { passive: true })

          function finish() {
              if (current === null) return
              const s = current
              current = null; anchor = null
              schedule()
              if (s.p.length < 4) return
              const pts = []
              for (let k = 0; k < s.p.length; k += 2)
                  pts.push([Math.round(s.p[k]*1e4)/1e4, Math.round(s.p[k+1]*1e4)/1e4])
              s.out = { v: s.v, closed: s.closed, filled: s.filled, pts: pts }
              strokes.push(s)
              drawStroke(ictx, s)
              publish()
          }
          liveC.addEventListener("pointerup", finish)
          liveC.addEventListener("pointercancel", finish)

          const onKey = e => {
              if (e.key === "Escape" && current) {
                  current = null; anchor = null; schedule()
              }
          }
          window.addEventListener("keydown", onKey)

          wrapper.querySelector(".undo").addEventListener("click", () => {
              if (strokes.length === 0) return
              strokes.pop(); rebuildInk(); publish()
          })
          wrapper.querySelector(".clear").addEventListener("click", () => {
              if (strokes.length === 0) return
              strokes = []; rebuildInk(); publish()
          })

          let mq = null
          function onDpr() {
              dpr = window.devicePixelRatio || 1
              fitAll(); drawGrid(); rebuildInk(); liveDirty = false; schedule(); watchDpr()
          }
          function watchDpr() {
              if (mq) mq.removeEventListener("change", onDpr)
              mq = window.matchMedia("(resolution: " + (window.devicePixelRatio || 1) + "dppx)")
              mq.addEventListener("change", onDpr)
          }

          // pageBg is sampled when the frame is painted, so a theme change has
          // to trigger a repaint. Pluto's own light/dark switch sets an
          // attribute on <html> instead of going through the media query.
          const themeMq = window.matchMedia("(prefers-color-scheme: dark)")
          const onTheme = () => drawGrid()
          themeMq.addEventListener("change", onTheme)
          const themeObs = new MutationObserver(onTheme)
          themeObs.observe(document.documentElement, {
              attributes: true, attributeFilter: ["class", "data-theme"]
          })

          const ro = new ResizeObserver(dropRect)
          ro.observe(stack)
          window.addEventListener("scroll", dropRect, true)
          window.addEventListener("resize", dropRect)

          invalidation.then(() => {
              if (raf !== 0) cancelAnimationFrame(raf)
              ro.disconnect()
              themeObs.disconnect()
              themeMq.removeEventListener("change", onTheme)
              window.removeEventListener("keydown", onKey)
              window.removeEventListener("scroll", dropRect, true)
              window.removeEventListener("resize", dropRect)
              if (mq) mq.removeEventListener("change", onDpr)
              if (unhookBoundary) unhookBoundary()
          })

          fitAll(); drawGrid(); watchDpr(); hookBoundary(); publish()
          </script>
        </div>
        """)
    end

    md"**Drawing pad.** `drawpad` builds the canvas you draw on and hands each finished stroke back as JSON, in the same coordinates as the grid. `parse_strokes` turns that into conductors."
end

# ╔═╡ 6049c3fc-0b98-41a8-9d7b-e8a6651c8a54
begin
    """
        parse_strokes(raw; halfwidth) -> Vector{Conductor}

    Read the pad's JSON, `[{"v":…, "closed":…, "filled":…, "pts":[[x,y],…]}, …]`, into conductors of the given half width.

    The input comes back from a browser, so nothing in it is trusted: bad JSON, missing keys, wrong types, and degenerate strokes are all skipped rather than raised, and an empty pad gives an empty vector. That matters because a single malformed stroke would otherwise take out the cell and blank the plot.
    """
    function parse_strokes(raw::AbstractString; halfwidth::Real)
        out = Conductor[]
        all(isspace, raw) && return out
        parsed = try
            JSON.parse(raw)
        catch
            return out
        end
        parsed isa AbstractVector || return out
        sizehint!(out, length(parsed))
        for s in parsed
            s isa AbstractDict || continue
            v = _f64(get(s, "v", 0.0))
            isfinite(v) || continue
            pts = _clean_pts(get(s, "pts", nothing), halfwidth)
            length(pts) ≥ 2 || continue
            push!(out, Conductor(pts, v, halfwidth;
                                 filled = get(s, "filled", false) === true,
                                 closed = get(s, "closed", false) === true))
        end
        out
    end

    """
        _f64(x) -> Float64

    Convert to `Float64` if `x` is a number, and return `NaN` otherwise. Lets a single `isfinite` check reject nulls, strings, and nested objects along with the actual non-finite numbers, instead of each needing its own guard.
    """
    _f64(x::Real) = Float64(x)
    _f64(::Any) = NaN

    """
        _clean_pts(rawpts, halfwidth) -> Vector{NTuple{2,Float64}}

    Convert one stroke's points to concrete `Float64` tuples, dropping anything non-numeric or non-finite and collapsing consecutive duplicates, which the pad's rounding to 1e-4 can produce.

    This also acts as a function barrier. `JSON.parse` hands back `Any`-typed containers, so element access here is dynamic, but that cost is paid once per point and never reaches the stamping loops, which see a concrete `Vector{Seg}`.
    """
    function _clean_pts(rawpts, halfwidth)
        rawpts isa AbstractVector || return NTuple{2,Float64}[]
        tol2 = (1e-6 * halfwidth)^2
        pts  = NTuple{2,Float64}[]
        sizehint!(pts, length(rawpts))
        px = py = NaN
        for p in rawpts
            (p isa AbstractVector && length(p) ≥ 2) || continue
            x, y = _f64(p[1]), _f64(p[2])
            (isfinite(x) && isfinite(y)) || continue
            if isempty(pts) || (x - px)^2 + (y - py)^2 > tol2
                push!(pts, (x, y))
                px = x; py = y
            end
        end
        pts
    end

    md"**Stroke parsing.** `parse_strokes` turns the pad's JSON into conductors, skipping anything malformed. `_clean_pts` and `_f64` do the per-point conversion."
end

# ╔═╡ 5b3b4779-d699-400e-a500-618c9bbefebb
begin
    """
        relax!(V, fixed; tol, maxiter, ω, check_every, open)

    Red-black successive over-relaxation. Each sweep visits the two color classes in turn, so the result does not depend on the order cells are traversed, and the columns within a class can run in parallel. Leaving `ω` as `nothing` picks the optimal value for the grid size, giving sweep counts that grow like `n` instead of `n²`. Cells flagged in `fixed` never change.

    Convergence is tested on the true residual every `check_every` sweeps, relative to the largest conductor voltage.

    Mutates `V`. Returns `(; V, iters, residual, converged, ω)`.
    """
    function relax!(V::AbstractMatrix{T}, fixed::AbstractMatrix{Bool};
                    tol = 1e-8, maxiter = 5_000, ω = nothing,
                    check_every = 10, open = false) where {T<:AbstractFloat}

        axes(V) == axes(fixed) || throw(DimensionMismatch("V and fixed must agree"))
        n1, n2 = size(V)
        (n1 ≥ 3 && n2 ≥ 3) || throw(ArgumentError("grid must be at least 3x3"))

        ρ = (cospi(1 / (n1 - 1)) + cospi(1 / (n2 - 1))) / 2
        w = T(something(ω, 2 / (1 + sqrt(1 - ρ^2))))
        0 < w < 2 || throw(ArgumentError("ω must lie in (0,2)"))

        # By the maximum principle the extremes of a harmonic function sit on the
        # boundary, so this scale is set by the conductors and never moves.
        scale = max(maximum(abs, V), eps(T))

        res = T(Inf)
        for iter in 1:maxiter
            open && mirror_edges!(V, fixed)
            for color in 0:1
                # Threads are safe without any coordination: a color pass writes
                # only cells of its own parity and reads only the other parity,
                # so no thread can read a cell another is writing. :static keeps
                # the chunking fixed, since the work per column is uniform and
                # dynamic scheduling would only add overhead.
                Threads.@threads :static for j in 2:n2-1
                    @inbounds for i in (2 + (color + j) % 2):2:n1-1
                        fixed[i,j] && continue
                        σ = (V[i-1,j] + V[i+1,j] + V[i,j-1] + V[i,j+1]) / 4
                        V[i,j] = muladd(w, σ - V[i,j], V[i,j])
                    end
                end
            end
            if iter % check_every == 0
                res = residual(V, fixed)
                if res ≤ tol * scale
                    return (; V, iters = iter, residual = res, converged = true, ω = w)
                end
            end
        end
        (; V, iters = maxiter, residual = res, converged = false, ω = w)
    end

    """
        mirror_edges!(V, fixed)

    Zero normal derivative on the outer ring, copying each edge cell from its inward neighbor, which approximates an unbounded domain. Cells belonging to a conductor are skipped. Rows are handled before columns so the corners settle consistently.
    """
    function mirror_edges!(V, fixed)
        n1, n2 = size(V)
        @inbounds for j in 1:n2
            fixed[1,j]  || (V[1,j]  = V[2,j])
            fixed[n1,j] || (V[n1,j] = V[n1-1,j])
        end
        @inbounds for i in 1:n1
            fixed[i,1]  || (V[i,1]  = V[i,2])
            fixed[i,n2] || (V[i,n2] = V[i,n2-1])
        end
        V
    end

    """
        residual(V, fixed)

    Largest amount by which any free cell differs from the average of its four neighbors. Zero exactly when `V` satisfies the discrete Laplace equation, so it measures convergence regardless of which solver produced `V`.

    Measured separately rather than read off the sweep, because over-relaxation overshoots the average by a factor `ω` and would report convergence early.
    """
    function residual(V::AbstractMatrix{T}, fixed) where {T}
        r = zero(T)
        n1, n2 = size(V)
        @inbounds for j in 2:n2-1, i in 2:n1-1
            fixed[i,j] && continue
            r = max(r, abs((V[i-1,j] + V[i+1,j] + V[i,j-1] + V[i,j+1]) / 4 - V[i,j]))
        end
        r
    end

    """
        solve_relax(V0, fixed; kwargs...)

    Copying wrapper around `relax!`, leaving `V0` untouched so Pluto's reactivity stays honest. Keyword arguments pass straight through.
    """
    solve_relax(V0, fixed; kw...) = relax!(copy(V0), fixed; kw...)

    md"**Solver.** Sweeps the grid replacing each free cell with the average of its four neighbors, held at the conductor voltages, until nothing moves. Each step overshoots the average slightly to speed up convergence. `mirror_edges!` applies the open boundary condition and `residual` measures how converged the potential is."
end

# ╔═╡ 1375973e-9111-4718-8fb8-d1caca8c8769
begin
    gr()
    default(framestyle = :box, grid = false, dpi = 130, size = (720, 620))
end

# ╔═╡ 3a0f0a2b-e142-48ea-8549-0895539213e9
# md"""**Open boundary** $(@bind open_boundary CheckBox(default = false))"""
@htl("""
<div class="pad-host" style="display:inline-block">
  $(@bind raw_strokes drawpad(xlims = xlims, ylims = ylims, px = 560, gridstep = 0.5))
  <div class="boundary" style="margin-top:2px;display:flex;gap:6px;align-items:center;font-family:sans-serif;font-size:13px">
    $(@bind open_boundary CheckBox(default = false))
    <span>open boundary</span>
  </div>
</div>
""")

# ╔═╡ e269f4de-b900-469e-ad3a-d09b55c28adc
conductors = parse_strokes(raw_strokes; halfwidth = 1.0 * h);
# conductors = vcat(parse_strokes(raw_strokes; halfwidth = 1.0 * h), ncku);

# ╔═╡ 9a3aa3c3-2c91-47a7-b3b8-88a0dcfe5104
bcs = build_bcs(xs, ys, conductors);

# ╔═╡ 11431ed4-6b55-493a-86f1-b199f2a1a665
sol = solve_relax(bcs.V, bcs.fixed; open = open_boundary, tol = 1e-6, maxiter = 40_000);

# ╔═╡ 79176171-a27e-4082-8029-4a7abc0b16eb
let
    V = sol.V
    vmax = max(maximum(abs, V), 1e-12)

    heatmap(xs, ys, V, aspect_ratio = 1, xlims = (-L, L), ylims = (-L, L), 
            color = :coolwarm, clims = (-vmax, vmax), colorbar = true, right_margin = 4Plots.mm, legend = false, xlabel = "x", ylabel = "y", title = "Electric potential V(x,y)")
    contour!(xs, ys, V, levels = 30, linewidth = 0.5, linecolor = :black, 
             colorbar_entry = false)
end

# ╔═╡ b481f7a6-7d5c-4909-a22b-f10b44973690
ncku = let
    hw = 1.5 * h            # diagonals need more than the 1.0h minimum
    y0, y1 = -1.5, 1.5      # baseline and cap height
    E(pts, v) = Conductor(pts, v, hw)
    [
        # N
        E([(-4.5, y0), (-4.5, y1)], 1.0),
        E([(-4.25, y1), (-3.25, y0)], -1.0),
        E([(-3.0, y0), (-3.0, y1)], 1.0),

        # C
        E([(-0.5, 1.0), (-1.5, y1), (-2.0, 1.0)], 1.0),
        E([(-2.0, 0.75), (-2.0, -0.75)], -1.0),
        E([(-2.0, -1.0), (-1.5, y0), (-0.5, -1.0)], 1.0),

        # K
        E([(0.5, y0), (0.5, y1)], -1.0),
        E([(2.0, 1.1), (0.9, 0.0), (2.0, -1.1)], 1.0),

        # U
        E([(3.0, y1), (3.0, -1.0), (3.55, y0)], -1.0),
        E([(3.95, y0), (4.5, -1.0), (4.5, y1)], +1.0),
    ]
end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
HypertextLiteral = "~1.0.0"
JSON = "~1.6.1"
Plots = "~1.41.6"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "619eb1aabce1be1747a309bd6faa7997f7dde06b"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.1+2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╠═378b5d23-5fb6-4bf8-be1d-a29aabdf4d61
# ╟─81fd7c66-5b5d-46e4-896e-8c9d0847ace6
# ╟─38f4ccdb-a719-470d-a149-ab62ad3fda66
# ╠═560381af-e4f7-4c0e-a898-77bc97bf3157
# ╟─0457e038-2d49-4162-9a3e-452ae472bcc1
# ╟─d6456a51-dbe9-4c23-a079-6097ffdd4a7a
# ╟─b14cb25e-801b-4256-9875-ccb47e438646
# ╟─ab257c68-8499-405b-8775-ed51a22b535a
# ╟─29c6ccbc-efda-4925-8cf2-b8d85e80c7f8
# ╟─6049c3fc-0b98-41a8-9d7b-e8a6651c8a54
# ╠═e269f4de-b900-469e-ad3a-d09b55c28adc
# ╠═9a3aa3c3-2c91-47a7-b3b8-88a0dcfe5104
# ╟─5b3b4779-d699-400e-a500-618c9bbefebb
# ╠═11431ed4-6b55-493a-86f1-b199f2a1a665
# ╠═1375973e-9111-4718-8fb8-d1caca8c8769
# ╟─3a0f0a2b-e142-48ea-8549-0895539213e9
# ╠═79176171-a27e-4082-8029-4a7abc0b16eb
# ╠═b481f7a6-7d5c-4909-a22b-f10b44973690
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
