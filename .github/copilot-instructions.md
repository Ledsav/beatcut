# BeatCut — Copilot Instructions

## Project Context

Local Flask web app: user uploads photos + a song, gets back a beat-synced, face-centered MP4 slideshow.
See `CLAUDE.md` for full architecture, pipeline details, and common failure modes.

**Stack**: Python 3.12 / Flask / moviepy 1.0.3 / OpenCV / librosa — Vanilla JS (no frameworks) — Plain CSS (no frameworks)

---

## Frontend & UI/UX Design Intelligence

Apply these rules automatically whenever working on `static/`, `templates/`, or any UI/frontend task.

### Design Thinking (Before Writing Any UI Code)

1. **Understand context first** — What problem does this interface solve? Who uses it? (BeatCut users are creators uploading photos + music to get a video slideshow.)
2. **Commit to a bold aesthetic direction** — Pick an extreme and execute it with full intention. Half-hearted design is worse than none.
3. **Ask: what makes this UNFORGETTABLE?** — One memorable detail beats ten forgettable ones.
4. **Match complexity to vision** — Maximalist needs elaborate animations. Minimal needs precision spacing and restraint.

### Typography

- **Never use**: Inter, Roboto, Arial, system-ui, Space Grotesk as primary fonts
- **Do use**: Distinctive, characterful font pairings — a strong display font + a refined body font
- BeatCut already uses **Bebas Neue** (headings) + **Space Mono** (body) — preserve and build on this identity
- Line height: 1.5–1.75 for body text; limit line length to 65–75 chars

### Color & Theme

- BeatCut design tokens (never change these):
  ```
  --bg: #060608        /* page background */
  --surface: #0e0e12   /* cards */
  --accent: #c8ff00    /* lime green — buttons, progress, stats */
  --danger: #ff3c6e    /* cancel, error */
  --text: #e8e8f0
  --muted: #6b6b80
  ```
- Always use CSS variables — never hardcode hex values that duplicate these tokens
- Dominant color + sharp accent outperforms evenly-distributed palettes

### Motion & Animation

- Use CSS transitions for micro-interactions (150–300ms)
- Use `transform` and `opacity` — never animate `width`, `height`, `top`, `left`
- One well-orchestrated entrance animation > scattered micro-interactions
- Always include `@media (prefers-reduced-motion: reduce)` overrides
- Loading states: skeleton screens or spinners — never leave blank space

### Layout & Spatial Composition

- Unexpected layouts beat safe grids — asymmetry, overlap, diagonal flow are encouraged
- Generous negative space OR controlled density — never the mushy middle
- Floating navbars: add `top-4 left-4 right-4` spacing, never flush `top-0`
- Account for fixed navbar height so content doesn't hide behind it
- Breakpoints to test: 375px, 768px, 1024px, 1440px
- No horizontal scroll on mobile — ever

### Backgrounds & Visual Depth

- Create atmosphere with gradient meshes, noise textures, layered transparencies, dramatic shadows
- Avoid flat solid color backgrounds unless it's a strict minimalist theme
- Match textural depth to the aesthetic (dark industrial = grain + subtle glow; clean = crisp shadows only)

---

## Accessibility (CRITICAL — Non-Negotiable)

- Color contrast: **minimum 4.5:1** for normal text
- All interactive elements: visible focus rings
- All meaningful images: descriptive `alt` text
- Icon-only buttons: `aria-label` required
- Tab order must match visual order
- Form inputs must have `<label>` with `for` attribute
- Touch targets: **minimum 44×44px**
- `cursor-pointer` on every clickable/hoverable element

---

## Icons & Visual Elements

- **No emojis as UI icons** — use SVG icon sets (Heroicons, Lucide, etc.)
- Consistent icon sizing: fixed `viewBox="0 0 24 24"` with explicit `width`/`height`
- Hover states must not cause layout shift — use `color`, `opacity`, `box-shadow` only
- Smooth transitions: `transition-colors duration-200` or equivalent CSS

---

## Light/Dark Mode Contrast Rules

| Situation | Do | Don't |
|---|---|---|
| Glass card in light mode | `bg-white/80` or higher opacity | `bg-white/10` (invisible) |
| Body text in light mode | `#0F172A` or equivalent dark | `#94A3B8` (too light) |
| Borders in light mode | `border-gray-200` or equivalent | `border-white/10` (invisible) |

BeatCut is dark-mode-only — but these rules apply if a light variant is ever added.

---

## Pre-Delivery UI Checklist

Before delivering any UI code, verify:

**Visual**
- [ ] No emojis used as icons
- [ ] All icons from consistent set
- [ ] Hover states don't shift layout
- [ ] CSS variables used (not hardcoded hex duplicates)

**Interaction**
- [ ] All clickable elements have `cursor-pointer`
- [ ] Hover states give clear visual feedback
- [ ] Transitions are 150–300ms
- [ ] Focus states visible for keyboard nav

**Layout**
- [ ] No content hidden behind fixed elements
- [ ] Responsive at 375px, 768px, 1024px, 1440px
- [ ] No horizontal scroll on mobile

**Accessibility**
- [ ] Images have `alt` text
- [ ] Form inputs have labels
- [ ] Color is not the only indicator
- [ ] `prefers-reduced-motion` respected

---

## BeatCut-Specific UI Rules

- **Only one `<section>` visible at a time** — toggle `.active` class for stage transitions
- **File inputs are `opacity:0; position:absolute` inside `<label>`** — never add a second `click` handler on the drop zone
- **SSE via `EventSource`** — always `.close()` on done/error/cancel
- **`resetToUpload()`** must reset file inputs (DataTransfer trick), sliders, `selTransition` to `flash`, and hide polaroid entry dropdown
- **Polaroid entry dropdown** shown/hidden via `updatePolaroidVisibility()` based on `#sel-transition` value
- **`showDone(data)`** only called when `data.file` is present — never trigger on the 99% intermediate event
