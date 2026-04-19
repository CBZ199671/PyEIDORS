# EIT Workstation Design System

> 电阻抗断层成像工作站 · Design System for the EIT (Electrical Impedance Tomography) Workstation desktop application.

The **EIT Workstation** (EIT 工作站) is a bilingual (Simplified Chinese / English), cross‑platform desktop app built on **PySide6 (Qt 6)** that supports the full EIT research workflow:

- **实测 (Hardware)** — Connect to a real EIT board over serial or 4G relay, set excitation / measurement parameters, acquire and record frames.
- **仿真 (Simulation)** — Define a mesh + electrodes, drop in non‑homogeneous regions, solve the forward problem, then reconstruct via various inverse methods.
- **数据集 (Dataset Generator)** — Batch‑generate synthetic σ / boundary‑voltage sample pairs for ML training.
- **数据库 (Database)** — Browse, filter and re‑reconstruct historical recordings.

The UI is a **modern engineering‑tool aesthetic** — dense, data‑first, calm, serious, with a steel‑blue accent and a light ↔ dark theme pair. Visually it sits closer to scientific instrumentation (think LabVIEW / MATLAB / National Instruments) than to a consumer SaaS app.

---

## Sources

- **Codebase:** the `eit_app/` Python/PySide6 package (mounted locally for design‑time reference). Design tokens, QSS stylesheets and component behaviour are all lifted directly from this source — especially `eit_app/ui/theme.py` (two full stylesheets, light + dark overlay), `eit_app/ui/fonts.py` (Times New Roman + CJK fallback), and `eit_app/i18n/{zh,en}.py` (all user‑facing copy).
- **No Figma / no slide deck / no logo files were provided.** Iconography is inferred from code — the app uses **no custom icons / SVG / emoji** anywhere in the UI chrome. All affordances are text‑label–driven. See *Iconography* below.

---

## Products represented

There is exactly **one product**: the EIT Workstation desktop app. Its four tabs are four *surfaces of the same product*, not separate products.

The UI kit at `ui_kits/eit_workstation/` recreates all four tabs with interactive click‑through behaviour.

---

## Content Fundamentals

### Language & casing

- **Primary language: Simplified Chinese (zh‑CN).** English is a full peer via `menu.language`. Both share the same string‑key structure in `eit_app/i18n/{en,zh}.py`.
- **Chinese copy style** (from `i18n/zh.py` header):
  - Imperative verbs for user actions: 「开始采集」, not 「开始采集一帧」
  - Menu / button mnemonics as `(&X)` with a capital letter, e.g. 「文件(&F)」, 「退出(&X)」
  - No redundant adverbs: 「已完成」, not 「已经完成」
  - Tab labels and step names are **2–4 character nouns** with parallel structure: 实测 · 仿真 · 数据集 · 数据库.
- **English copy style:** **Title Case for buttons, menus, and section headers** ("Start Acquisition", "Run Forward"). Sentence case for hints and status text.

### Pronouns & voice

- Chinese side avoids second‑person pronouns entirely — copy addresses the *action*, not the *user* (「请先选择传输方式并验证设备连接。」).
- English side uses implicit second person ("Select a transport method, then click Connect"). Never first person. Never "we".
- **Neutral, precise, technical.** Zero marketing voice. No exclamation marks outside of error banners.

### Section headers & hints

Every workflow panel opens with a **hint line** right under the title — one clear sentence stating the purpose of the panel. Examples:

- `hw.connection.flow_hint` → 「请先选择传输方式并验证设备连接。」 / *"Select a transport method, then verify the device link."*
- `hw.acquisition.flow_hint` → 「请设置保存路径和采集计划，然后启动采集。」
- `sim.inhom.title` / `sim.inhom.hint` pattern repeats across all four simulation steps.

Step headers use the form: 「步骤一 · 连接」 (「Step N · Noun」). The middle dot `·` is the canonical separator — never `-`, never `—`.

### State banners

Status is surfaced via **tone‑coded banners** with a strict three‑part anatomy: **title · detail · action**. E.g. the Hardware session summary has distinct `link_down / fault / verifying / acquiring / ready / ready_record_armed / link_verified …` states, each with all three strings defined. Copy is always actionable: after every "what's happening" sentence comes a "what to do next" sentence.

### Tone

Calm, matter‑of‑fact, lab‑notebook register. It sounds like firmware release notes, not a product announcement. No emoji. No "oops!", no "we couldn't find that". Failures read like `main.status.recon_failed → 重构失败：{error}`.

---

## Visual Foundations

### Palette

Two themes, **light** (default) and **dark** (overlay). Both sit on a **blue‑grey, slightly cool** canvas — never pure white, never pure black.

| Token | Light | Dark |
|---|---|---|
| Canvas | `#eef3f8` | `#1a1f26` |
| Panel / GroupBox body | `#ffffff` | `#222831` |
| Input fill | `#ffffff` | `#2a313a` |
| Border | `#e0e6ee` / `#d0d9e3` | `#3e4754` |
| Accent (primary) | `#1f5d8b` | `#5ca8e0` |
| Accent hover | `#2a6fa0` / `#1a5078` | `#226a9b` |
| Text (primary) | `#243447` | `#dbe1ea` |
| Text (muted) | `#5b6573` | `#8b97a7` |
| Section header text | `#1f3b5b` | `#9dc9ea` |
| GroupBox title text | `#1f5d8b` | `#8fc8ea` |
| Success | `#1f7a52` | `#1e7a52` |
| Danger | `#8b2f2f` | `#7a3a3a` |
| Warn fg (tone.warn) | `#8a4b08` | `#f3c97a` |

There is also a **full "tone palette"** for state chips and banners — `idle / warn / ready / active / error`, each with (fg, bg, border) per theme. See `colors_and_type.css`.

Plot palettes are separate (`_PLOT_PALETTE_LIGHT/_DARK` in `theme.py`) because matplotlib and pyqtgraph don't honor QSS — same family of blues and neutrals, but with extra tokens for `domain` outlines, `electrode` markers, and `highlight` overlays.

### Type

- **UI chrome:** Segoe UI → Noto Sans → DejaVu Sans + CJK fallback chain (Microsoft YaHei → Noto Sans CJK SC → Source Han Sans SC → PingFang SC → WenQuanYi Zen Hei → SimSun → SimHei). Base size: **10pt** (`QFont.setPointSize(10)`) which renders at ~13 px on standard DPI.
- **Plots & engineering labels:** `Times New Roman` (serif) — bundled from `C:\Windows\Fonts\times.ttf` when running on Windows, else picks the best available serif from *Liberation Serif / Nimbus Roman / DejaVu Serif*. Matplotlib's `font.family` is set to a list so per‑glyph CJK fallback works correctly.
- **Type scale (from QSS):** base 13 px, `QHeaderView` / `QToolBox#workflowToolbox::tab` 12 px 700 uppercase with `letter-spacing: 0.5px`, `QTabBar::tab` 13 px 600 / 700‑when‑selected, `QGroupBox::title` 12 px 700 uppercase‑ish with `letter-spacing: 0.3px`.
- **Weights used:** 600 (section headers, tabs), 700 (active tab, group titles, table headers, button labels).
- **Mono:** not used in the chrome. Terminal / log output, if any, inherits the serif family.

### Spacing & layout

- **Borders: 1 px** on panels, **1.5 px** on checkboxes, **2 px** for focus rings.
- **Corner radii are deliberate and varied by element:**
  - Inputs, buttons, tab tops: **8 px**
  - GroupBox, dock title, menus, status banners: **10 px**
  - Info cards, next‑action banner, embedded step panels: **8–12 px**
  - Checkbox indicators: **4 px**
  - Tone chips: **8 px** (compact) / **10 px** (default)
- **Input padding:** 6–7 px vertical × 10 px horizontal; buttons 8 × 16 px.
- **Group boxes** always have `margin-top: 14px; padding: 12px 12px 10px 12px` so the title sits cleanly in the top border.
- **Step panels** (`embeddedStepPanel="true"`) are tighter: 6 × 7 × 5 × 7 px padding, 7 px radius — they live inside the left `QToolBox` rail.
- Layout is **three‑column** on every tab: left workflow rail (QToolBox with step 1 / step 2 / step 3) → center canvas (plots, 3D views, tables) → right utility panel (frame browser, metrics, summary).

### Surfaces & elevation

- **No drop shadows anywhere.** Qt QSS doesn't do `box-shadow`; the design leans on **1 px borders + slightly tinted fills** for hierarchy.
- **Three brightness levels per theme** map to three depths of UI:
  - Canvas ← Panels / GroupBoxes ← Input fills. Each step is ~4–8 L* units apart — enough to read, gentle enough to feel flat.
- **No transparency, no blur** — the UI is fully opaque everywhere. The only "layering" is the loading/error **scrim overlay** which paints a full‑panel opaque `panel_bg` over stale plot content while new data loads.

### Backgrounds & imagery

- **No background images. No gradients. No textures.** The canvas is a flat tint. This is deliberate — the design wants all attention on numeric data and plots.
- Plots have their own subtly brighter canvas (`#fbfdff` / `#161b22`) vs. the surrounding panel, so the chart area reads as "the subject" and the panel chrome recedes.

### Animation & motion

- **Essentially zero motion.** No transitions on hover, no fade‑ins, no slide‑outs. State changes are instantaneous.
- The only thing that "moves" is the `QProgressBar::chunk` (native Qt animation) and streamed plot data.
- Rationale: this is an instrumentation tool. Jitter reduces trust in the readout.

### Hover / press / focus / disabled

- **Hover:** fills shift one step warmer/cooler, borders move from neutral grey to one of `{#b1c2d3, #5d6a7a}` depending on theme. Primary buttons on hover: `#1a5078` (light) / `#226a9b` (dark).
- **Press:** buttons get an **asymmetric pad tweak** (`padding-top: 9px; padding-bottom: 7px`) so the label nudges down 1 px — a physical "click" feel without any transform. Tables/lists use `#e4eef9` hover fill.
- **Focus (keyboard):** a **2 px accent border** replacing the 1 px neutral border, with padding reduced by 1 px to keep geometry stable. Rings are tinted per button variant (primary gets `#9fc8e4` in light mode; danger gets `#e8b7b4`).
- **Disabled:** `color: #6c7a8a` on `background: #eef2f7 / border: #d3dbe4` — all variants individually defined in QSS. The disabled palette was tuned to pass WCAG 2.1 AA (4.5:1) — see comment in `theme.py` L~1015.

### Charts

- **pyqtgraph + matplotlib** for all scientific visualization — EIT conductivity images, 3D tetrahedral meshes, boundary voltage plots, equipotential contours.
- Viridis colormap for σ (conductivity). Amber `#f39c12` (light) / pumpkin `#ffa94d` (dark) for inhomogeneity highlight overlay.
- Grid lines are quiet (`#d6e1ec` / `#2f3742`) and axes live at `#243447` / `#dbe1ea`.

### Iconography → see section below

---

## Iconography

- **There are no icons.** Zero SVG icons. Zero icon fonts. Zero emoji. Zero unicode pictographs.
- Every affordance is a **text label** — buttons say 「开始」/"Start", 「浏览…」/"Browse…", 「设为参考」/"Set as reference". Menu items use Qt mnemonic underlines (`(&F)`, `(&X)`) for keyboard access.
- The only painted glyphs anywhere in the UI are:
  - **Inline SVG triangle arrows** for `QSpinBox` / `QDateEdit` step buttons, built in code via `_arrow_data_url()` in `theme.py`. These are the up/down `◢◣` direction indicators on numeric inputs. Two colors each, for light vs dark.
  - **Qt's native QTabBar::tab "selected" underline** (a 3 px accent rule at the bottom of the active tab).
- **Middle dot `·`** is used as a separator in step names (「步骤一 · 连接」).
- The app therefore has **no logo** baked into the UI itself — the window title reads literally 「EIT 工作站」 (i.e. the product's name *is* the brand mark). For this design system we synthesize a simple wordmark treatment (see `assets/logo.svg`) as a reusable slug for doc headers and splash contexts.

**For this design system, when building mockups you should:**

1. **Prefer text labels over icons** wherever possible. It matches the source product.
2. If an icon is genuinely unavoidable (e.g. a close‑X in a custom dialog), use a **stroke‑style, 1.5 px weight, neutral grey (`#5b6573` / `#a7b2c2`)** glyph. Substitute closest match from **Lucide** (via CDN) and flag as a substitution — we have no evidence the product itself uses Lucide.
3. Never use emoji.
4. Never use colorful pictogram cards (the anti‑pattern: a grid of rounded squares with coloured icons).

---

## Index

```
README.md                 — this document
SKILL.md                  — Agent Skill manifest (cross‑compatible with Claude Code)
colors_and_type.css       — CSS custom properties: colors (light + dark), type, spacing, radii, tones
fonts/                    — (empty — fonts come from the system; see note below)
assets/
  logo.svg                — synthesized wordmark slug for the app
preview/                  — small HTML specimen cards rendered in the Design System tab
ui_kits/
  eit_workstation/
    index.html            — full interactive recreation of the 4‑tab desktop app
    README.md             — notes on component coverage
    components/*.jsx      — React components (WorkflowRail, GroupPanel, Buttons, Inputs, Tabs …)
```

### Font note (substitution flag)

**The original Windows fonts (Segoe UI, Times New Roman, Microsoft YaHei) are not bundled** — they are system fonts the Python app picks up via `QFontDatabase` at runtime. For HTML design work we substitute with the **nearest Google Fonts equivalents:**

| Original (Qt runtime) | Web substitute | Role |
|---|---|---|
| Segoe UI | **Noto Sans SC** (incl. Latin) | UI chrome |
| Microsoft YaHei | **Noto Sans SC** | CJK glyphs |
| Times New Roman | **Noto Serif SC** / generic `Times, serif` | Plot axes, engineering labels |

This substitution is flagged here so you know to swap fonts in when handing off to production. If you can supply licensed Segoe UI / Microsoft YaHei / Times New Roman webfont files, drop them into `fonts/` and we'll update the CSS imports.
