---
name: eit-workstation-design
description: Use this skill to generate well-branded interfaces and assets for the EIT Workstation (EIT 工作站 · Electrical Impedance Tomography desktop app), either for production or throwaway prototypes/mocks/etc. Contains essential design guidelines, colors, type, fonts, assets, and UI kit components for prototyping.
user-invocable: true
---

Read the `README.md` file within this skill, and explore the other available files:

- `colors_and_type.css` — drop-in CSS custom properties for all tokens (light + dark)
- `preview/*.html` — swatch / specimen cards showing every token in use
- `ui_kits/eit_workstation/` — full React + Babel interactive recreation of the 4-tab Qt app (Hardware · Simulation · Dataset · Database). `components/*.jsx` are the primitives.
- `assets/` — synthesized wordmark slug (the original app has no logo — title text is the brand)
- `fonts/` — intentionally empty; chrome substitutes **Noto Sans SC / Noto Serif SC** (Google Fonts) for the Windows system fonts (Segoe UI / Microsoft YaHei / Times New Roman) that the Qt app picks up at runtime

**Hard design rules for this brand:**

1. **No icons, no emoji, no unicode pictographs.** Every affordance is a text label in Chinese *and* English.
2. Copy is bilingual — Simplified Chinese first, with English as a peer. Tab names are 2–4 character Chinese nouns in parallel structure. Use the middle dot `·` as a separator.
3. Tone is calm, lab-notebook, firmware-release-notes. No marketing voice. No first person. Chinese side avoids second-person pronouns.
4. No drop shadows, no gradients, no background images, no transparency, no blur, no animation. Hierarchy comes from 1px borders + slightly tinted fills.
5. Every workflow panel has: title → one-line hint → fields → action row → state banner (title · detail · action).
6. State banners follow the tone system: `idle / warn / ready / active / error` — each has a (fg, bg, border) triple defined as CSS vars.
7. Three-column layout on every surface: left workflow rail (QToolBox steps + summary) → center canvas → (optional) right utility pane.

**If creating visual artifacts** (slides, mocks, throwaway prototypes, etc), copy assets out of this skill and create static HTML files that import `colors_and_type.css` directly. Reuse the UI kit components by copying the JSX files — they're deliberately dependency-free.

**If working on production code**, read the source of truth at `eit_app/ui/theme.py` (QSS base + `_DARK_OVERLAY`), `eit_app/ui/fonts.py`, and `eit_app/i18n/{zh,en}.py`. The CSS vars here are a 1:1 port; any deviation is a bug in this skill, not the app.

**If the user invokes this skill without any other guidance**, ask them what they want to build or design, ask some clarifying questions (which of the four tabs? what workflow step? any specific state/error to show? print asset or interactive?), and act as an expert designer who outputs HTML artifacts *or* production code, depending on the need.
