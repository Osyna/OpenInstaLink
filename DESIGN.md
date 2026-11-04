# Link Camera interface

Linux desktop, Slint 1.18.1. Operate mode. A camera workspace, with the live image as the main content and task-specific controls beside it.

## Direction

The user-provided image is the visual reference: warm pale surfaces, generous round corners, soft offset depth, fine monochrome icons, mint/cyan highlights and clear type. Transfer that visual language, not its unrelated dashboard widgets, photography, personal names or numbers. This is a full redesign of the previous dark utility interface.

A fixed compact navigation rail leads to Camera, Effects, Views, Output, Script, AI tools and Settings. A persistent workspace keeps the preview and capture controls visible while the inspector changes. Profiles live with saved views rather than preceding basic picture controls. Native window decoration remains; no imitation operating-system chrome.

## Visual system

`ui/studio.slint` owns the palette, button states, icon treatment, corner radii and type roles. The canvas is warm gray-green, panels near-white, text deep evergreen, selected navigation pale mint. Cyan is reserved for focus and secondary emphasis. The camera image remains on a neutral dark stage, with its aspect ratio unchanged. Use restrained offset shadows for surface separation, not low-contrast neumorphic edges.

Use the platform sans-serif font, 13–14px body, 16–18px section headings and a 26–28px workspace title. Group related controls closely and separate sections generously. Keep numeric values next to sliders. Line icons share a 24px grid, 1.7px stroke, rounded caps and joins. Icons supplement labels.

Preferred window: 1440×900. Preserve useful layout and every action at 1280×800 and 900×640. The rail has a fixed footprint; preview flexes and inspectors scroll. Do not derive a pane's layout constraints from Window.width, which previously caused layout-info cycles.

Constrain vertical form content to the visible width with a scrollbar gutter; do not let preferred child widths silently create horizontal clipping. Stateless media actions are conditional, while editable pages remain persistent. Expanded error details can shrink the image stage so the full message remains inside the compact window.

## Interaction and state

Mouse, keyboard and accessibility activation follow the same callbacks. Pointer navigation moves focus before changing pages so a numeric edit commits on blur. Inspector pages retain their input state. Navigation clears armed destructive confirmations. Unsupported values remain unknown; errors include a recoverable action and their full details remain available.

Keep fixed per-section heading instances under the same visibility choice as their page. A dynamically replaced heading intermittently stayed stale in the software renderer; changing its string lookup to a ternary did not fix it.

Capture, recording, pause, stop and camera release stay visible and preserve all existing guards. Stop preview remains usable while camera controls are busy. A recording is finalized before closure. Never describe a merely detected device as an active video stream.

Optional AI stays off until enabled in Settings. Models are downloaded, listed and deleted from the AI tools page or the `models` commands, each in the app's own model directory; enabling AI or running a model never downloads, and the local library has a manual refresh. Inference runs in a managed local Python worker with internet sockets blocked. One exclusive resident-worker lease per model store prevents duplicate workers across windows/CLI; it survives delayed reaping and abrupt parent exit until the child really exits. Each engine admits one request at a time on the newest frame. Stop immediately invalidates pending results, ends live submission and asynchronously unloads the owned model, even idle, without deleting history. It restores the pre-analysis stage unless the user explicitly selected another view, and never starts physical capture. Busy/stopping remains latched through real cleanup and catalog refresh; repeated Stop is idempotent. Active cancellation, a deadline, disabling AI or closing the window terminates only owned work. The next request reloads a terminated model; late inference/history-render replies cannot re-freeze a stopped view. No design change resets saved hardware setup, preview format or effects.

The AI inspector has persistent Model, Analyze and History navigation. The model picker belongs to Model and Analyze; History spans all models. Model explanations separate purpose, useful cases, outputs, performance and limitations from expandable source/file details. Analyze reserves a fixed 34 px scrolling viewport each for availability and normal progress, even when empty; hiding the busy-availability line was the remaining source of frame-status jumping. Actual errors instead take their full wrapped height. Run/Live-on refuse duplicate or stopping work; Run once stays disabled while Live is enabled, and Stop uses owned-session/worker state rather than result presence. History has full-height Browse, Render and Raw views, not a short raw editor squeezed beneath a list.

Keep the latest 100 completed analyses total in a RAM-only ring: losslessly compressed source frames and complete shared raw output bytes, not 100 decoded images. Encoding and restoring older analyses run off the UI thread; only one restore runs, with the newest pending selection replacing obsolete requests. Pinning a result may retain it beyond ring eviction. Opening Raw, focusing a reader, choosing an entry or scrolling back in Browse pauses following without stopping inference; new rows preserve the scrolled list anchor. Follow latest explicitly resumes view updates. Clear requires confirmation, leaves exports and the Follow choice alone, and does not stop new analysis. Model/camera/format/effects changes cancel obsolete requests but preserve completed source-matched history. AI off and window close discard it. The buffer's byte readout excludes model/GPU and decoded display memory; lossless 4K history can still use several GB.

The image, frozen SAM frame and pointer surface share the same rectangle and contain fit. Never move overlays independently or put old inference geometry on a newer frame. The teleprompter remains UI-only.

## Range and switch controls

`RangeSlider` and `Switch` in `ui/studio.slint` replace the stock slider and checkbox. Both are drawn by the app, so their input and accessibility behavior is specified here and not inherited from the toolkit.

A slider edits a bounded value and keeps its numeric readout beside it. Dragging or pressing the track updates the value live. A drag is sent to hardware on release. A plain press without a drag is sent after 500 ms, so a double-click reset cannot race the first click's write. Wheel deltas move in steps and small touchpad deltas accumulate; pushing past an end passes the wheel on so the page keeps scrolling. Arrow keys move one step, Page Up and Page Down ten, Home and End go to the ends, and each of these is sent at once. Double-click, Backspace, Delete and a reset button beside the readout request the exact default, including values outside the editing grid such as a 33 ms model interval. Camera defaults come from the device's V4L2 metadata. When no default is known nothing is written and the control says so. The reset button is disabled while the value already equals the default, and the default is drawn as a tick above the rail.

A switch is for settings that apply immediately. A pointer press, Space and Return each toggle it once, a pointer press first moves focus so an open numeric edit commits, and the state shows as thumb position and a check mark as well as colour. Both controls expose role, enabled state and label to assistive technology. The slider also exposes value, minimum, maximum and step, accepts set-value, increment and decrement actions, and describes its default and how to reset it. Reset buttons are labelled "Reset NAME to its default" (or "to the camera default" on the camera page). Keyboard focus draws a ring around the slider grip and around the switch track. Native pointer, keyboard and saved-state checks were exercised on a private X11 display; screen-reader and physical-touchpad behavior was not checked.

## Grounding

Native principles: `layout.md › Visual hierarchy`, `typography.md › Conveying hierarchy`, `accessibility.md › Vision`, `buttons.md › Style`, `sidebars.md › Desktop`, `scroll-views.md › Best practices`, and `sliders.md › Best practices` in signet's Apple HIG reference set. These guide native grouping and input behavior; the supplied reference controls the visual direction.
