# VSA UI performance notes

## 2026-08-04: main-window move responsiveness

### Symptom

Dragging the offline VSA main window on Windows was visibly jerky even while no
analysis was running.

### Cause and policy

The IQ Power and instantaneous-frequency plots can each retain up to 100,000
line vertices. Windows may request repeated repaints while moving or exposing a
top-level window. Re-rendering every retained vertex for every repaint is not
useful when the plot viewport is only hundreds of pixels wide.

All VSA `PlotWidget` instances now use pyqtgraph's automatic downsampling with
the `peak` method and clip curves to the visible X range. Peak downsampling was
chosen so narrow power or frequency excursions remain visible; this is a
display-only optimization and does not change captured IQ, analysis results,
export data, or the numeric Symbol Table.

Dock animation is disabled. Dock widgets remain movable, floatable, closable,
and resizable, but Qt no longer continuously animates their layout transitions.

### Verification

`tests/test_vsa_ui.py` verifies the common plot policy and disabled dock
animation. Real Windows drag smoothness still requires visual verification on
the target PC because the offscreen Qt test backend cannot exercise the desktop
compositor.

### Follow-up if needed

If movement is still visibly slow, profile paint events on the target GPU and
Qt backend before enabling OpenGL. OpenGL is intentionally not enabled by this
change because driver-dependent Qt/OpenGL paths can introduce native crashes.
