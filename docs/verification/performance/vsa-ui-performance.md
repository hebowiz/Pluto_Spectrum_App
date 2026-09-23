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

## 2026-08-20: symbol-safe display decimation

### Symptom

When a PSK Result Range exceeded 2,000 symbols, the symbol overlay itself was
uniformly decimated. For pi/4-DQPSK this could alias with the alternating
constellation states and make the IQ trajectory overlay appear to contain only
four symbol clusters. Separately, peak-decimated time traces and the decimated
PSK IQ trajectory did not necessarily pass through the independently
interpolated symbol markers.

### Policy and implementation

The symbol overlay is no longer decimated: every result-range symbol is drawn.
This adds scatter points for long results but leaves captured IQ, DSP, decoded
symbols, and exports unchanged.

Time-domain power and frequency traces retain peak decimation, then add the
interpolated coordinates of every visible symbol. The PSK IQ trajectory retains
its 10,000-point base limit, but also includes both original waveform samples
that bracket every visible symbol time. The point limit is therefore a soft
limit when symbol display is enabled. This preserves the trace segment used by
the same interpolation that produces each symbol marker, so markers remain on
the visible trace without rendering the complete oversampled waveform.

### Verification

`tests/test_vsa_ui.py` verifies that more than 2,000 symbol points are retained,
required time-plot coordinates survive peak decimation, and both waveform
samples bracketing each required symbol time survive IQ-trajectory decimation.
