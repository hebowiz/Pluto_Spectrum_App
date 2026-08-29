# Graceful application shutdown

Updated: 2026-08-29

## Purpose

VSA and VSG no longer reject a window-close request merely because an ADALM-Pluto
operation is active. Closing starts an orderly shutdown and the window closes
automatically after every worker that can access Pluto or Qt-owned state has
finished.

## VSA sequence

1. Mark the generic and standard-specific workspaces as shutting down.
2. Discard queued re-analysis requests and reject new analysis requests.
3. Cancel an active Pluto capture. An analysis already executing is allowed to
   finish because its current DSP implementation is not cooperatively cancellable.
4. Stop ADS-B capture, stream analysis, and route lookup workers.
5. Poll worker state with a Qt timer so the GUI thread remains responsive.
6. Save the startup measurement configuration, close the shared Pluto source once,
   and accept the close event.

Device discovery and aircraft-database work are not force-terminated; shutdown
waits for them asynchronously. This avoids deleting a live `QThread`, which can
cause native Qt/Shiboken crashes.

## VSG sequence

1. If finite Pluto TX is active, request the existing backend stop operation once.
2. Wait for the TX worker's normal cleanup and mute path to complete, then close.
3. If Pluto preparation or explicit calibration is active, do not interrupt the
   calibration transaction. Wait for it to complete and close automatically.

The status bar reports the shutdown state. Repeated window-close operations do not
issue duplicate TX stop requests.

## Maintenance rule

Any new Pluto, DSP, discovery, database, or network worker owned by these windows
must be included in `request_shutdown()` and `shutdown_busy_reason()`. Never close
the shared Pluto source or destroy a window while one of its workers can still use
it.
