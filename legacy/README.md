# Legacy / experimental code

This directory holds code that is not part of the core Keke Excel tool runtime:

- `rtos/` — C scheduler experiment (not used by the web app)
- `boot/` — boot assembly stub
- `freertos_integration.py` — Python simulation of RTOS primitives

The production app uses `api/excel_processor.py` with on-disk session storage.
