const { spawn } = require('child_process');
const path = require('path');

// Every Python helper call used to spawn() directly with no upper bound, so a
// hung or wedged interpreter held the HTTP request open forever. This wrapper
// puts a hard deadline on every call and logs the failure server-side; the raw
// detail never reaches the client (see the error handler in server.js).
const PYTHON_TIMEOUT_MS = Number(process.env.PYTHON_TIMEOUT_MS) || 120000;

function spawnPython(args, options = {}) {
    const child = spawn('python3', args, options);
    const timeoutMs = options.timeoutMs || PYTHON_TIMEOUT_MS;
    const label = `${path.basename(args[0] || 'python3')} ${args[1] || ''}`.trim();

    const timer = setTimeout(() => {
        child.timedOut = true;
        console.error(`[python] timeout after ${timeoutMs}ms, killing: ${label}`);
        child.kill('SIGKILL');
    }, timeoutMs);
    timer.unref();

    child.on('error', (err) => {
        console.error(`[python] could not start ${label}:`, err.message);
    });
    child.on('close', (code, signal) => {
        clearTimeout(timer);
        if (child.timedOut) {
            console.error(`[python] killed after ${timeoutMs}ms timeout: ${label}`);
        } else if (code !== 0) {
            console.error(`[python] exit code=${code} signal=${signal}: ${label}`);
        }
    });

    return child;
}

module.exports = { spawnPython, PYTHON_TIMEOUT_MS };
