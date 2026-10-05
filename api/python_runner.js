const { spawn } = require('child_process');
const path = require('path');
const { recordPythonCall } = require('./metrics');

// Every Python helper call used to spawn() directly with no upper bound, so a
// hung or wedged interpreter held the HTTP request open forever. This wrapper
// puts a hard deadline on every call and logs the failure server-side; the raw
// detail never reaches the client (see the error handler in server.js).
const PYTHON_TIMEOUT_MS = Number(process.env.PYTHON_TIMEOUT_MS) || 120000;

function spawnPython(args, options = {}) {
    const child = spawn('python3', args, options);
    const timeoutMs = options.timeoutMs || PYTHON_TIMEOUT_MS;
    const script = path.basename(args[0] || 'python3');
    const label = `${script} ${args[1] || ''}`.trim();
    const startedAt = process.hrtime.bigint();
    const elapsed = () => Number(process.hrtime.bigint() - startedAt) / 1e9;

    const timer = setTimeout(() => {
        child.timedOut = true;
        console.error(`[python] timeout after ${timeoutMs}ms, killing: ${label}`);
        child.kill('SIGKILL');
    }, timeoutMs);
    timer.unref();

    child.on('error', (err) => {
        console.error(`[python] could not start ${label}:`, err.message);
        recordPythonCall(script, 'error', elapsed());
    });
    child.on('close', (code, signal) => {
        clearTimeout(timer);
        if (child.timedOut) {
            console.error(`[python] killed after ${timeoutMs}ms timeout: ${label}`);
            recordPythonCall(script, 'timeout', elapsed());
        } else if (code !== 0) {
            console.error(`[python] exit code=${code} signal=${signal}: ${label}`);
            recordPythonCall(script, 'error', elapsed());
        } else {
            recordPythonCall(script, 'success', elapsed());
        }
    });

    return child;
}

module.exports = { spawnPython, PYTHON_TIMEOUT_MS };
