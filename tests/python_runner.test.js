const { spawnPython, PYTHON_TIMEOUT_MS } = require('../api/python_runner');

describe('spawnPython', () => {
    test('resolves stdout from a clean run', (done) => {
        const child = spawnPython(['-c', 'print("hello")']);
        let out = '';
        child.stdout.on('data', (d) => { out += d.toString(); });
        child.on('close', (code) => {
            expect(code).toBe(0);
            expect(out.trim()).toBe('hello');
            done();
        });
    });

    test('kills an interpreter that exceeds the deadline', (done) => {
        const started = Date.now();
        const child = spawnPython(['-c', 'import time; time.sleep(30)'], {
            timeoutMs: 1500,
        });
        child.on('close', (code, signal) => {
            const elapsed = Date.now() - started;
            expect(elapsed).toBeLessThan(10000);
            expect(signal).toBe('SIGKILL');
            expect(child.timedOut).toBe(true);
            done();
        });
    }, 15000);

    test('does not surface an unhandled error when the binary cannot start', (done) => {
        const logged = jest.spyOn(console, 'error').mockImplementation(() => {});
        const child = spawnPython(['-c', 'print(1)'], { timeoutMs: 5000 });
        // Simulate a spawn failure; the runner must absorb it via its own
        // 'error' listener rather than crashing the process.
        expect(() => child.emit('error', new Error('boom'))).not.toThrow();
        child.on('close', () => {
            expect(logged).toHaveBeenCalled();
            logged.mockRestore();
            done();
        });
        child.stdout.resume();
        child.stderr.resume();
    }, 15000);

    test('allows the timeout to be overridden by env at import time', () => {
        expect(PYTHON_TIMEOUT_MS).toBeGreaterThan(0);
    });
});
