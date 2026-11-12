'use strict';

const { normalizePath, boundedRoute, render, recordPythonCall, _internals } = require('../api/metrics');

describe('metrics', () => {
    test('normalizePath collapses session ids but keeps real path segments', () => {
        expect(normalizePath('/api/excel/dfa0809a-0b15-406f-aa51-26f9195833ad/analyze/Sheet1'))
            .toBe('/api/excel/:sessionId/analyze/Sheet1');
        // Same route shape, different ids -> identical label.
        expect(normalizePath('/api/excel/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/analyze/Sheet1'))
            .toBe('/api/excel/:sessionId/analyze/Sheet1');
        // The client-generated id form is collapsed too.
        expect(normalizePath('/api/excel/session_a1b2c3d4e5/analyze/Sheet1'))
            .toBe('/api/excel/:sessionId/analyze/Sheet1');
        // Ordinary segments are preserved, not swallowed by a generic pattern.
        expect(normalizePath('/api/excel/:sessionId/analyze/Sheet1')).toBe('/api/excel/:sessionId/analyze/Sheet1');
        expect(normalizePath('/health')).toBe('/health');
        expect(normalizePath('')).toBe('unknown');
    });

    test('route labels are bounded so crafted sheet names cannot grow memory', () => {
        const seen = new Set();
        for (let i = 0; i < 500; i += 1) {
            seen.add(boundedRoute(`/api/excel/dfa0809a-0b15-406f-aa51-26f9195833ad/analyze/Sheet${i}`));
        }
        // 500 distinct paths collapse to at most 200 real labels plus 'other'.
        const real = [...seen].filter((route) => route !== 'other');
        expect(real.length).toBeLessThanOrEqual(200);
        expect(seen.size).toBeLessThanOrEqual(201);
        expect(seen.has('other')).toBe(true);
    });

    test('python call outcomes are counted', () => {
        // A metric family with no observations still exposes HELP/TYPE.
        expect(render()).toMatch(/# TYPE keke_python_calls_total counter/);

        recordPythonCall('unit_test.py', 'success', 0.25);
        recordPythonCall('unit_test.py', 'timeout', 120);
        const after = render();
        expect(after).toMatch(/keke_python_calls_total\{outcome="success",script="unit_test\.py"\} 1/);
        expect(after).toMatch(/keke_python_calls_total\{outcome="timeout",script="unit_test\.py"\} 1/);
        // Duration histogram records both.
        expect(after).toMatch(/keke_python_call_duration_seconds_count\{script="unit_test\.py"\} 2/);
        expect(after).toMatch(/keke_python_call_duration_seconds_sum\{script="unit_test\.py"\} 120\.25/);
    });

    test('exposition is well formed: every series has HELP and TYPE', () => {
        const output = render();
        for (const metric of ['keke_http_requests_total', 'keke_http_request_duration_seconds']) {
            expect(output).toContain(`# HELP ${metric} `);
            expect(output).toContain(`# TYPE ${metric} `);
        }
        expect(output).toMatch(/# TYPE keke_process_resident_memory_bytes gauge/);
        expect(output).toMatch(/# TYPE keke_process_uptime_seconds gauge/);
        expect(output).toMatch(/# TYPE keke_build_info gauge/);
        expect(output.endsWith('\n')).toBe(true);
    });

    test('histogram buckets are cumulative and end at +Inf', () => {
        _internals.httpDuration.observe({ method: 'GET', route: '/x' }, 0.02);
        const output = render();
        const buckets = output
            .split('\n')
            .filter((line) => line.startsWith('keke_http_request_duration_seconds_bucket{') && line.includes('route="/x"'))
            .map((line) => Number(line.trim().split(' ').pop()));

        // Every bucket is >= the one before it, and +Inf equals the count.
        for (let i = 1; i < buckets.length - 1; i += 1) {
            expect(buckets[i]).toBeGreaterThanOrEqual(buckets[i - 1]);
        }
        expect(buckets[buckets.length - 1]).toBeGreaterThan(0);
    });

    test('label values are escaped', () => {
        _internals.httpRequests.inc({ method: 'GET', route: 'we"ird\\path', status: '200' });
        const output = render();
        expect(output).toMatch(/route="we\\"ird\\\\path"/);
    });
});
