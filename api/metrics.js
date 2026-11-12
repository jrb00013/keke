'use strict';

// Minimal Prometheus metrics for the Keke app.
//
// The monitoring configs scraped /metrics, /node-metrics and /freertos-metrics,
// none of which the server implemented, so every target was permanently down.
// This module provides a real /metrics endpoint in the Prometheus text exposition
// format without pulling in a client library.
//
// Deliberately dependency-free: prom-client is not in the dependency tree, and
// the handful of series needed here does not justify adding it.

const COUNTER = 'counter';
const HISTOGRAM = 'histogram';
const GAUGE = 'gauge';

const DEFAULT_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300];

function escapeLabelValue(value) {
    return String(value).replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/"/g, '\\"');
}

function formatLabels(labels) {
    const keys = Object.keys(labels).filter((key) => labels[key] !== undefined);
    if (keys.length === 0) {
        return '';
    }
    const parts = keys
        .sort()
        .map((key) => `${key}="${escapeLabelValue(labels[key])}"`);
    return `{${parts.join(',')}}`;
}

class Metric {
    constructor(type, name, help) {
        this.type = type;
        this.name = name;
        this.help = help;
        this.values = new Map();
    }

    keyFor(labels) {
        return Object.keys(labels)
            .sort()
            .map((key) => `${key}=${labels[key]}`)
            .join(',');
    }
}

class Counter extends Metric {
    constructor(name, help) {
        super(COUNTER, name, help);
    }

    inc(labels = {}, value = 1) {
        const key = this.keyFor(labels);
        const existing = this.values.get(key);
        if (existing) {
            existing.value += value;
        } else {
            this.values.set(key, { labels, value });
        }
    }

    render() {
        const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} counter`];
        for (const entry of this.values.values()) {
            lines.push(`${this.name}${formatLabels(entry.labels)} ${entry.value}`);
        }
        return lines;
    }
}

class Gauge extends Metric {
    constructor(name, help) {
        super(GAUGE, name, help);
    }

    set(labels, value) {
        this.values.set(this.keyFor(labels), { labels, value });
    }

    inc(labels = {}, value = 1) {
        const key = this.keyFor(labels);
        const existing = this.values.get(key);
        if (existing) {
            existing.value += value;
        } else {
            this.values.set(key, { labels, value });
        }
    }

    render() {
        const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} gauge`];
        for (const entry of this.values.values()) {
            lines.push(`${this.name}${formatLabels(entry.labels)} ${entry.value}`);
        }
        return lines;
    }
}

class Histogram extends Metric {
    constructor(name, help, buckets = DEFAULT_BUCKETS) {
        super(HISTOGRAM, name, help);
        this.buckets = [...buckets].sort((a, b) => a - b);
    }

    observe(labels = {}, value = 1) {
        const key = this.keyFor(labels);
        let entry = this.values.get(key);
        if (!entry) {
            entry = {
                labels,
                counts: new Array(this.buckets.length).fill(0),
                sum: 0,
                count: 0
            };
            this.values.set(key, entry);
        }
        entry.sum += value;
        entry.count += 1;
        for (let i = 0; i < this.buckets.length; i += 1) {
            if (value <= this.buckets[i]) {
                entry.counts[i] += 1;
            }
        }
    }

    render() {
        const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} histogram`];
        for (const entry of this.values.values()) {
            // Prometheus expects cumulative counts, which is what observe()
            // already maintains, so emit them in ascending bucket order.
            for (let i = 0; i < this.buckets.length; i += 1) {
                lines.push(
                    `${this.name}_bucket${formatLabels({ ...entry.labels, le: this.buckets[i] })} ${entry.counts[i]}`
                );
            }
            lines.push(`${this.name}_bucket${formatLabels({ ...entry.labels, le: '+Inf' })} ${entry.count}`);
            lines.push(`${this.name}_sum${formatLabels(entry.labels)} ${entry.sum}`);
            lines.push(`${this.name}_count${formatLabels(entry.labels)} ${entry.count}`);
        }
        return lines;
    }
}

// Collapse identifiers so a metric cannot explode in cardinality. Session ids
// are the only unbounded path component the server itself generates, so they
// are the only thing rewritten here. Ordinary segments such as "analyze" or
// "Sheet1" are kept: they are a small, fixed set and are useful in the label.
function normalizePath(reqPath) {
    if (!reqPath) {
        return 'unknown';
    }
    return String(reqPath)
        .split('?')[0]
        .replace(/\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=\/|$)/gi, '/:sessionId')
        .replace(/\/session_[0-9a-z]{4,}(?=\/|$)/gi, '/:sessionId')
        || '/';
}

// Sheet names come from uploaded files, so a crafted upload could otherwise add
// unbounded distinct route labels. Past this many, extra routes collapse into
// a single bucket.
const MAX_ROUTE_LABELS = 200;
const seenRoutes = new Set();

function boundedRoute(reqPath) {
    const route = normalizePath(reqPath);
    if (seenRoutes.has(route)) {
        return route;
    }
    if (seenRoutes.size >= MAX_ROUTE_LABELS) {
        return 'other';
    }
    seenRoutes.add(route);
    return route;
}

const httpRequests = new Counter(
    'keke_http_requests_total',
    'Total HTTP requests handled, by method, normalized route and status code.'
);
const httpDuration = new Histogram(
    'keke_http_request_duration_seconds',
    'HTTP request duration in seconds, by method and normalized route.',
    [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120]
);
const pythonCalls = new Counter(
    'keke_python_calls_total',
    'Python helper invocations, by script and outcome (success, error or timeout).'
);
const pythonDuration = new Histogram(
    'keke_python_call_duration_seconds',
    'Python helper duration in seconds, by script.'
);
const residentMemory = new Gauge(
    'keke_process_resident_memory_bytes',
    'Resident set size of the Node process in bytes.'
);
const uptime = new Gauge(
    'keke_process_uptime_seconds',
    'Uptime of the Node process in seconds.'
);
const buildInfo = new Gauge(
    'keke_build_info',
    'Build information; always 1. Labelled with version and node version.'
);

// Records one Python helper invocation. Called from python_runner.js.
function recordPythonCall(script, outcome, durationSeconds) {
    pythonCalls.inc({ script, outcome });
    if (typeof durationSeconds === 'number' && Number.isFinite(durationSeconds)) {
        pythonDuration.observe({ script }, durationSeconds);
    }
}

function collectGauges() {
    const memory = process.memoryUsage();
    residentMemory.set({}, memory.rss);
    uptime.set({}, process.uptime());
    buildInfo.set(
        { version: process.env.npm_package_version || '1.0.0', node: process.version },
        1
    );
}

// Express middleware. Records every completed request, including 404s and
// errors, because it hooks the response 'finish' event rather than wrapping
// the handler.
function metricsMiddleware(req, res, next) {
    // Never count scrapes of the endpoint itself.
    if (req.path === '/metrics') {
        return next();
    }

    const startedAt = process.hrtime.bigint();

    res.on('finish', () => {
        const seconds = Number(process.hrtime.bigint() - startedAt) / 1e9;
        const labels = {
            method: req.method,
            route: boundedRoute(req.path),
            status: String(res.statusCode)
        };
        httpRequests.inc(labels);
        httpDuration.observe({ method: labels.method, route: labels.route }, seconds);
    });

    return next();
}

function render() {
    collectGauges();
    return [
        httpRequests.render(),
        httpDuration.render(),
        pythonCalls.render(),
        pythonDuration.render(),
        residentMemory.render(),
        uptime.render(),
        buildInfo.render()
    ]
        .flat()
        .join('\n') + '\n';
}

module.exports = {
    metricsMiddleware,
    render,
    recordPythonCall,
    normalizePath,
    boundedRoute,
    // Exported for tests.
    _internals: { httpRequests, pythonCalls, httpDuration, Counter, Gauge, Histogram }
};
