const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const compression = require('compression');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const app = express();
const port = process.env.PORT || 3000;

// Import your API routes
const apiRoutes = require('./api_routes.js');

// Security middleware
app.use(helmet());
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    credentials: true
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// Logging middleware
app.use(morgan('combined'));

// Compression middleware
app.use(compression());

// Middleware to parse JSON with size limit
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        version: process.env.npm_package_version || '1.0.0',
        service: 'Keke Excel Datasheet Tool'
    });
});

// Root endpoint - serve the main interface
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Use the API routes
app.use('/api', apiRoutes);

// Global error handler.
//
// Every route funnels failures here via next(error). Most of them are Python
// helper failures whose message embeds the interpreter's stderr, i.e. a full
// traceback with absolute paths, e.g.:
//
//   Python process failed: Traceback (most recent call last):
//     File "/home/keke/api/excel_processor.py", line 884, in main
//       raise FileNotFoundError(f"Session not found: {session_id}")
//
// None of that may reach the client. Server-side failures are answered with a
// generic message plus a correlation id that is logged in full.
function clientSafeMessage(message) {
    const raw = String(message || '');
    if (/Traceback \(most recent call last\)/.test(raw)) {
        return 'Internal Server Error';
    }
    return raw.replace(/(?:\/[\w.@+-]+){2,}/g, '<path>');
}

function classifyError(err) {
    if (err.status && err.status < 500) {
        return { status: err.status, message: clientSafeMessage(err.message) };
    }
    if (err.code === 'LIMIT_FILE_SIZE' || err.code === 'LIMIT_UNEXPECTED_FILE') {
        return { status: 413, message: 'Uploaded file is too large.' };
    }
    const raw = String(err.message || '');
    if (/not found/i.test(raw)) {
        return { status: 404, message: 'The requested session, sheet, or column was not found.' };
    }
    if (/Unsupported |must be |Invalid |Unsupported format|Invalid file type/i.test(raw)) {
        return { status: 400, message: 'The request was invalid.' };
    }
    return { status: 500, message: null };
}

app.use((err, req, res, next) => { // eslint-disable-line no-unused-vars
    const errorId = crypto.randomBytes(6).toString('hex');
    const { status, message } = classifyError(err);

    console.error(`[error ${errorId}] ${req.method} ${req.originalUrl} -> ${status}:`, err);

    res.status(status).json({
        error: {
            message: message || 'The server failed to process the request.',
            status,
            error_id: errorId,
            timestamp: new Date().toISOString()
        }
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        error: {
            message: 'Route not found',
            status: 404,
            timestamp: new Date().toISOString()
        }
    });
});

app.listen(port, () => {
    console.log(`Keke Excel Datasheet Tool running at http://localhost:${port}`);
    console.log(`Health check available at http://localhost:${port}/health`);
    console.log(`Web interface available at http://localhost:${port}`);
});
