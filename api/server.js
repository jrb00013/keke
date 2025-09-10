const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const compression = require('compression');
const path = require('path');
const fs = require('fs');
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

// Global error handler
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(err.status || 500).json({
        error: {
            message: err.message || 'Internal Server Error',
            status: err.status || 500,
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
