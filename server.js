const express = require('express');
const app = express();
const port = 3000;

// Import your API routes
const apiRoutes = require('./api_routes.js');

// Middleware to parse JSON
app.use(express.json());

// Use the API routes
app.use('/api', apiRoutes);

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
