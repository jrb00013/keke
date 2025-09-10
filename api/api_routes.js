const express = require('express');
const { body, param, query, validationResult } = require('express-validator');
const router = express.Router();

// Validation middleware
const validateRequest = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            error: {
                message: 'Validation failed',
                details: errors.array()
            }
        });
    }
    next();
};

// Stock symbol validation
const validateSymbol = param('symbol')
    .isLength({ min: 1, max: 10 })
    .matches(/^[A-Z]+$/)
    .withMessage('Symbol must be 1-10 uppercase letters');

// Enhanced stock data endpoint
router.get('/stock/:symbol', 
    validateSymbol,
    validateRequest,
    async (req, res, next) => {
        try {
            const symbol = req.params.symbol.toUpperCase();
            
            // Simulate API call with timeout
            const stockData = await fetchStockData(symbol);
            
            if (!stockData) {
                return res.status(404).json({
                    error: {
                        message: `Stock data not found for symbol: ${symbol}`,
                        status: 404
                    }
                });
            }
            
            res.json({
                symbol: symbol,
                data: stockData,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            next(error);
        }
    }
);

// Enhanced prediction endpoint
router.get('/predict/:symbol',
    validateSymbol,
    query('days').optional().isInt({ min: 1, max: 30 }).withMessage('Days must be between 1 and 30'),
    validateRequest,
    async (req, res, next) => {
        try {
            const symbol = req.params.symbol.toUpperCase();
            const days = parseInt(req.query.days) || 7;
            
            const prediction = await generatePrediction(symbol, days);
            
            res.json({
                symbol: symbol,
                prediction: prediction,
                confidence: prediction.confidence,
                prediction_date: new Date().toISOString(),
                forecast_days: days
            });
        } catch (error) {
            next(error);
        }
    }
);

// New endpoint: Get multiple stocks
router.post('/stocks/batch',
    body('symbols').isArray({ min: 1, max: 10 }).withMessage('Symbols must be an array of 1-10 items'),
    body('symbols.*').matches(/^[A-Z]+$/).withMessage('Each symbol must be uppercase letters'),
    validateRequest,
    async (req, res, next) => {
        try {
            const symbols = req.body.symbols.map(s => s.toUpperCase());
            const results = await Promise.allSettled(
                symbols.map(symbol => fetchStockData(symbol))
            );
            
            const stockData = results.map((result, index) => ({
                symbol: symbols[index],
                success: result.status === 'fulfilled',
                data: result.status === 'fulfilled' ? result.value : null,
                error: result.status === 'rejected' ? result.reason.message : null
            }));
            
            res.json({
                stocks: stockData,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            next(error);
        }
    }
);

// New endpoint: Get market summary
router.get('/market/summary', async (req, res, next) => {
    try {
        const summary = await getMarketSummary();
        res.json({
            summary: summary,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        next(error);
    }
});

// Helper functions (these would be moved to separate modules in production)
async function fetchStockData(symbol) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Mock data - in production, this would call a real API
    const mockData = {
        'AAPL': { price: 175.43, change: 2.15, changePercent: 1.24 },
        'GOOGL': { price: 142.56, change: -1.23, changePercent: -0.86 },
        'MSFT': { price: 378.85, change: 5.67, changePercent: 1.52 },
        'TSLA': { price: 248.50, change: -3.20, changePercent: -1.27 }
    };
    
    return mockData[symbol] || null;
}

async function generatePrediction(symbol, days) {
    // Simulate AI prediction delay
    await new Promise(resolve => setTimeout(resolve, 200));
    
    const basePrice = 150.0;
    const volatility = 0.02;
    const trend = 0.001;
    
    const prediction = basePrice * (1 + trend * days + (Math.random() - 0.5) * volatility * days);
    const confidence = Math.max(0.6, 1 - Math.random() * 0.4);
    
    return {
        predicted_price: Math.round(prediction * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
        trend: trend > 0 ? 'bullish' : 'bearish'
    };
}

async function getMarketSummary() {
    await new Promise(resolve => setTimeout(resolve, 150));
    
    return {
        market_status: 'open',
        total_volume: 45000000,
        gainers: 1250,
        losers: 980,
        unchanged: 320
    };
}

module.exports = router;
